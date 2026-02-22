# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import ifeval

import langdetect

import nltk
import numpy as np
import pandas as pd
from scipy.stats import bootstrap


logger: logging.Logger = logging.getLogger(__name__)



def gen_acc_strict(x: Dict[str, Any]) -> Dict[str, float]:
    # reference: fbcode/gen_ai/github/fair_evals/evals/tasks/finetune/ifeval.py
    response = str(x["response"])
    instruction_list = x["instruction_id_list"]
    is_following_list = []
    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = ifeval.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        instruction.build_description(**x["kwargs"][index])
        if response and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return {
        "follow_instruction_list": is_following_list,
        "instruction_id_list": instruction_list,
    }


def gen_acc_loose(x: Dict[str, Any]) -> Dict[str, float]:
    response = str(x["response"])
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    instruction_list = x["instruction_id_list"]
    is_following_list = []
    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = ifeval.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        instruction.build_description(**x["kwargs"][index])

        is_following = False
        for r in all_responses:  # type: ignore
            if r.strip() and instruction.check_following(r):  # type: ignore
                is_following = True
                break

        is_following_list.append(is_following)
    return {
        "follow_instruction_list": is_following_list,
        "instruction_id_list": instruction_list,
    }


def parse_result(outputs: List[Dict[str, Any]]) -> Tuple[float, float]:

    prompt_total = 0
    prompt_correct = 0
    instruction_total = 0
    instruction_correct = 0

    for example in outputs:
        follow_instruction_list = example["follow_instruction_list"]
        instruction_id_list = example["instruction_id_list"]

        prompt_total += 1
        if all(follow_instruction_list):
            prompt_correct += 1

        instruction_total += len(instruction_id_list)
        instruction_correct += sum(follow_instruction_list)

    return prompt_correct / prompt_total, instruction_correct / instruction_total


def parse_result_no_reduce(outputs: List[Dict[str, Any]]) -> Tuple[List, List]:

    prompt_res = []
    inst_res = []

    for example in outputs:
        follow_instruction_list = example["follow_instruction_list"]
        instruction_id_list = example["instruction_id_list"]
        if all(follow_instruction_list):
            prompt_res.append(1)
        else:
            prompt_res.append(0)
        inst_res.append(sum(follow_instruction_list)/len(instruction_id_list))

    return prompt_res, inst_res


class MultiTurnInstructionFollowingPromptSolution:
    PROMPT_COLUMN_NAME = "multi_turn_prompt_column"

    def reformat_prompt(
        df_row: pd.core.series.Series,
        prompt_columns: List[str],
    ) -> str:
        """
        Reformat a DataFrame row into the format that can be used with the prompt generation template.
        """
        # TODO: revisit the prompt reformatting logic for oss
        prompt_col = None
        response_col = None
        if len(prompt_columns) >= 2:
            prompt_col = prompt_columns[0]  # turns
            response_col = prompt_columns[1]  # responses

        if prompt_col != "turns" or response_col != "responses":
            raise ValueError(
                f"Expecting prompt_columns to be [turns, responses], got {prompt_columns}"
            )

        if "turn_index" in df_row:
            turn_index = int(df_row["turn_index"])
        else:
            turn_index = 1
        if turn_index > 1:
            try:
                old_prompt = json.loads(df_row[prompt_col])
                old_response = [{"role": "assistant", "content": df_row[response_col]}]
            except Exception as e:
                raise ValueError(
                    f"Failed to parse old prompt and response due to error {e}"
                )
            new_turn_index = f"turn_{turn_index}_prompt"
            if new_turn_index in df_row.index:
                if df_row[new_turn_index] != "None" and df_row[new_turn_index]:
                    new_prompt = [json.loads(df_row[new_turn_index])]
                    output_prompt = old_prompt + old_response + new_prompt
                else:
                    output_prompt = [{"role": "user", "content": "None"}]
            else:
                logger.warning(f"Column {new_turn_index} does not exist!")
                output_prompt = [{"role": "user", "content": "None"}]
        else:
            # original input soruce table
            output_prompt = [json.loads(df_row[f"turn_{turn_index}_prompt"])]
        return output_prompt

    def compute_ci_via_bootstrap(
        result_list,
        n_resamples=10000,
        method='percentile',
        confidence_level=0.95
    ) -> float:
        prompt_lst, inst_lst = parse_result_no_reduce(result_list)
        prompt_pct_low, prompt_pct_high = bootstrap(
                    (np.array(prompt_lst),),
                    np.mean,
                    n_resamples=n_resamples,
                    method=method,
                    confidence_level=confidence_level,
                ).confidence_interval
        inst_pct_low, inst_pct_high = bootstrap(
                    (np.array(inst_lst),),
                    np.mean,
                    n_resamples=n_resamples,
                    method=method,
                    confidence_level=confidence_level,
                ).confidence_interval
        return prompt_pct_low, prompt_pct_high, inst_pct_low, inst_pct_high

    def metrics_gen(
        output_df: pd.DataFrame,
        return_outputs: bool = False
    ) -> Dict[str, Any]:
        """
        Generate metrics from the given table
        """
        language_list = [
            "all_languages",
            "German",
            "Italian",
            "Vietnamese",
            "Spanish",
            "Hindi",
            "Portuguese",
            "English",
            "French",
            "Thai",
            "Chinese",
            "Russian",
        ]
        outputs_strict = {language: [] for language in language_list}
        outputs_loose = {language: [] for language in language_list}

        row = output_df.iloc[0]
        turn_index = int(row["turn_index"])
        turn_index_prompt = f"turn_{turn_index}_prompt"

        index_counter = []
        for _, row in output_df.iterrows():
            if row[turn_index_prompt] == "None" or len(str(row[turn_index_prompt])) == 0:
                continue
            try:
                instruction_id_list = json.loads(
                            row[f"turn_{turn_index}_instruction_id_list"]
                        )
            except:
                continue
            kwargs_list = json.loads(row[f"turn_{turn_index}_kwargs"])
            kwargs = [json.loads(kwarg) for kwarg in kwargs_list]
            try:
                response = json.loads(row['responses'])[0]['response']
            except:
                response = row["responses"]

            input_dict = {
                "response": response,
                "instruction_id_list": instruction_id_list,
                "kwargs": kwargs,
            }

            outputs_strict["all_languages"].append(gen_acc_strict(input_dict))
            outputs_loose["all_languages"].append(gen_acc_loose(input_dict))

            language = row["language"]
            outputs_strict[language].append(gen_acc_strict(input_dict))
            outputs_loose[language].append(gen_acc_loose(input_dict))
            index_counter.append(row['key'])

        result_dict = {}

        result_dict.update(
            {f"turn_{turn_index}_prompts_number": len(outputs_strict["all_languages"])}
        )
        for language in language_list:
            if outputs_strict[language] == []:
                continue
            res_strict = parse_result(outputs=outputs_strict[language])
            res_strict_cis = MultiTurnInstructionFollowingPromptSolution.compute_ci_via_bootstrap(outputs_strict[language])
            res_loose = parse_result(outputs=outputs_loose[language])
            res_loose_cis = MultiTurnInstructionFollowingPromptSolution.compute_ci_via_bootstrap(outputs_loose[language])
            result_list = [res_strict[0], res_strict[1], res_loose[0], res_loose[1]]
            average = sum(result_list) / len(result_list)

            result_dict.update({f"turn_{turn_index}_{language}_overall": average})
            result_dict[f'{language}_cis_strict'] = res_strict_cis
            result_dict[f'{language}_cis_loose'] = res_loose_cis
        if not return_outputs:
            return result_dict
        else:
            outputs_strict['counter'] = index_counter
            outputs_loose['counter'] = index_counter
            return result_dict, outputs_strict, outputs_loose

    def get_text_column_name() -> str:
        """
        Get the column name that stores text in the dataframe
        """
        return MultiTurnInstructionFollowingPromptSolution.PROMPT_COLUMN_NAME

    def expand_df(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        logger.info("Expanding/updating dataframe...")
        new_index = 1
        if "turn_index" in df.columns:
            # TODO: need update astype thing.
            new_index = df["turn_index"].astype(int) + 1
            df["turn_index"] = new_index.astype(str)
        else:
            df["turn_index"] = str(new_index)
        return df


if __name__ == '__main__':
    evaluator = MultiTurnInstructionFollowingPromptSolution()
    df = pd.read_csv('data/o1-preview/eval_result_step_1.csv', keep_default_na=False)
    res = MultiTurnInstructionFollowingPromptSolution.metrics_gen(df)
    print(res)
