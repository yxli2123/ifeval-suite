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

import argparse
import logging
import os
import time
from typing import Any, Dict, List, Optional

import pandas as pd
from metrics import MultiTurnInstructionFollowingPromptSolution
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
)
from vllm import LLM
from utils import GenerationSetting, get_inference_batch_vllm, preprocess_data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def main(
    model_path: str,
    tokenizer_path: str,
    input_data_csv: str = "dataset/multi_turn_sample.csv",
    batch_size=24,
    generation_setting=GenerationSetting(
        max_new_tokens=1024, temperature=0.6, top_p=0.9
    ),
    need_write2file: bool = True,
    output_filepath_prefix: str = "eval_result",
    tensor_parallel_size: int = 8,
    steps: List[int] = [1, 2, 3],
    tag: str = '_test_long_sys'
) -> None:
    benchmark_df = pd.read_csv(input_data_csv, keep_default_na=False)
    num_rows = len(benchmark_df)
    logger.info(f"Number of rows: {num_rows}")

    final_metric_result = {}

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    model = LLM(
        model_path,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        distributed_executor_backend="mp",
        seed=generation_setting.seed,
    )

    start = time.time()
    model_name = model_path.split('/')[-1].strip() + tag
    step_input_df = benchmark_df.copy()

    for step in steps:
        output_filepath = (
            f"results/{model_name}/{output_filepath_prefix}_step_{step}.csv"
        )
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        step_output_df = run_step(
            model=model,
            tokenizer=tokenizer,
            input_df=step_input_df,
            row_limit=-1,
            need_write2file=need_write2file,
            output_filepath=output_filepath,
            device=0,
            generation_setting=generation_setting,
            batch_size=batch_size,
        )
        step_input_df = step_output_df.copy()
        step_metric_result = run_metric(
            output_filepath=output_filepath,
            step=step,
        )
        final_metric_result[step] = step_metric_result

    logger.info(
        f"Total time: {time.time() - start}\n Number of rows: {num_rows}, \n Number of processes: 1"
    )
    logger.info(f"Final metrics: {final_metric_result}")


def run_step(
    model: LLM,  # Changed to vllm's LLM
    tokenizer: PreTrainedTokenizerBase,
    input_df: pd.DataFrame,
    prompt_columns: List[str] = ["turns", "responses"],
    step: int = 0,
    row_limit: int = -1,
    need_write2file: bool = True,
    output_filepath: str = "eval_result.csv",
    device: Optional[str] = None,
    generation_setting: GenerationSetting = GenerationSetting(),
    batch_size: int = 24
) -> pd.DataFrame:
    output_df = preprocess_data(
        input_df, prompt_columns=prompt_columns, row_limit=row_limit
    )
    step_output_df = get_inference_batch_vllm(
        model=model,
        tokenizer=tokenizer,
        input_df=output_df,
        batch_size=batch_size,
        generation_setting=generation_setting,
        need_write2file=need_write2file,
        output_filepath=output_filepath,
        device=device,
    )
    return step_output_df


def run_metric(
    output_filepath: str = "eval_result", step: int = 0
) -> Dict[str, Any]:
    step_output_df = None
    csv = output_filepath
    logger.info(f"calculating metrics for step_{step}")
    temp_df = pd.read_csv(csv, keep_default_na=False)
    step_output_df = temp_df.copy()
    metric_result = MultiTurnInstructionFollowingPromptSolution.metrics_gen(
        step_output_df
    )
    metric_result_df = pd.DataFrame.from_dict(metric_result, orient="index")
    metric_result_df.to_csv(output_filepath.replace('.csv', '_metric.csv'), index=False)
    logger.info(f"step_{step} metrics \n: {metric_result}")
    return metric_result


if __name__ == "__main__":
    """
    !!NOTE!!: make sure the number of available GPUs == tensor_parallel_size
    Usage:
    python multi_turn_instruct_following_eval_vllm_final.py \
        --model_path <MODEL_PATH> \
        --tokenizer_path <TOKENIZER_PATH> \
        --input_data_csv <INPUT_DATA_CSV> \
        --batch_size <BATCH_SIZE> \
        --need_write2file <NEED_WRITE2FILE> \
        --output_filepath_prefix <OUTPUT_FILEPATH_PREFIX> \
        --tensor_parallel_size <TENSOR_PARALLEL_SIZE>

    Example:
    python multi_turn_instruct_following_eval_vllm_final.py \
        --model_path meta-llama/Llama-3.1-70B-Instruct \
        --tokenizer_path meta-llama/Llama-3.1-70B-Instruct \
        --input_data_csv dataset/multi_turn_sample_v6.csv \
        --batch_size 64 \
        --tensor_parallel_size 8

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Meta-Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--tokenizer_path", type=str, default="Meta-Llama-3.1-8B-Instruct"
    )
    parser.add_argument(
        "--input_data_csv", type=str, default="dataset/multi_turn_sample.csv"
    )
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--need_write2file", type=bool, default=True)
    parser.add_argument("--output_filepath_prefix", type=str, default="eval_result")
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument(
        '--steps',
        type=int,
        nargs='+',
        default=[1, 2, 3],
        help='List of steps to process (e.g., --steps 1 2 3)'
    )
    args = parser.parse_args()
    main(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        input_data_csv=args.input_data_csv,
        batch_size=args.batch_size,
        need_write2file=args.need_write2file,
        output_filepath_prefix=args.output_filepath_prefix,
        tensor_parallel_size=args.tensor_parallel_size,
        steps=args.steps
    )
