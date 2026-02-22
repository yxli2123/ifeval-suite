import re

from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Any, Literal

from .models.bedrock import BedrockModel
from .models.openai import OpenAIModel
from tqdm import tqdm
import boto3

class JudgeResponse(BaseModel):
    reasoning: str
    verdict: Literal["YES", "NO"]

JUDGE_PROMPT = '''You are tasked with evaluating a model response to see if it meets a specific criteria.
The criteria will always be YES/NO evaluation.

The model response is as follows:
<MODEL_RESPONSE>
{}
</MODEL_RESPONSE>

The criteria that the model response must meet is as follows. Be VERY STRICT!:
<CRITERIA>
{}
</CRITERIA>

Give your reasoning followed by your verdict, either "YES" or "NO".
Wrap the reasoning in <reasoning></reasoning> and wrap the answer in <verdict></verdict>.

'''

def extract_content_within_tag(
    text: str,
    tag: str | None = None,
    tag_pair: tuple[str, str] | None = None,
    strict: bool = True,
) -> str | None:
    assert tag is not None or tag_pair is not None, "Use either tag or tag_pair"

    tag_l, tag_r = tag_pair if tag_pair else (f"<{tag}>", f"</{tag}>")
    pattern = rf"{re.escape(tag_l)}\s*(.*?)\s*{re.escape(tag_r)}"
    text_match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)

    default = "" if strict else text
    target_text = text_match.group(1) if text_match else default
    text = target_text if target_text else default

    return text

class Evaluator:
    def __init__(
        self,
        conversations: List[Any],
        responses: Dict[int, List[str]],
        model_name: str,
        base_url: str | None,
        api_backend: Literal["OpenAIModel", "BedrockModel"],
    ):
        self.conversations = conversations
        self.responses = responses
        self.api_backend = api_backend

        if api_backend == "OpenAIModel":
            self.evaluation_model = OpenAIModel(
                model_name=model_name,
                base_url=base_url,
                temp=0,
            )
        elif api_backend == "BedrockModel":
            self.evaluation_model = BedrockModel(
                model_id=model_name,
                temp=0,
            )
        else:
            raise ValueError(f"Unsupported API backend: {api_backend}")
        self.results = []

    def evaluate_helper(self, i: int, conversation: Any, response: str) -> Tuple[int, str, str, str, str]:
        """Evaluate a single response."""
        target_question = conversation.target_question
        pass_criteria = conversation.pass_criteria
        prompt = JUDGE_PROMPT.format(response, target_question)

        judgement: str = self.evaluation_model.generate([{"role": "user", "content": prompt}])

        reasoning = extract_content_within_tag(judgement, tag="reasoning")
        verdict = extract_content_within_tag(judgement, tag="verdict")
        if not verdict:
            print("Unable to parse verdict.\n", judgement)
        return i, conversation.axis, reasoning, verdict, pass_criteria

    def evaluate(self, max_workers:int = 1) -> List[Dict]:
        """Evaluate all responses for each conversation"""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, convo in enumerate(self.conversations):
                if convo.question_id not in self.responses:
                    # Handle missing question_id
                    self.results.append({
                        'question_id': convo.question_id,
                        'axis': convo.axis,
                        'attempt': 0,
                        'reasoning': 'NA - Question ID not found in responses',
                        'verdict': 'NO',
                        'pass_criteria': convo.pass_criteria,
                        'passed': False
                    })
                else:
                    for j, response in enumerate(self.responses[convo.question_id]):
                        futures.append(
                            executor.submit(self.evaluate_helper, i, convo, response)
                        )

            for future in tqdm(futures, desc="Evaluating responses", total=len(futures)):
                # try:
                i, axis, reasoning, verdict, pass_criteria = future.result()
                self.results.append({
                    'question_id': self.conversations[i].question_id,
                    'axis': axis,
                    'attempt': j,
                    'reasoning': reasoning,
                    'verdict': verdict,
                    'pass_criteria': pass_criteria,
                    'passed': verdict == pass_criteria
                })
                # except Exception as e:
                #     # Handle any other unexpected errors
                #     print(e)
                #     self.results.append({
                #         'question_id': self.conversations[i].question_id if i < len(self.conversations) else 'Unknown',
                #         'axis': 'NA',
                #         'attempt': 'NA',
                #         'reasoning': f'Error during evaluation: {str(e)}',
                #         'verdict': 'NO',
                #         'pass_criteria': 'NA',
                #         'passed': False
                #     })

        # Calculate the final pass/fail status for each question
        question_results = {}
        for result in self.results:
            question_id = result['question_id']
            if question_id not in question_results:
                question_results[question_id] = {'attempts': 0, 'passes': 0}
            question_results[question_id]['attempts'] += 1
            if result['passed']:
                question_results[question_id]['passes'] += 1

        # Update results with final pass/fail status
        for result in self.results:
            question_id = result['question_id']
            attempts = question_results[question_id]['attempts']
            passes = question_results[question_id]['passes']
            result['final_status'] = f"{'PASS' if passes > 0 else 'FAIL'} ({passes}/{attempts} attempts passed)"

        return self.results