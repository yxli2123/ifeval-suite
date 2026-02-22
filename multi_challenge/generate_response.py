#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# vLLM
from vllm import LLM, SamplingParams

torch.backends.cuda.enable_cudnn_sdp(False)

END_MARKERS = ("<end_chat>", "<|end|>", "<end_of_turn>", "<|eot_id|>")


def maybe_remove_think(response: str | None, think_tag: str = "think") -> str:
    if not response:
        return ""
    if f"</{think_tag}>" in response:
        parts = response.split(f"</{think_tag}>")
        return parts[-1].lstrip("\n")
    elif f"<{think_tag}>" in response and f"</{think_tag}>" not in response:
        return ""
    else:
        return response


def clean_text(text: str) -> str:
    text = maybe_remove_think(text, "think")
    for m in END_MARKERS:
        text = text.replace(m, "")
    return text.strip()


def render_inputs(ds, tokenizer, use_chat_template: bool = True):
    """Create an 'inputs' column for the dataset (chat templated or raw)."""
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):

        def render(batch):
            ins = []
            for msgs in batch["CONVERSATION"]:
                ins.append(
                    tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
                    )
                )
            return {"inputs": ins}

        keep_cols = [c for c in ds.column_names if c in ("CONVERSATION", "QUESTION_ID")]
        return ds.map(
            render,
            batched=True,
            remove_columns=[c for c in ds.column_names if c not in keep_cols],
        )
    else:
        if "inputs" not in ds.column_names:
            ds = ds.add_column("inputs", ds["CONVERSATION"])
        return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", required=True, type=str, help="JSON/JSONL with fields CONVERSATION, QUESTION_ID.")
    parser.add_argument("--response_file", required=True, type=str, help="Final JSONL output.")
    parser.add_argument("--model_path", required=True, type=str, help="HF model ID or local path.")

    # vLLM / performance knobs
    parser.add_argument("--tensor_parallel_size", type=int, default=8, help="Use 8 GPUs via tensor parallelism.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--swap_space", type=float, default=4.0, help="GiB CPU swap per GPU (set 0 if always best_of=1).")
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--enable_chunked_prefill", action="store_true", help="Helps throughput for long prompts.")
    parser.add_argument("--max_num_seqs", type=int, default=None, help="Optional: cap concurrent sequences.")
    parser.add_argument("--max_num_batched_tokens", type=int, default=None, help="Optional: cap total batched tokens/step.")

    # generation
    parser.add_argument("--batch_size", type=int, default=256, help="How many prompts to submit to vLLM per chunk.")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)

    # HF compatibility
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--no_chat_template", action="store_true")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.response_file)), exist_ok=True)

    # Build vLLM engine (single process; weights distributed across GPUs)
    # LLM supports tensor_parallel_size, seed, gpu_memory_utilization, swap_space, trust_remote_code, etc. :contentReference[oaicite:1]{index=1}
    engine_kwargs: Dict[str, Any] = {
        "max_model_len": args.max_model_len,  # forwarded as EngineArgs kwarg
    }
    if args.enable_chunked_prefill:
        engine_kwargs["enable_chunked_prefill"] = True  # :contentReference[oaicite:2]{index=2}
    if args.max_num_seqs is not None:
        engine_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.max_num_batched_tokens is not None:
        engine_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,  # :contentReference[oaicite:3]{index=3}
        dtype=args.dtype,                               # :contentReference[oaicite:4]{index=4}
        trust_remote_code=args.trust_remote_code,       # :contentReference[oaicite:5]{index=5}
        seed=args.seed,                                 # :contentReference[oaicite:6]{index=6}
        gpu_memory_utilization=args.gpu_memory_utilization,  # :contentReference[oaicite:7]{index=7}
        swap_space=args.swap_space,                     # :contentReference[oaicite:8]{index=8}
        **engine_kwargs,
    )

    # Tokenizer for chat templating (prefer vLLM’s tokenizer; fallback to HF)
    tokenizer = None
    try:
        tokenizer = llm.get_tokenizer()
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=args.trust_remote_code)

    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"

    # Load dataset and render inputs
    ds_full = load_dataset("json", data_files=args.test_file, split="train")
    ds_full = render_inputs(ds_full, tokenizer, use_chat_template=not args.no_chat_template)
    n = len(ds_full)

    # Sampling params (vLLM uses max_tokens; stop strings supported and removed from output by default) :contentReference[oaicite:9]{index=9}
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
        stop=list(END_MARKERS),
        include_stop_str_in_output=False,
    )

    # Generate in chunks to avoid holding everything in memory
    with open(args.response_file, "w", encoding="utf-8") as fout:
        for start in tqdm(range(0, n, args.batch_size), desc="[vLLM] Generating"):
            end = min(start + args.batch_size, n)
            batch = ds_full[start:end]
            prompts: List[str] = batch["inputs"]
            qids: List[Any] = batch["QUESTION_ID"]

            # vLLM returns outputs aligned with input order. :contentReference[oaicite:10]{index=10}
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

            for qid, out in zip(qids, outputs):
                # n=1 by default -> take first completion
                gen = out.outputs[0].text if out.outputs else ""
                gen = clean_text(gen)
                record = {"QUESTION_ID": qid, "RESPONSE": [gen]}
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[main] Wrote responses to {args.response_file}")


if __name__ == "__main__":
    main()
