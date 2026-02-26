#!/bin/bash

# ========== Config here ==========
OUTPUT_DIR=${OUTPUT_DIR:-"eval_results/run-$(date '+%Y%m%d-%H%M%S')"}

# Model path, local or huggingface.
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
HF_TOKEN=${HF_TOKEN:-}

# OpenAI API config
#JUDGE_BACKEND=OpenAIModel
#JUDGE_NAME=${JUDGE_NAME:-openai.gpt-oss-120b-1:0}
#JUDGE_URL=${JUDGE_URL:-https://api.openai.com}
API_KEY=${API_KEY:-}
if [[ -n "${API_KEY:-}" ]]; then
  export OPENAI_API_KEY="$API_KEY"
elif [[ -n "${API_KEY_PATH:-}" && -r "$API_KEY_PATH" && -f "$API_KEY_PATH" ]]; then
  export OPENAI_API_KEY="$(tr -d '\n\r' <"$API_KEY_PATH")"
else
  echo "Error: provide API_KEY or a readable API_KEY_PATH." >&2
  exit 1
fi

## AWS Bedrock config
#JUDGE_BACKEND=BedrockModel
#JUDGE_NAME=${JUDGE_NAME:-}
#export AWS_REGION=${AWS_REGION:-}
#export AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID:-}
#export AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY:-}
#export AWS_SESSION_TOKEN=${AWS_SESSION_TOKEN:-}


# Install dependency
bash /root/ifeval-suite/install_dependency.sh


# Download the model.
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model dir not found: ${MODEL_PATH}"
  echo "Downloading ${MODEL_PATH} -> /root/ckpt/${MODEL_PATH}"
  hf download "${MODEL_PATH}" --local-dir "/root/ckpt/${MODEL_PATH}"
  export MODEL_PATH="/root/ckpt/${MODEL_PATH}"
else
  echo "Model dir exists: ${MODEL_PATH} (skip download)"
fi


##### IFEval GSM8K #####
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

lm_eval --model vllm \
  --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,dtype=auto,gpu_memory_utilization=0.8,max_model_len=8192,think_end_token="</think>" \
  --apply_chat_template \
  --tasks ifeval,gsm8k_cot \
  --gen_kwargs "temperature=0.6,top_p=0.95" \
  --batch_size auto \
  --apply_chat_template \
  --output_path ${OUTPUT_DIR}/lm-eval-harness

find ${OUTPUT_DIR}/lm-eval-harness -type f \( -name "*.txt" -o -name "*.json" -o -name "*.jsonl" \) -exec sh -c '
  for f do
    printf "\n===== %s =====\n" "$f"
    head -n 50 "$f"
  done
' sh {} +

##### IFBench #####
# 1) Generate responses to IFBench test data
IFBENCH_RESPONSE_FILE=${OUTPUT_DIR}/if_bench/response_output.jsonl

python if_bench/generate_response.py \
  --test_file if_bench/data/IFBench_test.jsonl \
  --response_file ${IFBENCH_RESPONSE_FILE} \
  --model_path ${MODEL_PATH} \
  --tensor_parallel_size 8 \
  --max_new_tokens 4096 \
  --batch_size 256 \
  --temperature 0.6

head -50 ${IFBENCH_RESPONSE_FILE}

# 2) Run evaluation
python3 if_bench/run_eval.py \
  --input_data=if_bench/data/IFBench_test.jsonl \
  --input_response_data=$IFBENCH_RESPONSE_FILE \
  --output_dir=$OUTPUT_DIR/if_bench


###### Multi-Challenge #####
# 1) Generate responses to IFBench test data
MULTI_RESPONSE_FILE=$OUTPUT_DIR/multi_challenge/response_output.jsonl

python multi_challenge/generate_response.py \
  --test_file multi_challenge/data/benchmark_questions.jsonl \
  --response_file ${MULTI_RESPONSE_FILE} \
  --model_path ${MODEL_PATH} \
  --tensor_parallel_size 8 \
  --max_new_tokens 4096 \
  --batch_size 256 \
  --temperature 0.6

head -50 ${MULTI_RESPONSE_FILE}

# 2) Run evaluation
python multi_challenge/main.py \
  --test-file multi_challenge/data/benchmark_questions.jsonl \
  --responses-file ${MULTI_RESPONSE_FILE} \
  --output-file ${OUTPUT_DIR}/multi_challenge/evaluation_results.txt \
  --raw ${OUTPUT_DIR}/multi_challenge/detailed_results.csv \
  --judge-name ${JUDGE_NAME} \
  --judge-url ${JUDGE_URL} \
  --judge-backend ${JUDGE_BACKEND} \
  --max-workers_eval 64

cat $OUTPUT_DIR/multi_challenge/evaluation_results.txt


##### Multi-IF #####
python multi_if/multi_turn_instruct_following_eval_vllm.py \
  --model_path ${MODEL_PATH} \
  --tokenizer_path ${MODEL_PATH} \
  --input_data_csv multi_if/multiIF_20241018.csv \
  --batch_size 256 \
  --tensor_parallel_size 8
