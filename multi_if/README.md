# Multi-Turn Evaluation Framework

This repository contains a Python implementation of a multi-turn evaluation benchmark for large language model (LLM). The benchmark is designed to evaluate the performance of LLM models' capabilities in multi-turn instruction following within a multilingual environment.



## Files

The repository contains the following files:

* `api_client.py`: This file contains the implementation of the interface for LLMs interactions via API calls.
* `ifeval.py`: This file contains the implementation of the Inference Evaluation (IFEVAL) metric, which is used to evaluate the capability of LLM following natural language instructions
* `metrics.py`: This file contains the implementation of various metrics that can be used to calculate ifeval, data preprocess and enrichment for multi turn instructions.
* `utils.py`: This file contains utility functions that are used throughout the framework, e.g., GenerationSetting, get_inference_batch(),preprocess_data.
* `multi_turn_instruct_following_eval_api.py`: This file contains the main function that executes the multi-turn evaluation benchmark via API calls.
* `multi_turn_instruct_following_eval_vllm.py`: This file contains the main function that executes the multi-turn evaluation benchmark via running on local GPUs.


## Usage

To use the multi-turn evaluation benchmar, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/your-username/multi-turn-evaluation.git
```
2. Install the required dependencies:
```bash
cd multi_turn_eval
pip install -r requirements.txt
```

3. Download the data from huggingface:
```
git clone https://huggingface.co/datasets/facebook/Multi-IF data/Multi-IF
```

4. Run the main function in `multi_turn_instruct_following_eval_vllm.py`:
```bash
python multi_turn_instruct_following_eval_vllm.py \
        --model_path <MODEL_PATH> \
        --tokenizer_path <TOKENIZER_PATH> \
        --input_data_csv <INPUT_DATA_CSV> \
        --batch_size <BATCH_SIZE> \
        --tensor_parallel_size <TENSOR_PARALLEL_SIZE>
```
This will execute the multi-turn evaluation benchmar and output the results to the console and intermediate generation results saved in csv files.

For example, for Meta-Llama-3.1-70B-Instruct,
```bash
python multi_turn_instruct_following_eval_vllm.py \
        --model_path meta-llama/Llama-3.1-70B-Instruct \
        --tokenizer_path meta-llama/Llama-3.1-70B-Instruct \
        --input_data_csv data/Multi-IF/multiIF_20241018.csv \
        --batch_size 4 \
        --tensor_parallel_size 8
```

Or for running evaluation via API please use

4. Run the main function in `multi_turn_instruct_following_eval_api.py` with `claude-3.5-sonnet-20240620`:
```bash
python multi_turn_instruct_following_eval_api.py \
        --max_workers 5 \
        --api_model_name claude-3.5-sonnet-20240620 \
        --input_data_csv data/Multi-IF/multiIF_20241018.csv \
        --max_new_tokens 1024 \
        --steps 1 2 3
```

## Bibtex
If you use the code or benchmark, please consider citing the following paper:
```
@article{he2024multi,
  title={Multi-IF: Benchmarking LLMs on Multi-Turn and Multilingual Instructions Following},
  author={He, Yun and Jin, Di and Wang, Chaoqi and Bi, Chloe and Mandyam, Karishma and Zhang, Hejia and Zhu, Chen and Li, Ning and Xu, Tengyu and Lv, Hongjiang and others},
  journal={arXiv preprint arXiv:2410.15553},
  year={2024}
}
```

## Contributing

We welcome contributions to this repository! If you have any suggestions or improvements, please open an issue or submit a pull request.

## License

This project is licensed under the Apache License, Version 2.0 (the "License"). See the LICENSE file for details.
