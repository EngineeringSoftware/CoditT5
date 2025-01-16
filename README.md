# CoditT5: Pretraining for Source Code and Natural Language Editing

This repo hosts the code and data for the following ASE 2022 paper:

Title: [CoditT5: Pretraining for Source Code and Natural Language Editing](https://arxiv.org/abs/2208.05446)

Authors: [Jiyang Zhang](https://jiyangzhang.github.io/), [Sheena Panthaplackel](https://panthap2.github.io/), [Pengyu Nie](https://pengyunie.github.io/), [Junyi Jessy Li](https://jessyli.com/), [Milos Gligoric](http://users.ece.utexas.edu/~gligoric/)

We propose a novel pretraining objective which explicitly models edits and use it to build CoditT5, a large language model for software-related editing tasks that is pretrained on large amounts of source code and natural language comments. We fine-tune it on various downstream editing tasks, including comment updating, bug fixing, and automated code review. 

```bibtex
@inproceedings{ZhangETAL22CoditT5,
  author = {Zhang, Jiyang and Panthaplackel, Sheena and Nie, Pengyu and Li, Junyi Jessy and Gligoric, Milos},
  title = {Codit{T}5: Pretraining for Source Code and Natural Language Editing},
  booktitle = {International Conference on Automated Software Engineering},
  year = {2022},
}
```

## News
**Aug 2023**

**Pretrained CoditT5** model is released on 🤗 ! 🔥 
[link](https://huggingface.co/JiyangZhang/CoditT5) 
\
Note: It is recommended fine-tuning it before applying to downstream tasks.
## Introduction

This repo contains the code and artifacts for reproducing the experiments in [CoditT5: Pretraining for Source Code and Natural Language Editing](https://arxiv.org/abs/2208.05446).

The code includes:

- scripts for synthesizing pretraining data for CoditT5
- scripts for processing data for downstream tasks
- scripts for training and evaluating CoditT5 on three downstream tasks
- scripts for combining CoditT5 and CodeT5 through reranking

The artifacts include:

- dataset used for pretraining CoditT5
- datasets for downstream tasks
- checkpoint for the pretrained CoditT5
- checkpoints for the CoditT5 models fine-tuned for downstream tasks

## Table of Contents


1. [How to Use][sec-howto]
2. [Dependency][sec-dependency]
3. [Data Downloads][sec-downloads]
4. [Code for Pretraining][sec-pretrain]
5. [Code for Processing Fine-tuning Data][sec-process]
6. [Code for Training and Evaluating Models][sec-traineval]
7. [Code for Combining CodeT5 and CoditT5][sec-rerank]

## How to Use

[sec-howto]: #how-to-use

```python
from transformers import T5ForConditionalGeneration, AutoTokenizer

checkpoint = "JiyangZhang/CoditT5"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint)

code_input = """class HelloWorld { public static void main(String[] args) { System.out.println("Hello, World!")"""

input_ids = tokenizer(code_input, return_tensors="pt").input_ids
generated_ids = model.generate(input_ids, max_length=200)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
# output: <INSERT>; } } ;<INSERT_END> class HelloWorld { public static void main(String[] args) { System.out.println("Hello, World!") ; } } ;
```

## Dependency

[sec-dependency]: #dependency

Our code require the following hardware and software environments.

- Operating system: Linux (tested on Ubuntu 20.04)
- Minimum disk space: 4 GB
- Python: 3.8
- Anaconda/Miniconda: appropriate versions for Python 3.8 or higher

Additional requirements for training and evaluating ML models:

- GPU: NVIDIA GTX 1080 Ti or better (with >= 11GB memory)
- CUDA: 10.2 or 11.3
- Disk space: 2 GB per trained model

[Anaconda](https://www.anaconda.com/products/individual#Downloads) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is required for installing the other Python library dependencies. Once Anaconda/Miniconda is installed, you can use the following command to setup a virtual environment, named `cdt`, with the Python library dependencies installed:

```
cd python/
./prepare_conda_env.sh
```

And then use `conda activate cdt` to activate the created virtual environment.

## Data Downloads

[sec-downloads]: #data-downloads

All our data is hosted on UTBox via [a zip file](https://utexas.box.com/s/9rkqnlp6wjhwyfmxce97pgb4ersfc1f9).

Data should be downloaded to this directory with the same directory structure (e.g., `data/` from the shared folder should be downloaded as `data/` under current directory).

## Code for Pretraining

[sec-pretrain]: #code-for-pretraining

### Synthesize Pretraining Data

We provide sample scripts to synthesize the pretraining dataset (by corrupting programming language code snippets and natural language comments) for CoditT5.

First, prepare the programming language and natural language data for pretraining; Then specify the following variables in the function `corrupt_pretrain_data()` in `python/run.sh`:

- `source_pl_file`: the path of data file where each line is a programming language function;
- `tokenized_pl_file`: the path of tokenized version of `source_pl_file`;
- `corrupt_pl_file`: corrupted version of `tokenized_pl_file` which is the input of pretrained model.
- `source_nl_file`: the path of data file where each line is a natural language sequence;
- `tokenized_nl_file`: the path of tokenized version of `source_nl_file`;
- `corrupt_nl_file`: corrupted version of `tokenized_nl_file` which is the input of pretrained model.

```
cd python/
./run.sh corrupt_pretrain_data
```

### Pretrain CoditT5

Requires the pretrain dataset at `data/CoditT5/pretrain/`

```
cd python/
./run.sh pretrain_CoditT5
```

## Code for Processing Fine-tuning Data

[sec-process]: #code-for-processing-fine-tuning-data

We provide the sample script to process the downstream datasets for CoditT5. Requires the raw data files at `raw_data/`.

```
cd python/
./run.sh process_coditT5_dataset --dataset ${dataset}

# Example: ./run.sh process_coditT5_dataset --dataset comment-update
```

Where `${dataset}` is the name of the dataset (comment-update, code-review, bf-small, bf-medium). The data files are generated to `data/CoditT5/${dataset}/`.

Notes:

- CoditT5's input data file name ends with `.buggy`; CoditT5's target output (edit plan + generation) file name ends with `.fixed`; target generation file name ends with `.seq`.
- CoditT5's input is in the form of `source_sequence </s> context_sequence`; and CoditT5's output is in the form of `edit_plan <s> target_sequence`
- Raw data files are stored in `raw_data/` (we provide some examples for demo), processed data files are generated to `data/CoditT5/${dataset}`
- Note that for the comment-update dataset, the processed `edit_plan` is the edits applied to the comment w/o parameter (@return, @param)

## Code for Training and Evaluating Models

[sec-traineval]: #code-for-training-and-evaluating-models

### Train ML models

Requires the dataset at `data/${model}/${dataset}/`, where `${model}` is the name of the model (CodeT5, CoditT5); `${dataset}` is the name of the dataset.

```
cd python/
./run.sh ${model}_train ${dataset}

# Example: ./run.sh CoditT5_train comment-update
```

Results are generated to `models/${model}/${dataset}/`, where:

- `model/`: stores the trained model.

- `logs/`: stores logs during training.

### Evaluate ML models

Requires the dataset at `data/${model}/${dataset}/`, the trained model at `models/${model}/${dataset}/model/`.

```
cd python/
./run.sh ${model}_generate ${dataset}

# Example: ./run.sh CoditT5_generate comment-update
```

Results are generated to `models/${model}/${dataset}/`, where:

- `output.hyp`: the predictions.

### Compute automatic metrics

Requires the model's predictions at `models/${model}/${dataset}/`. Note that the provided script assumes the names for the data files conform the what described in [Code for Processing Fine-tuning Data][sec-process]

```
./run.sh ${model}_eval ${dataset}

# Example: ./run.sh CoditT5_eval comment-update
```

Results are generated to `results/`:

- `results-${dataset}-${model}.json`: the average of automatic metrics.

- `scores-${dataset}-${model}.json`: the list of automatic metrics per sample.

## Code for Combining CodeT5 and CoditT5

[sec-rerank]: #code-for-combining-codet5-and-coditt5

Requires the dataset at `data/${model}/${dataset}/`, the trained models at `models/${model}/${dataset}/model/`.

### Rerank Models' outputs

```
cd python/
# Rerank CodeT5's outputs with CoditT5
./run.sh CodeT5_rerank ${dataset}
# Rerank CoditT5's outputs with CodeT5
./run.sh CodeT5_rerank ${dataset}

# Example: ./run.sh CoditT5_rerank comment-update
```

Main results are generated to `results/reranks/`:

- `test-${dataset}-${model}-top-20-rerank-${reranker}-results.json`: `${model}`'s top 20 beam outputs and `${reranker}`'s likelihood score for each beam output.

### Compute automatic metrics

Requires the model's reranking results file
`results/reranks/test-${dataset}-${model}-top-20-rerank-${reranker}-results.json`.

```
./run.sh eval_rerank_${model}_${reranker} ${dataset}

# Example: compute metrics for top 1 CoditT5 prediction reranked by CodeT5
./run.sh eval_rerank_CoditT5_CodeT5 comment-update
```

Results are generated to `results/`:

- `results-${dataset}-${model}-rerank-${reranker}.json`: the average of automatic metrics.

- `scores-${dataset}-${model}-rerank-${reranker}.json`: the list of automatic metrics per sample.
