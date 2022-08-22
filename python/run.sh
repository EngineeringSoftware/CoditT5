#!/bin/bash

_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
readonly DATASET_PATH=${_DIR}/../data
readonly MODELS_DIR=${_DIR}/../models
readonly RESULTS_DIR=${_DIR}/../results


readonly source_pl_file=${_DIR}
readonly tokenized_pl_file=${_DIR}
readonly corrupt_pl_file=${_DIR}

readonly source_nl_file=${_DIR}
readonly tokenized_nl_file=${_DIR}
readonly corrupt_nl_file=${_DIR}

# corrupt file for pretraining CoditT5
function corrupt_pretrain_data() {

        python -m cdt.collector.mask corrupt_code \
                --java_file $source_code_file \
                --fixed_file $tokenized_code_file \
                --output_file $corrupt_code_file
        
        python -m cdt.collector.mask corrupt_nl \
                --nl_file $source_nl_file \
                --fixed_file $tokenized_nl_file \
                --output_file $corrupt_nl_file
}

# Pretrain CoditT5 from CodeT5 checkpoint
function pretrain_CoditT5() {

        mkdir -p ${MODELS_DIR}/CoditT5/pretrain

        python -m cdt.coditT5.CodeT5 fit \
                --exp_dir ${MODELS_DIR}/pretrain \
                --data.model CoditT5 \
                --data.dataset pretrain \
                --config configs/pretrain-codeT5.yaml
}

function process_coditT5_dataset() {
        local dataset=${1:?1st arg: dataset name}; shift

        python -m cdt.coditT5.DataProcessor process_dataset --dataset ${dataset}
}

# train
function CoditT5_train() {
        local dataset=${1:?1st arg: dataset name}; shift
        local args="$@"

        set -e
        set -x

        mkdir -p ${MODELS_DIR}/CoditT5/${dataset}
        python -m cdt.coditT5.CodeT5 fit \
                --exp_dir ${MODELS_DIR}/CoditT5/${dataset} \
                --data.dataset ${dataset} \
                --data.model CoditT5 \
                --config configs/coditT5.yaml \
                $args
}

function CodeT5_train() {
        local dataset=${1:?1st arg: dataset name}; shift
        local args="$@"

        set -e
        set -x

        mkdir -p ${MODELS_DIR}/CodeT5/${dataset}
        python -m cdt.coditT5.CodeT5 fit \
                --exp_dir ${MODELS_DIR}/CodeT5/${dataset} \
                --data.dataset ${dataset} \
                --config configs/codeT5.yaml \
                $args
}

function CodeT5_edit_train() {
        local dataset=${1:?1st arg: dataset name}; shift
        local args="$@"

        set -e
        set -x

        mkdir -p ${MODELS_DIR}/CodeT5-edit/${dataset}
        python -m cdt.coditT5.CodeT5 fit \
                --exp_dir ${MODELS_DIR}/CodeT5-edit/${dataset} \
                --data.dataset ${dataset} \
                --data.model CoditT5 \
                --config configs/codeT5.yaml \
                $args
}

# generate
function CoditT5_generate() {
        local dataset=${1:?1st arg: dataset name}; shift
        local args="$@"

        set -e
        set -x

        mkdir -p ${MODELS_DIR}/CoditT5/${dataset}
        python -m cdt.coditT5.CodeT5 test \
                --exp_dir ${MODELS_DIR}/CoditT5/${dataset} \
                --data.dataset ${dataset} \
                --data.model CoditT5 \
                --config configs/coditT5.yaml \
                $args
}

function CodeT5_generate() {
        local dataset=${1:?1st arg: dataset name}; shift
        local args="$@"

        set -e
        set -x

        mkdir -p ${MODELS_DIR}/CodeT5/${dataset}
        python -m cdt.coditT5.CodeT5 test \
                --exp_dir ${MODELS_DIR}/CodeT5/${dataset} \
                --data.dataset ${dataset} \
                --config configs/codeT5.yaml \
                $args
}

function CodeT5_edit_generate() {
        local dataset=${1:?1st arg: dataset name}; shift
        local args="$@"

        set -e
        set -x

        mkdir -p ${MODELS_DIR}/CodeT5-edit/${dataset}
        python -m cdt.coditT5.CodeT5 test \
                --exp_dir ${MODELS_DIR}/CodeT5-edit/${dataset} \
                --data.dataset ${dataset} \
                --data.model CoditT5 \
                --config configs/codeT5.yaml \
                $args
}

# eval, compute metrics
function CoditT5_eval() {
        local dataset=${1:?1st arg: dataset name}; shift
        local args="$@"

        set -e
        set -x
        
        mkdir -p ${RESULTS_DIR}/
        # prepare data files
        cp ${DATASET_DIR}/CoditT5/${dataset}/test.${dataset}.buggy ${MODELS_DIR}/CoditT5/${dataset}/output.src
        cp ${DATASET_DIR}/CoditT5/${dataset}/test.${dataset}.seq ${MODELS_DIR}/CoditT5/${dataset}/output.ref

        python -m cdt.coditT5.DataProcessor post_process_model_generation --dataset ${dataset}
        python -m cdt.eval.evaluate run_evaluation \
                --dataset=${dataset} \
                --model="CoditT5"
}


function CodeT5_eval() {
        local dataset=${1:?1st arg: dataset name}; shift
        local args="$@"

        set -e
        set -x
        
        mkdir -p $RESULTS_DIR
        # prepare data files
        cp ${DATASET_DIR}/CodeT5/${dataset}/test.${dataset}.buggy ${MODELS_DIR}/CodeT5/${dataset}/output.src
        cp ${DATASET_DIR}/CodeT5/${dataset}/test.${dataset}.seq ${MODELS_DIR}/CodeT5/${dataset}/output.ref

        python -m cdt.eval.evaluate run_evaluation \
                --dataset=${dataset} \
                --model="CodeT5"
}

function CodeT5_edit_eval() {
        local dataset=${1:?1st arg: dataset name}; shift
        local args="$@"

        set -e
        set -x

        mkdir -p $RESULTS_DIR
        # prepare data files
        cp ${DATASET_DIR}/CodeT5/${dataset}/test.${dataset}.buggy ${MODELS_DIR}/CodeT5-edit/${dataset}/output.src
        cp ${DATASET_DIR}/CodeT5/${dataset}/test.${dataset}.seq ${MODELS_DIR}/CodeT5-edit/${dataset}/output.ref
        
        python -m cdt.coditT5.DataProcessor post_process_model_generation \
                --dataset ${dataset} \
                --model CodeT5-edit

        python -m cdt.eval.evaluate run_evaluation \
                --dataset=${dataset} \
                --model="CodeT5-edit"
}



# Rerank

function CoditT5_rerank() {

        local dataset=${1:?1st arg: dataset name}; shift
        local args="$@"

        set -e
        set -x
        
        # 0. create reranks dir
        mkdir -p ${RESULTS_DIR}/reranks
        # 1. generate top 20 sequences
        ./run.sh CoditT5_generate ${dataset} --model.num_return_sequences 20 --model.beam_size 20 --data.eval_batch_size 2
        # 2. rerank
        python -m cdt.collector.T5Rerank run_rerank --model CoditT5 --dataset ${dataset} --reranker CodeT5     
}

function CodeT5_rerank() {

        local dataset=${1:?1st arg: dataset name}; shift
        local args="$@"

        set -e
        set -x
        
        # 0. create reranks dir
        mkdir -p ${RESULTS_DIR}/reranks
        # 1. generate top 20 sequences
        ./run.sh CodeT5_generate ${dataset} --model.num_return_sequences 20 --model.beam_size 20 --data.eval_batch_size 2
        # 2. rerank
        python -m cdt.collector.T5Rerank run_rerank --model CodeT5 --dataset ${dataset} --reranker CoditT5
}

function eval_rerank_CoditT5_CodeT5() {

        local dataset=${1:?1st arg: dataset name}; shift
        local args="$@"

        python -m cdt.eval.evaluate calculate_metrics_after_rerank --model CoditT5 \
                --reranker CodeT5 \
                --dataset ${dataset} \
                --beam_size 20
}

function eval_rerank_CodeT5_CoditT5() {

        local dataset=${1:?1st arg: dataset name}; shift
        local args="$@"

        python -m cdt.eval.evaluate calculate_metrics_after_rerank --model CodeT5 \
                --reranker CoditT5 \
                --dataset ${dataset} \
                --beam_size 20
}



# ==========
# Main function -- program entry point
# This script can be executed as ./run.sh the_function_to_run

function main() {
        local action=${1:?Need Argument}
        shift

        (
                cd ${_DIR}
                $action "$@"
        )
}

main "$@"

# ==========
# Some notes of useful Bash commands

# Export Anaconda environment
# conda env export --from-history > env.yml

# Load Anaconda envrionment
# conda env create -n NAME -f env.yml

# To let pycharm recognize the structure, make python/ as source folder
