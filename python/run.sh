#!/bin/bash

_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
readonly DATASET_DIR=${_DIR}/../data
readonly MODELS_DIR=${_DIR}/../models
readonly RESULTS_DIR=${_DIR}/../results


# corrupt file for pretraining CoditT5
function corrupt_pretrain_data() {

        mkdir -p ${DATASET_DIR}/pretrain

        local RAWDATA_DIR=${_DIR}/../raw_data
        local source_pl_file=${1:-${RAWDATA_DIR}/pretrain/train.csn.pl}; shift
        local tokenized_pl_file=${1:-${DATASET_DIR}/pretrain/csn.pl.fixed}; shift
        local corrupt_pl_file=${1:-${DATASET_DIR}/pretrain/csn.pl.buggy}; shift

        local source_nl_file=${1:-${RAWDATA_DIR}/pretrain/train.csn.nl}; shift
        local tokenized_nl_file=${1:-${DATASET_DIR}/pretrain/csn.nl.fixed}; shift
        local corrupt_nl_file=${1:-${DATASET_DIR}/pretrain/csn.nl.buggy}; shift

        mkdir -p "$(dirname $source_pl_file)"
        mkdir -p "$(dirname $tokenized_pl_file)"
        mkdir -p "$(dirname $corrupt_pl_file)"
        mkdir -p "$(dirname $source_nl_file)"
        mkdir -p "$(dirname $tokenized_nl_file)"
        mkdir -p "$(dirname $corrupt_nl_file)"

        python -m cdt.collector.mask corrupt_code \
                --java_file $source_pl_file \
                --fixed_file $tokenized_pl_file \
                --output_file $corrupt_pl_file
        
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
