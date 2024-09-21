import os
from pathlib import Path
import collections
import math
from seutil import IOUtils, LoggingUtils
from typing import List, Tuple, Union
from jsonargparse import CLI
import subprocess
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from recordclass import RecordClass, asdict
from statistics import mean

from cdt.eval.pycocoevalcap.meteor.meteor import Meteor
from cdt.collector.EditSeqProducer import EditSeqProducer
from cdt.Macros import Macros
from cdt.eval.SARI import SARIsent
from cdt.collector.DataProcessor import DataProcessor
from cdt.eval.CodeBLEU.Evaluator import Evaluator

code_bleu_evaluator = Evaluator(lang="java")  # Initialize CodeBLEU evaluator


class EvaluationMetrics(RecordClass):
    xMatch: float = 0.0
    xMatch_list: List[int] = None
    BLEU: float = 0.0
    BLEU_list: List[float] = None
    CodeBLEU: float = 0.0
    CodeBLEU_list: List[float] = None
    GLEU: float = 0.0
    GLEU_list: List[float] = None
    SARI: float = 0.0
    SARI_list: List[float] = None
    METEOR: float = 0.0
    METEOR_list: List[float] = None
    COPY: float = 0.0


class EvalOutputs(RecordClass):
    refs: List[str] = None
    srcs: List[str] = None
    preds: List[str] = None
    ctxs: List[str] = None
    edits: List[str] = None


def compute_sentence_meteor(
    reference_list: List[List[List[str]]], sentences: List[List[str]]
) -> List[float]:
    """Compute sentence level meteor.

    input:
    sentences: list of str.split()
    reference_list: list of [str.split()]
    """

    preds = dict()
    refs = dict()

    for i in range(len(sentences)):
        preds[i] = [" ".join([s for s in sentences[i]])]
        refs[i] = [" ".join(l) for l in reference_list[i]]

    final_scores = dict()

    scorers = [(Meteor(), "METEOR")]

    for scorer, method in scorers:
        score, scores = scorer.compute_score(refs, preds)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                final_scores[m] = scs
        else:
            final_scores[method] = scores

    meteor_scores = final_scores["METEOR"]
    return [s * 100 for s in meteor_scores]


def compute_meteor(
    reference_list: List[List[List[str]]], sentences: List[List[str]]
) -> Tuple[float, List[float]]:
    meteor_scores = compute_sentence_meteor(reference_list, sentences)
    return sum(meteor_scores) / len(meteor_scores), meteor_scores


def compute_bleu_scores(
    references: List[str], hypotheses: List[str], dataset: str,
) -> Tuple[float, List]:
    """Compute BLEU score and return the Tuple[average BLEU, list of bleu]"""

    if "comment-update" in dataset:
        refs = [DataProcessor.subtokenize_comment(ref) for ref in references]
        hypos = [DataProcessor.subtokenize_comment(hyp) for hyp in hypotheses]
    else:
        refs = references
        hypos = hypotheses
    bleu_4_sentence_scores = []
    for ref, hyp in zip(refs, hypos):
        if hyp == "":
            hyp = "<EMPTY>"
        bleu_4_sentence_scores.append(
            sentence_bleu(
                [ref.split()],
                hyp.split(),
                smoothing_function=SmoothingFunction().method2,
                auto_reweigh=True,
            )
            * 100
        )
    return (
        sum(bleu_4_sentence_scores) / float(len(bleu_4_sentence_scores)),
        bleu_4_sentence_scores,
    )


def write_outputs_to_file(
    output_dir: Path,
    code_srcs: List[str],
    code_refs: List[str],
    code_preds: List[str],
    dataset: str = None,
) -> Tuple[Path]:
    """Write inference results to files to run BLEU and GLEU scripts."""
    ref_file = output_dir / "ref.txt"
    src_file = output_dir / "src.txt"
    pred_file = output_dir / "pred.txt"
    with open(ref_file, "w") as f:
        for ref in code_refs:
            if "comment-update" in dataset:
                r = DataProcessor.subtokenize_comment(ref)
            else:
                r = ref
            f.write(r)
            f.write("\n")

    with open(src_file, "w") as f:
        for src in code_srcs:
            if "comment-update" in dataset:
                s = DataProcessor.subtokenize_comment(src)
            else:
                s = src
            f.write(s)
            f.write("\n")

    with open(pred_file, "w") as f:
        for pred in code_preds:
            if "comment-update" in dataset:
                p = DataProcessor.subtokenize_comment(pred)
            else:
                p = pred
            f.write(p)
            f.write("\n")
    return src_file, ref_file, pred_file


def process_moedit_outputs(
    ref_file: Union[Path, str],
    pred_file: Union[Path, str],
    src_file: Union[Path, str],
    tokenizer: str,
    nbest=1,
) -> EvalOutputs:
    """Process the edit model's raw output and return the list. For each example, model will predicte nbest

    return: src_list, pred_list, ref_list, edit_list
    """
    import sentencepiece as spm

    code_refs = [
        " ".join(x.strip().split())
        for x in open(ref_file, "r", encoding="utf-8").readlines()
    ]  # List[str]
    edit_preds = [
        " ".join(x.strip().split())
        for x in open(pred_file, "r", encoding="utf-8").readlines()
    ]  # List[str]
    code_srcs = [
        x.strip().split("</s>")[0].rstrip()
        for x in open(src_file, "r", encoding="utf-8").readlines()
    ]
    code_ctxts = [
        "</s>".join(x.strip().split("</s>")[1:]).strip()
        for x in open(src_file, "r", encoding="utf-8").readlines()
    ]

    producer = EditSeqProducer(tokenizer)
    assert len(code_refs) == len(edit_preds) / nbest == len(code_srcs)

    code_preds = []

    for i in range(len(code_refs)):
        sp = spm.SentencePieceProcessor(
            str(Macros.model_dir / "sentencepiece" / "sentencepiece.edit.bpe.model")
        )
        src_sub_tok = " ".join(sp.encode(code_srcs[i], out_type=str))
        for bs in range(nbest):
            edit_index = nbest * i + bs
            pred_str = producer.reconstruct_edit_sequence(
                src_sub_tok.split(), edit_preds[edit_index].split()
            )  # sub-toks
            code_pred = sp.decode(pred_str.split())
            if code_pred == "":
                code_preds.append("<EMPTY>")
            else:
                code_preds.append(code_pred)
        # end bs
    # end for

    return EvalOutputs(code_refs, code_srcs, code_preds, code_ctxts, edit_preds,)


def process_plbart_outputs(
    ref_file: Union[Path, str],
    pred_file: Union[Path, str],
    src_file: Union[Path, str],
    dataset: str = None,
    nbest=1,
) -> EvalOutputs:
    """Process the generation model's raw output and return the list.

    return: src_list, pred_list, ref_list
    """
    code_refs = [
        " ".join(x.strip().split())
        for x in open(ref_file, "r", encoding="utf-8").readlines()
    ]  # List[str]
    code_preds = [
        " ".join(x.strip().split())
        for x in open(pred_file, "r", encoding="utf-8").readlines()
    ]  # List[str]
    # This will be used if we use ICSE 21 code review dataset
    # if dataset == "code-review":  # </s> in source seq to indicate the line
    #     code_srcs = [
    #         "".join(x.strip().split("</s>")[:-1])
    #         for x in open(src_file, "r", encoding="utf-8").readlines()
    #     ]
    #     code_ctxts = [
    #         x.strip().split("</s>")[-1]
    #         for x in open(src_file, "r", encoding="utf-8").readlines()
    #     ]
    if True:
        code_srcs = [
            x.strip().split("</s>")[0].rstrip()
            for x in open(src_file, "r", encoding="utf-8").readlines()
        ]
        code_ctxts = [
            "</s>".join(x.strip().split("</s>")[1:]).rstrip()
            for x in open(src_file, "r", encoding="utf-8").readlines()
        ]

    assert len(code_refs) == len(code_preds) / nbest == len(code_srcs)

    return EvalOutputs(code_refs, code_srcs, code_preds, code_ctxts, None,)


def compute_metrics(
    ref_file: Union[Path, str],
    pred_file: Union[Path, str],
    src_file: Union[Path, str],
    tokenizer: str,
    dataset: str,
    model: str,
) -> Tuple[EvaluationMetrics, EvalOutputs]:
    """Evaluate model's performance and return the metrics."""

    eval_outputs = process_plbart_outputs(ref_file, pred_file, src_file)
    code_refs, code_preds, code_srcs, code_ctxts = (
        eval_outputs.refs,
        eval_outputs.preds,
        eval_outputs.srcs,
        eval_outputs.ctxs,
    )
    # Compute metrics
    acc, xmatch_scores_list = tokenized_xmatch(code_refs, code_preds, dataset)
    copy_pct = pct_model_copy(code_srcs, code_preds, dataset)
    sari_score, sari_scores_list = compute_sari(
        code_srcs, code_refs, code_preds, dataset
    )
    bleu_score, bleu_scores_list = compute_bleu_scores(code_refs, code_preds, dataset)
    # CodeBLEU
    if "comment-update" in dataset:
        # CodeBLEU not used in comment-update dataset
        CodeBLEU = -1.0
        codebleu_scores_list = [-1.0] * len(code_preds)
        # Meteor used here
        meteor, meteor_scores_list = compute_meteor(
            reference_list=[
                [DataProcessor.subtokenize_comment(ref).split()] for ref in code_refs
            ],
            sentences=[
                DataProcessor.subtokenize_comment(pred).split() for pred in code_preds
            ],
        )
    else:
        meteor = -1.0
        meteor_scores_list = [-1.0] * len(code_preds)
        ref_tks, pred_tks = (
            [[ref_toks.split()] for ref_toks in code_refs],
            [pred_toks.split() for pred_toks in code_preds],
        )
        CodeBLEU = code_bleu_evaluator.corpus_code_bleu(ref_tks, pred_tks)
        codebleu_scores_list = code_bleu_evaluator.code_bleu_scores_list(
            ref_tks, pred_tks
        )

    output_dir = Path(os.path.dirname(os.path.realpath(ref_file)))
    src_output_file, ref_output_file, pred_output_file = write_outputs_to_file(
        output_dir, code_srcs, code_refs, code_preds, dataset
    )
    gleu_score, gleu_scores_list = compute_gleu(
        len(code_preds), src_output_file, ref_output_file, pred_output_file,
    )

    return (
        EvaluationMetrics(
            acc,
            xmatch_scores_list,
            bleu_score,
            bleu_scores_list,
            CodeBLEU,
            codebleu_scores_list,
            gleu_score,
            gleu_scores_list,
            sari_score,
            sari_scores_list,
            meteor,
            meteor_scores_list,
            copy_pct,
        ),
        eval_outputs,
    )


def calculate_metrics_after_rerank(
    model: str, dataset: str, beam_size=20, mode="test", reranker="CodeT5",
):
    """Calculate metrics on test set after rerank."""

    rerank_results = IOUtils.load_json_stream(
        Macros.results_dir
        / "reranks"
        / f"{mode}-{dataset}-{model}-top-{beam_size}-rerank-{reranker}-results.json"
    )

    output_dir = Macros.model_dir / model / dataset
    hyp_output_file = output_dir / f"top-{beam_size}-rerank.output.hyp"

    with open(hyp_output_file, "w+") as f:
        for example in rerank_results:
            rerank_pred = example["rerank_pred"]
            f.write(f"{rerank_pred}\n")
    # end with

    # calculate eval metrics
    evaluation_metrics, _ = compute_metrics(
        output_dir / "output.ref",
        hyp_output_file,
        output_dir / "output.src",
        "code",
        dataset,
        f"{model}-{reranker}-rerank",
    )
    result = {
        "BLEU": evaluation_metrics.BLEU,
        "Acc-top-1": evaluation_metrics.xMatch,
        "SARI": evaluation_metrics.SARI,
        "GLEU": evaluation_metrics.GLEU,
        "CodeBLEU": evaluation_metrics.CodeBLEU,
        "METEOR": evaluation_metrics.METEOR,
        "COPY-PCT": evaluation_metrics.COPY,
    }
    IOUtils.dump(
        Macros.results_dir / f"results-{dataset}-{model}-rerank-{reranker}.json", result
    )
    # Dump score lists
    score_lists = {
        "BLEU": evaluation_metrics.BLEU_list,
        "Acc-top-1": evaluation_metrics.xMatch_list,
        "SARI": evaluation_metrics.SARI_list,
        "GLEU": evaluation_metrics.GLEU_list,
        "CodeBLEU": evaluation_metrics.CodeBLEU_list,
        "METEOR": evaluation_metrics.METEOR_list,
    }
    IOUtils.dump(
        Macros.results_dir / f"scores-{dataset}-{model}-rerank-{reranker}.json",
        score_lists,
        IOUtils.Format.json,
    )


def run_evaluation(
    dataset: str, model: str, tokenizer: str = "code",
):
    """Run evaluation and write metrics to file."""

    ref_file = Macros.model_dir / model / dataset / "output.ref"
    pred_file = Macros.model_dir / model / dataset / "output.hyp"
    src_file = Macros.model_dir / model / dataset / "output.src"

    evaluation_metrics, _ = compute_metrics(
        ref_file, pred_file, src_file, tokenizer, dataset, model,
    )
    result = {
        "BLEU": evaluation_metrics.BLEU,
        "Acc-top-1": evaluation_metrics.xMatch,
        "SARI": evaluation_metrics.SARI,
        "GLEU": evaluation_metrics.GLEU,
        "CodeBLEU": evaluation_metrics.CodeBLEU,
        "METEOR": evaluation_metrics.METEOR,
        "COPY-PCT": evaluation_metrics.COPY,
    }
    IOUtils.dump(Macros.results_dir / f"results-{dataset}-{model}.json", result)
    # Dump score lists
    score_lists = {
        "BLEU": evaluation_metrics.BLEU_list,
        "Acc-top-1": evaluation_metrics.xMatch_list,
        "SARI": evaluation_metrics.SARI_list,
        "GLEU": evaluation_metrics.GLEU_list,
        "CodeBLEU": evaluation_metrics.CodeBLEU_list,
        "METEOR": evaluation_metrics.METEOR_list,
    }
    IOUtils.dump(
        Macros.results_dir / f"scores-{dataset}-{model}.json",
        score_lists,
        IOUtils.Format.json,
    )


def pct_model_copy(code_srcs: List[str], code_preds: List[str], dataset: str):
    """Check percent that the system simply copy input."""
    length = len(code_srcs)
    count = 0
    copy_results = []
    for i in range(length):
        r = code_srcs[i]
        p = code_preds[i]
        if "comment-update" in dataset:
            p_subtokens = DataProcessor.subtokenize_comment(p)
            r_subtokens = DataProcessor.subtokenize_comment(r)
        else:
            p_subtokens = p
            r_subtokens = r
        if r_subtokens == p_subtokens:
            copy_results.append(1)
            count += 1
        else:
            copy_results.append(0)
    # end for
    return count / length * 100


def tokenized_xmatch(
    code_refs: List[str], code_preds: List[str], dataset: str
) -> Tuple[float, List]:
    """Check xMatch for preds and refs after tokenization."""

    length = len(code_refs)
    count = 0
    xmatch_results = []
    for i in range(length):
        r = code_refs[i]
        p = code_preds[i]
        if "comment-update" in dataset:
            p_subtokens = DataProcessor.subtokenize_comment(p)
            r_subtokens = DataProcessor.subtokenize_comment(r)
        else:
            p_subtokens = p
            r_subtokens = r
        if r_subtokens == p_subtokens:
            xmatch_results.append(1)
            count += 1
        else:
            xmatch_results.append(0)
    # end for
    return count / length * 100, xmatch_results


def compute_sari(
    src_corpus: List[str], tgt_corpus: List[str], pred_corpus: List[str], dataset=None
) -> Tuple[float, List]:
    """Computer SARI metrics for edit-related tasks. Note predictions should be predicted string sequences."""

    inp = zip(src_corpus, tgt_corpus, pred_corpus)
    scores = []

    for source, target, predicted in inp:
        if "comment-update" in dataset:
            predicted = DataProcessor.subtokenize_comment(predicted)
            source = DataProcessor.subtokenize_comment(source)
            target = DataProcessor.subtokenize_comment(target)
        scores.append(SARIsent(source, predicted, [target]) * 100)

    return sum(scores) / float(len(scores)), scores


def compute_gleu(
    test_data_size: int, src_file: Path, ref_file: Path, pred_file: Path
) -> Tuple[float, List]:
    """Compute GLEU metrics."""
    command = "python2.7 {}/scripts/compute_gleu -s {} -r {} -o {} -d".format(
        Macros.gleu_dir, src_file, ref_file, pred_file
    )
    output = subprocess.check_output(command.split())

    output_lines = [
        l.strip() for l in output.decode("utf-8").split("\n") if len(l.strip()) > 0
    ]
    l = 0
    while l < len(output_lines):
        if output_lines[l][0] == "0":
            break
        l += 1

    scores = np.zeros(test_data_size, dtype=np.float32)
    while l < test_data_size:
        terms = output_lines[l].split()
        idx = int(terms[0])
        val = float(terms[1])
        scores[idx] = val
        l += 1
    scores = np.ndarray.tolist(scores * 100)
    return sum(scores) / float(len(scores)), scores


if __name__ == "__main__":
    LoggingUtils.setup(LoggingUtils.INFO, str(Macros.log_file))
    CLI(as_positional=False)
