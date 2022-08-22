from typing import *
from seutil import IOUtils, LoggingUtils
from jsonargparse import CLI

from cdt.Macros import Macros
from cdt.eval.evaluate import compute_metrics


class ExperimentsAnalyzer:
    def check_model_rerank_examples(
        self, model: str, dataset: str, mode: str, reranker: str, beam_size=20
    ):
        """Check model's output after reranking train/valid/test set and write to file."""

        # First process rerank results
        model_results_dir = Macros.model_dir / model / dataset
        rerank_results = IOUtils.load(
            Macros.results_dir
            / "reranks"
            / f"{mode}-{dataset}-{model}-top-{beam_size}-rerank-{reranker}-results.json"
        )
        if mode == "valid":
            file_ref = "valid.output"
        else:
            file_ref = "output"

        ref_file, pred_file, src_file = (
            model_results_dir / f"{file_ref}.ref",
            model_results_dir / f"{file_ref}.{reranker}.rerank-{beam_size}.hyp",
            model_results_dir / f"{file_ref}.src",
        )
        with open(pred_file, "w+") as f:
            for example in rerank_results:
                rerank_pred = example["rerank_pred"]
                f.write(f"{rerank_pred}\n")
        # end if
        evaluation_metrics, eval_outputs = compute_metrics(
            ref_file,
            pred_file,
            src_file,
            "code",
            dataset,
            f"{model}-{reranker}-rerank",
        )
        results_to_check = []

        srcs, refs, preds, ctxs = (
            eval_outputs.srcs,
            eval_outputs.refs,
            eval_outputs.preds,
            eval_outputs.ctxs,
        )
        original_preds = []
        for ex in rerank_results:
            original_preds.append(
                (ex["full-beam-pred"][0][0], ex["full-beam-pred"][0][2])
            )
        for i in range(len(srcs)):
            results_to_check.append(
                {
                    "id": i,
                    "src": srcs[i],
                    "rerank-pred": preds[i],
                    "ref": refs[i],
                    "context": ctxs[i],
                    "BLEU": evaluation_metrics.BLEU_list[i],
                    "Acc-top-1": evaluation_metrics.xMatch_list[i],
                    "SARI": evaluation_metrics.SARI_list[i],
                    "GLEU": evaluation_metrics.GLEU_list[i],
                    "CodeBLEU": evaluation_metrics.CodeBLEU_list[i],
                    "base-model-pred": original_preds[i],
                }
            )
        IOUtils.dump(
            Macros.results_dir
            / f"{mode}-{dataset}-{model}-{reranker}-rerank-outputs.json",
            results_to_check,
        )

    def check_model_outputs(self, model: str, dataset: str, mode: str):
        """Check model's output at train/valid/test set and write to file."""

        model_results_dir = Macros.model_dir / model / dataset
        if mode == "valid":
            file_ref = "valid.output"
        else:
            file_ref = "output"

        ref_file, pred_file, src_file = (
            model_results_dir / f"{file_ref}.ref",
            model_results_dir / f"{file_ref}.hyp",
            model_results_dir / f"{file_ref}.src",
        )
        evaluation_metrics, eval_outputs = compute_metrics(
            ref_file, pred_file, src_file, "code", dataset, model,
        )
        results_to_check = []

        srcs, refs, preds, ctxs = (
            eval_outputs.srcs,
            eval_outputs.refs,
            eval_outputs.preds,
            eval_outputs.ctxs,
        )
        for i in range(len(srcs)):
            results_to_check.append(
                {
                    "id": i,
                    "src": srcs[i],
                    "pred": preds[i],
                    "ref": refs[i],
                    "context": ctxs[i].strip(),
                    "BLEU": evaluation_metrics.BLEU_list[i],
                    "Acc-top-1": evaluation_metrics.xMatch_list[i],
                    "SARI": evaluation_metrics.SARI_list[i],
                    "GLEU": evaluation_metrics.GLEU_list[i],
                    "CodeBLEU": evaluation_metrics.CodeBLEU_list[i],
                    "no-changes": preds[i] == srcs[i],
                }
            )
        IOUtils.dump(
            Macros.results_dir / f"{mode}-{dataset}-{model}-outputs.json",
            results_to_check,
        )


if __name__ == "__main__":
    LoggingUtils.setup(LoggingUtils.INFO, Macros.log_file)
    CLI(ExperimentsAnalyzer, as_positional=False)
