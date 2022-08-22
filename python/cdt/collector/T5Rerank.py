from seutil import LoggingUtils, BashUtils, IOUtils
from pathlib import Path
from typing import List, Any, Dict
from tqdm import tqdm
from jsonargparse import CLI

from cdt.eval.evaluate import process_plbart_outputs
from cdt.collector.DataProcessor import DataProcessor
from cdt.coditT5.CodeT5score import codeT5_score
from cdt.Macros import Macros
from cdt.coditT5.DataProcessor import DataProcessor as dps


logger = LoggingUtils.get_logger(__name__, LoggingUtils.INFO)
dp = DataProcessor()

# This class provides utility to rerank CodeT5 and CoditT5 models
class T5Rerank:
    def run_rerank(
        self, model: str, reranker: str, dataset: str, beam_size=20, mode="test",
    ):
        """Do model beam outputs rerank and write outputs to file."""

        logger.info(f"processing model's top {beam_size} predictions..")
        model_output_dir = Macros.model_dir / model / dataset
        model_topk_predictions = self.aggregate_model_topk_preds(
            model,
            model_output_dir,
            dataset,
            mode=mode,
            nbest=beam_size,
            reranker=reranker,
        )
        self.process_model_beam_outputs(
            model, dataset, mode=mode, beam_size=beam_size, reranker=reranker,
        )
        source_file = (
            Macros.data_dir / f"{reranker}-rerank" / dataset / f"{mode}.output.source"
        )
        target_file = (
            Macros.data_dir / f"{reranker}-rerank" / dataset / f"{mode}.output.target"
        )
        reranker_topk_prediction = codeT5_score(
            model_cls=reranker,
            dataset=dataset,
            source_file=source_file,
            target_file=target_file,
            beam_size=beam_size,
        )
        IOUtils.dump(
            Macros.results_dir
            / "reranks"
            / f"{mode}-{dataset}-{model}-{reranker}-rerank-top-{beam_size}-scores.json",
            reranker_topk_prediction,
        )

        rerank_results = self.do_reranker_scores(
            model_topk_predictions, reranker_topk_prediction, reranker
        )
        IOUtils.dump(
            Macros.results_dir
            / "reranks"
            / f"{mode}-{dataset}-{model}-top-{beam_size}-rerank-{reranker}-results.json",
            rerank_results,
        )

    @classmethod
    def do_reranker_scores(
        cls, beam_outputs: Any, reranker_beam_outputs: Any, reranker: str
    ):
        """
        Add f'{reranker}_likelihood' key to the beam_outputs dictionary,
        which contains the likelihood for each beam output.
        """

        # Merge with top-k results json file [(pred, score)]
        for i, output in enumerate(beam_outputs):
            reranker_scores = reranker_beam_outputs[i]
            output[f"{reranker}_likelihood"] = reranker_scores["hyps"]

        ref_list, top1_pred_list, rerank_pred_list = [], [], []
        top1_xMatch, rerank_xMatch = 0, 0
        topk_xMatch = 0

        for example in beam_outputs:
            model_preds = example["full-beam-pred"]
            reranker_preds = example[f"{reranker}_likelihood"]
            merged_preds = []
            ref_list.append(" ".join(example["ref"].split()))
            top1_pred_list.append(
                sorted(model_preds, key=lambda item: float(item[2]), reverse=True)[0][0]
            )

            for i, reranker_pred in enumerate(reranker_preds):
                full_pred, reranker_score = reranker_pred
                assert full_pred == model_preds[i][1]
                if "<unk>" in full_pred:
                    full_pred = full_pred.replace("<unk>", "\u8d54")
                merged_score = float(reranker_score) + float(model_preds[i][2])
                merged_preds.append((model_preds[i][0], full_pred, merged_score))
            rerank_best_pred = sorted(
                merged_preds, key=lambda item: item[2], reverse=True
            )[0][0]
            rerank_pred_list.append(rerank_best_pred)
            example[
                "rerank_pred"
            ] = rerank_best_pred  # add pred after reranking to the output
        # end for

        # Calculate Xmatch
        for ref, top1, rerank, topk in zip(
            ref_list, top1_pred_list, rerank_pred_list, beam_outputs
        ):
            if ref == top1:
                top1_xMatch += 1
            if ref == rerank:
                rerank_xMatch += 1
            topk_preds = [p[0] for p in topk["full-beam-pred"]]
            if ref in topk_preds:
                topk_xMatch += 1
        logger.info(f"Top1 xMatch is {top1_xMatch}, rerank xMatch is {rerank_xMatch}.")

        return beam_outputs

    @classmethod
    def process_model_beam_outputs(
        cls,
        base_model: str,
        dataset: str,
        beam_size=20,
        mode="eval",
        reranker="CodeT5",
    ):
        """Write output and input to files for rerank """

        base_model_beam_outputs = IOUtils.load(
            Macros.results_dir
            / "reranks"
            / f"{mode}-{dataset}-{base_model}-top-{beam_size}-{reranker}-rerank-beam-predictions.json"
        )
        # first extract the hypos
        target_data_list = []
        for example in base_model_beam_outputs:
            top_k_preds = example["full-beam-pred"]
            assert len(top_k_preds) == beam_size
            for i, hypo in enumerate(top_k_preds):
                target_data_list.append(hypo[1])
        # end for
        # second create directory and write down the data
        IOUtils.mk_dir(Macros.data_dir / f"{reranker}-rerank" / dataset)
        model_data_dir = Macros.data_dir / reranker / dataset
        BashUtils.run(
            f"cp {model_data_dir}/{mode}.{dataset}.buggy {Macros.data_dir}/{reranker}-rerank/{dataset}/{mode}.output.source",
            expected_return_code=0,
        )

        with open(
            Macros.data_dir / f"{reranker}-rerank" / dataset / f"{mode}.output.target",
            "w+",
        ) as f:
            for target_str in target_data_list:
                f.write(f"{target_str}\n")
        # end for

    def aggregate_model_topk_preds(
        cls,
        model: str,
        model_output_dir: str,
        dataset: str,
        mode: str = "test",
        nbest=20,
        reranker="plbart",
    ) -> List[Dict]:
        """Inspect the top k beam predictions by the model."""

        model_output_dir = Path(model_output_dir)
        ref_file, pred_file, src_file = (
            model_output_dir / "output.ref",
            model_output_dir / f"output.{nbest}.hyp",
            model_output_dir / "output.src",
        )

        if model == "CoditT5":
            p, r, s, c = cls.process_model_predictions(
                ref_file, pred_file, src_file, nbest, dataset, model,
            )
            # preprocess pred file for reranking
            cls.add_sep_to_CoditT5_pred(pred_file)
            base_model_beam_scores = codeT5_score(
                "CoditT5", dataset, src_file, pred_file, nbest,
            )
            base_model_preds = cls.collect_base_model_scores(
                s, c, r, p, nbest, reranker, base_model_beam_scores
            )
        elif model in ["CodeT5"]:
            p, r, s, c = cls.process_model_predictions(
                ref_file, pred_file, src_file, nbest, dataset, model
            )
            base_model_beam_scores = codeT5_score(
                "CodeT5", dataset, src_file, pred_file, nbest,
            )
            base_model_preds = cls.collect_base_model_scores(
                s, c, r, p, nbest, reranker, base_model_beam_scores
            )
        IOUtils.dump(
            Macros.results_dir
            / "reranks"
            / f"{mode}-{dataset}-{model}-top-{nbest}-{reranker}-rerank-beam-predictions.json",
            base_model_preds,
        )
        return base_model_preds

    @classmethod
    def add_sep_to_CoditT5_pred(cls, pred_file: Path):
        """Post process CoditT5 hyp file, to add back special <s> to the predictions."""
        processed_hyps = []
        with open(pred_file, "r") as pf:
            for pred in pf:
                if "<s>" not in pred:
                    gen, edit = dps.seperate_gen_edit(pred.strip())
                    processed_hyps.append(f"{edit} <s> {gen}")
        assert len(processed_hyps) % 20 == 0
        with open(pred_file, "w") as pf:
            for p in processed_hyps:
                pf.write(f"{p}\n")

    @classmethod
    def process_model_predictions(
        cls,
        ref_file: Path,
        pred_file: Path,
        src_file: Path,
        nbest: int,
        dataset: str,
        model: str,
    ) -> List[Dict[str, Any]]:

        eval_outputs = process_plbart_outputs(
            ref_file, pred_file, src_file, nbest=nbest, dataset=dataset,
        )
        code_srcs, code_refs, code_preds, code_context, edit_preds = (
            eval_outputs.srcs,
            eval_outputs.refs,
            eval_outputs.preds,
            eval_outputs.ctxs,
            eval_outputs.edits,
        )
        if model == "CoditT5":
            code_preds = [dps.remove_edits(p) for p in code_preds]
        if "comment-update" in dataset:
            code_preds = [DataProcessor.remove_tag_string(p) for p in code_preds]
            code_refs = [DataProcessor.remove_tag_string(r) for r in code_refs]
            code_srcs = [DataProcessor.remove_tag_string(r) for r in code_srcs]

        return code_preds, code_refs, code_srcs, code_context

    @classmethod
    def collect_base_model_scores(
        cls,
        code_srcs: List[str],
        code_ctxs: List[str],
        code_refs: List[str],
        code_preds: List[str],
        nbest: int,
        reranker: str,
        beam_model_scores: List[dict],
    ):
        """Collect and aggregate base model scores and prepare examples for ranker model."""

        results_list = []
        model_input = []
        for src, ctx in zip(code_srcs, code_ctxs):
            model_input.append(f"{src} </s> {ctx}")

        for i in tqdm(range(len(code_srcs))):
            top_k_preds = []  # List[(str, float)]

            for bs in range(nbest):
                bs_index = nbest * i + bs
                if reranker == "CoditT5":
                    edits_seq = " ".join(
                        dp.edit_producer.compute_edit_sequence(
                            code_srcs[i].replace("</s>", "").split(),
                            code_preds[bs_index].split(),
                        )
                    )
                    edit_seq_no_keep = dps.remove_keep_span(edits_seq)
                    seq_to_score = f"{edit_seq_no_keep} <s> {code_preds[bs_index]}"
                else:
                    seq_to_score = code_preds[bs_index]
                base_model_score = float(beam_model_scores[i]["hyps"][bs][1])
                top_k_preds.append(
                    (code_preds[bs_index], seq_to_score, base_model_score)
                )
            assert len(top_k_preds) == nbest

            results_list.append(
                {
                    "id": i,
                    "src": code_srcs[i],
                    "ref": code_refs[i],
                    "full-src": model_input[i],
                    "full-ref": code_refs[i],
                    "full-beam-pred": top_k_preds,
                    "pred-edits": None,
                }
            )
        return results_list


if __name__ == "__main__":
    LoggingUtils.setup(LoggingUtils.INFO, Macros.log_file)
    CLI(T5Rerank, as_positional=False)
