from pathlib import Path
import re
from typing import List, Tuple
from seutil import LoggingUtils, IOUtils, BashUtils
from jsonargparse import CLI
import sentencepiece as spm
from tqdm import tqdm
import csv
from typing import NamedTuple
from pathlib import Path

from cdt.Macros import Macros
from cdt.collector.code_tokenizer import tokenize_java
from cdt.collector.EditSeqProducer import EditSeqProducer
from cdt.collector.diff_utils import EDIT_TOKENS


class RawData(NamedTuple):
    source: str = None
    target: str = None
    context: str = None


class DataProcessor:
    # data are processed based on EditT5 dataset
    logger = LoggingUtils.get_logger(__name__, LoggingUtils.INFO)

    MAX_LEN = 512

    edit_producer = EditSeqProducer("code")

    def process_dataset(
        self,
        dataset: str,
        source_data_dir: Path = Macros.raw_data_dir,
        target_data_dir: Path = Macros.data_dir / "CoditT5",
    ):
        """Process dataset for CoditT5."""

        for split in ["train", "valid", "test"]:
            source_input_file = source_data_dir / dataset / f"{split}.{dataset}.buggy"
            source_output_file = source_data_dir / dataset / f"{split}.{dataset}.fixed"

            model_input_file = target_data_dir / dataset / f"{split}.{dataset}.buggy"
            model_output_file = target_data_dir / dataset / f"{split}.{dataset}.fixed"
            model_target_file = target_data_dir / dataset / f"{split}.{dataset}.seq"

            model_inputs, model_outputs, model_targets = [], [], []

            with open(source_input_file, "r") as input_file, open(
                source_output_file, "r"
            ) as output_file:
                for input_line, output_line in zip(input_file, output_file):
                    source = input_line.split("<s>")[0].strip()
                    target = output_line.strip()
                    context = "</s>".join(input_line.split("<s>")[1:]).strip()

                    input, output, target = self.process_raw_data_point(
                        RawData(source, target, context), self.MAX_LEN,
                    )

                    model_inputs.append(input)
                    model_outputs.append(output)
                    model_targets.append(target)

            with open(model_input_file, "w+") as input_f, open(
                model_output_file, "w+"
            ) as output_f, open(model_target_file, "w+") as target_f:
                for input, output, target in zip(
                    model_inputs, model_outputs, model_targets
                ):
                    input_f.write(input + "\n")
                    output_f.write(output + "\n")
                    target_f.write(target + "\n")
            # end for

    def process_raw_data_point(
        self, dp: RawData, is_edit=True, max_len=MAX_LEN
    ) -> Tuple[str, str]:
        """Process each example to generate data used for CoditT5 for different tasks.

        Return: Tuple[input_str, edit_str, target_str]
        """
        source = ""
        try:
            source = " ".join(dp.source.strip().split())
            context = " ".join(dp.context.strip().split())
            input = f"{source} </s> {context}"
            target_tk = " ".join(dp.target.strip().split()).split()
            tgt = " ".join(target_tk)
        except TypeError:
            source, tgt, context = (
                dp.source.encode("utf-8", "replace").decode("utf-8"),
                dp.target.encode("utf-8", "replace").decode("utf-8"),
                dp.context.encode("utf-8", "replace").decode("utf-8"),
            )
            input = f"{source} </s> {context}"
            target_tk = tgt.split()
            tgt = " ".join(target_tk)
        edits: str = " ".join(
            self.edit_producer.compute_edit_sequence(source.split(), target_tk)
        )
        edit_span = self.remove_keep_span(edits.strip())
        output = f"{edit_span} <s> {tgt}"

        return input, output, tgt

    def post_process_model_generation(
        self, dataset: str, output_file: str = "output.hyp", model: str = "CoditT5"
    ):
        """Remove the edit spans in the generations. NOTE: this is only for CoditT5 model"""

        model_output_dir = Macros.model_dir / model / dataset
        model_output_file = model_output_dir / output_file
        cleaned_preds = []
        with open(model_output_file, "r") as hf:
            for prediction in hf:
                tokenized_pred = prediction.split()
                pred_len = len(tokenized_pred)
                for i in reversed(range(pred_len)):
                    if tokenized_pred[i] in EDIT_TOKENS:
                        cleaned_preds.append(" ".join(tokenized_pred[i + 1 :]))
                        break
        BashUtils.run(f"cp {model_output_file} {model_output_dir}/raw.hyp")
        with open(model_output_file, "w+") as f:
            for pred in cleaned_preds:
                f.write(pred)
                f.write("\n")

    @classmethod
    def remove_edits(self, pred: str):
        """Removes edit tokens from the predicted string and returns the cleaned string."""

        tokenized_pred = pred.split()
        pred_len = len(tokenized_pred)
        for i in reversed(range(pred_len)):
            if tokenized_pred[i] in EDIT_TOKENS:
                return " ".join(tokenized_pred[i + 1 :])
        return ""

    @classmethod
    def seperate_gen_edit(cls, pred: str) -> Tuple[str, str]:
        """Return: generation, edits"""
        tokenized_pred = pred.split()
        pred_len = len(tokenized_pred)
        for i in reversed(range(pred_len)):
            if tokenized_pred[i] in EDIT_TOKENS:
                return (
                    " ".join(tokenized_pred[i + 1 :]),
                    " ".join(tokenized_pred[: i + 1]),
                )
        return " ".join(tokenized_pred[:]), ""

    # Helper functions

    @classmethod
    def _tokenize(cls, input: str, is_java: bool = True) -> List[str]:
        """Tokenize input sequence based on the modality (Java, NL)"""
        if is_java:
            tokenized_java = tokenize_java(input)
            try:
                assert len(tokenized_java) > 0
            except AssertionError:
                tokenized_java = []
            return tokenized_java
        else:
            return input.strip().split()

    @classmethod
    def remove_keep_span(cls, edit_seq: str) -> str:
        """Remove all the <keep> special tokens from the input string."""
        span_no_keep = []
        tokenized_edits = edit_seq.split()
        i = 0
        while i < len(tokenized_edits):
            if i >= len(tokenized_edits):
                break
            tk = tokenized_edits[i]
            if tk == "<KEEP>":
                while tokenized_edits[i] != "<KEEP_END>" and i + 1 < len(
                    tokenized_edits
                ):
                    i = i + 1
            elif tk == "<KEEP_END>":
                i += 1
                continue
            else:
                i += 1
                span_no_keep.append(tk)
        assert "<KEEP>" not in span_no_keep
        assert "<KEEP_END>" not in span_no_keep

        return " ".join(span_no_keep)


if __name__ == "__main__":
    LoggingUtils.setup(LoggingUtils.INFO, Macros.log_file)
    CLI(DataProcessor, as_positional=False)
