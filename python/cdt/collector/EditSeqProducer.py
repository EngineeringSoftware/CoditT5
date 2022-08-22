from pathlib import Path
from typing import List
from tqdm import tqdm

from cdt.collector.diff_utils import (
    compute_minimal_comment_diffs,
    compute_code_diffs,
    format_diff_spans,
    format_minimal_diff_spans,
    compute_minimal_diffs,
)
from cdt.collector.code_tokenizer import tokenize_java


class EditSeqProducer:

    MAX_LEN = 512

    def __init__(self, tokenizer: str = "code") -> None:
        self.tokenizer = tokenizer  # two approches: comment, code

    def compute_edit_sequence(
        self, old_seq: List[str], new_seq: List[str]
    ) -> List[str]:
        """Return edit sequence given old and new seuqnece."""
        if self.tokenizer == "comment":
            span, _, _ = compute_minimal_comment_diffs(old_seq, new_seq)
            return span
        elif self.tokenizer == "code":
            span, _, _ = compute_code_diffs(old_seq, new_seq)
            return span
        else:
            span, _, _ = compute_minimal_diffs(old_seq, new_seq)
            return span
        # end if

    # end func

    def reconstruct_edit_sequence(self, old_seq: List[str], edit_seq: List[str]) -> str:
        """Return new sequence after applying edit seq to old seq."""
        if self.tokenizer == "comment":
            new_seq = format_minimal_diff_spans(old_seq, edit_seq)
            return new_seq
        elif self.tokenizer == "code":
            new_seq = format_diff_spans(old_seq, edit_seq)
            return new_seq
        # end if

    # end func
