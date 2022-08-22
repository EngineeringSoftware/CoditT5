import re
from seutil import LoggingUtils
from typing import NamedTuple

from cdt.Macros import Macros
from cdt.collector.EditSeqProducer import EditSeqProducer


class RawData(NamedTuple):
    source: str = None
    target: str = None
    context: str = None


class DataProcessor:

    logger = LoggingUtils.get_logger(__name__, LoggingUtils.INFO)

    MAX_LEN = 512
    comment_types = ["return", "param", "summary"]

    def __init__(self, model_name: str = "CoditT5", tokenize: str = "code"):

        self.raw_data_dir = None
        self.model_data_dir = Macros.data_dir / model_name
        self.edit_producer = EditSeqProducer(tokenize)

    # Helper functions

    @classmethod
    def subtokenize_comment(cls, comment_line: str, remove_tag=True) -> str:
        """Subtokenize comments from https://github.com/panthap2/deep-jit-inconsistency-detection/blob/master/data_processing/data_formatting_utils.py"""

        if remove_tag:
            comment_line = cls.remove_tag_string(comment_line)
        comment_line = cls.remove_html_tag(
            comment_line.replace("/**", "")
            .replace("**/", "")
            .replace("/*", "")
            .replace("*/", "")
            .replace("*", "")
            .strip()
        )
        comment_line = re.findall(
            r"[a-zA-Z0-9]+|[^\sa-zA-Z0-9]|[^_\sa-zA-Z0-9]", comment_line.strip()
        )
        comment_line = " ".join(comment_line)
        comment_line = comment_line.replace("\n", " ").strip()

        tokens = comment_line.split(" ")
        subtokens = []
        for token in tokens:
            curr = re.sub("([a-z0-9])([A-Z])", r"\1 \2", token).split()
            try:
                new_curr = []
                for c in curr:
                    by_symbol = re.findall(
                        r"[a-zA-Z0-9]+|[^\sa-zA-Z0-9]|[^_\sa-zA-Z0-9]", c.strip()
                    )
                    new_curr = new_curr + by_symbol

                curr = new_curr
            except:
                curr = []
            subtokens = subtokens + [c.lower() for c in curr]

        comment_line = " ".join(subtokens)
        return comment_line.lower()

    @classmethod
    def remove_tag_string(cls, line: str) -> str:
        search_strings = [
            "@return",
            "@ return",
            "@param",
            "@ param",
            "@throws",
            "@ throws",
        ]
        for s in search_strings:
            line = line.replace(s, "").strip()
        return line

    @classmethod
    def remove_html_tag(cls, line: str):
        SPECIAL_TAGS = [
            "{",
            "}",
            "@code",
            "@docRoot",
            "@inheritDoc",
            "@link",
            "@linkplain",
            "@value",
        ]
        clean = re.compile("<.*?>")
        line = re.sub(clean, "", line)

        for tag in SPECIAL_TAGS:
            line = line.replace(tag, "")

        return line
