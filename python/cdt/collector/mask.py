from typing import List, Tuple, Any
from transformers import RobertaTokenizer
from seutil import LoggingUtils
import numpy as np
import random
from jsonargparse import CLI
from tqdm import tqdm

from cdt.collector.code_tokenizer import tokenize_java
from cdt.Macros import Macros

logger = LoggingUtils.get_logger(__name__, LoggingUtils.INFO)

NL_COUNT = 401402
CODE_COUNT = 454451


class DatasetCorrupt:
    def __init__(self) -> None:
        random.seed(7)
        self.batch_size = 64
        # code
        self.code_delete_prob = (0, 0.495)
        self.code_insert_prob = (0.495, 0.71)
        self.code_replace_prob = (0.71, 1)
        self.mean_code_span_length = 6.5
        self.mean_code_span_count = 1.9
        # natural language
        self.nl_delete_prob = (0, 0.07)
        self.nl_insert_prob = (0.07, 0.18)
        self.nl_replace_prob = (0.18, 1)
        self.mean_nl_span_length = 3
        self.mean_nl_span_count = 1.4
        self.tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")

    def corrupt_nl(self, nl_file: str, output_file: str, fixed_file: str):
        """
        Corrupt the natural language sequence for pretraining. Basically use corrupt_comment() function.
        """
        with open(nl_file, "r") as f, open(output_file, "w+") as of, open(
            fixed_file, "w+"
        ) as ff:
            for comment in tqdm(f, total=NL_COUNT):
                tokenized_nl = comment.split()
                if len(tokenized_nl) > 512:
                    continue
                corrupted_nl_ids, edit_labels = self.corrupt_comment(
                    comment.strip().split()
                )
                if corrupted_nl_ids is not None:
                    corrupted_nl = self.tokenizer.convert_tokens_to_string(
                        self.tokenizer.convert_ids_to_tokens(
                            corrupted_nl_ids, skip_special_tokens=False
                        )
                    )
                    # clean and sanity check
                    corrupted_nl = " ".join(corrupted_nl.split())
                    assert len(
                        self.tokenizer.convert_tokens_to_ids(
                            self.tokenizer.tokenize(corrupted_nl)
                        )
                    ) == len(edit_labels)
                    of.write(f"{corrupted_nl}\n")
                    ff.write(f"{comment.strip()}\n")

    def corrupt_code(self, java_file: str, output_file: str, fixed_file: str):
        """
        Corrupt the code for pretraining by tokenizing Java functions, corrupting them, 
        and writing the results to output files.
        :param java_file: Path to the input file containing Java functions, one per line.
        :param output_file: Path to the output file where corrupted code will be written.
        :param fixed_file: Path to the output file where tokenized Java functions will be written.
        """
        with open(java_file, "r") as f, open(output_file, "w+") as of, open(
            fixed_file, "w+"
        ) as ff:
            for java_code in tqdm(f, total=CODE_COUNT):
                tokenized_code = tokenize_java(java_code)
                if len(tokenized_code) > 512:
                    continue
                corrupted_code_ids, edit_labels = self.corrupt_code_snippet(
                    tokenized_code
                )
                if corrupted_code_ids is not None:
                    corrupted_code = self.tokenizer.convert_tokens_to_string(
                        self.tokenizer.convert_ids_to_tokens(
                            corrupted_code_ids, skip_special_tokens=False
                        )
                    )
                    # sanity check
                    assert len(
                        self.tokenizer.convert_tokens_to_ids(
                            self.tokenizer.tokenize(corrupted_code)
                        )
                    ) == len(edit_labels)
                    try:
                        ff.write(f"{' '.join(tokenized_code)}\n")
                    except:
                        continue
                    of.write(f"{corrupted_code}\n")

    def corrupt_comment(self, comment: List[str]):
        """
        Corrupt the comment through randomly delete tokens or mask tokens etc.
        """
        try:
            length = len(comment)
            mask_indices = random_spans_noise_mask(
                length, self.mean_nl_span_count, self.mean_nl_span_length
            )
            corrupted_nl_ids, edit_labels = self.corrupt_token_ids(
                comment, np.asarray([mask_indices]), modality="nl"
            )
            return corrupted_nl_ids, edit_labels
        except:
            return None, None

    def corrupt_code_snippet(self, java_code: List[str]):
        """
        Corrupts a given Java code snippet by randomly deleting or masking tokens based on a certain distribution.
        Args:
            java_code (List[str]): A list of strings representing the Java code snippet to be corrupted.
        Returns:
            Tuple[List[str], List[int]]: A tuple containing two elements:
            - corrupted_code_ids (List[str]): The list of strings representing the corrupted Java code.
            - edit_labels (List[int]): The list of integers representing the labels for the edits made.
        """
        try:
            length = len(java_code)
            mask_indices = random_spans_noise_mask(
                length, self.mean_code_span_count, self.mean_code_span_length
            )
            corrupted_code_ids, edit_labels = self.corrupt_token_ids(
                java_code, np.asarray([mask_indices]), modality="code"
            )
            return corrupted_code_ids, edit_labels
        except:
            return None, None

    def corrupt_token_ids(
        self, input_seq: List[str], mask_indices: Any, modality: str
    ) -> Tuple[np.array, List[str]]:
        """Corrupt the input given the token-level input ids and mask indices AND return edit labels."""
        mask_start_ids, mask_end_ids = self.get_mask_start_end_ids(mask_indices)
        mask_count = len(mask_start_ids)
        edit_labels = []
        spans_to_concatenate = []
        for i in range(mask_count):
            if i == 0:
                input_ids = self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize(" ".join(input_seq[: mask_start_ids[i]]))
                )
                spans_to_concatenate.append(input_ids)
            else:
                input_ids = self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize(
                        " "
                        + " ".join(
                            input_seq[mask_end_ids[i - 1] + 1 : mask_start_ids[i]]
                        ),
                    )
                )
                spans_to_concatenate.append(input_ids)
            # needs further check
            edit_labels += ["<KEEP>"] * len(input_ids)
            # sample the corruption method
            action = self.sample_edit_action(modality)
            if action == "delete":
                # noise_length = mask_end_ids[i] - mask_start_ids[i] + 1
                noise_span = self.create_delete_span(i + 1, len(self.tokenizer))
                spans_to_concatenate.append([225])
                spans_to_concatenate.append(noise_span)
                # spans_to_concatenate.append([225])
                edit_labels += ["<DELETE>"]
                if mask_start_ids[i] != mask_end_ids[i]:
                    input_ids = self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.tokenize(
                            " "
                            + " ".join(
                                input_seq[mask_start_ids[i] : mask_end_ids[i] + 1]
                            ),
                        )
                    )
                    spans_to_concatenate.append(input_ids)
                    edit_labels += ["<KEEP>"] * len(input_ids)
            elif action == "replace":
                # noise_length = mask_end_ids[i] - mask_start_ids[i] + 1
                noise_span = self.create_replace_span(i + 1, len(self.tokenizer))
                spans_to_concatenate.append([225])
                spans_to_concatenate.append(noise_span)
                # spans_to_concatenate.append([225])
                edit_labels += ["<REPLACE>"]
            elif action == "insert":
                edit_labels[-1] = "<INSERT>"
            else:
                raise NotImplementedError
        # endfor
        input_ids_noise = np.concatenate(spans_to_concatenate, axis=-1,)
        return input_ids_noise, edit_labels

    def sample_edit_action(self, modality: str):
        """
        Use uniform distribution to sample the noise action from the common developers' used edit actions.
        """
        random_prob = random.uniform(0, 1)
        if modality == "code":
            if (
                random_prob >= self.code_delete_prob[0]
                and random_prob < self.code_delete_prob[1]
            ):
                return "delete"
            elif (
                random_prob >= self.code_insert_prob[0]
                and random_prob < self.code_insert_prob[1]
            ):
                return "insert"
            else:
                return "replace"
        elif modality == "nl":
            if (
                random_prob >= self.nl_delete_prob[0]
                and random_prob < self.nl_delete_prob[1]
            ):
                return "delete"
            elif (
                random_prob >= self.nl_insert_prob[0]
                and random_prob < self.nl_insert_prob[1]
            ):
                return "insert"
            else:
                return "replace"

    @classmethod
    def create_delete_span(cls, mask_id: int, tokenizer_size: int):
        # random_new_tokens = np.random.choice(np.arange(tokenizer_size-length, tokenizer_size), random.randint(1, length))
        mask_span_token = [tokenizer_size - mask_id]
        return mask_span_token

    @classmethod
    def create_replace_span(cls, mask_id: int, tokenizer_size: int):
        # random_new_tokens = np.random.choice(np.arange(tokenizer_size-length, tokenizer_size), random.randint(1, length))
        return [tokenizer_size - mask_id]

    @classmethod
    def get_mask_start_end_ids(
        cls, mask_indices: np.ndarray
    ) -> Tuple[List[int], List[int]]:
        """Return mask start and end ids."""
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        end_indices = mask_indices - np.roll(mask_indices, -1, axis=-1) * mask_indices
        end_indices[:, -1] = mask_indices[:, -1]

        return (
            np.nonzero(start_indices)[1].tolist(),
            np.nonzero(end_indices)[1].tolist(),
        )


def random_spans_noise_mask(
    length: int, mean_span_count: float, mean_noise_span_length: int
) -> np.array:
    """
    Generates a mask array with indices for random spans of noise within a sequence of a given length.
    Args:
        length (int): The total length of the sequence.
        mean_span_count (float): The average number of noise spans to generate.
        mean_noise_span_length (int): The average length of each noise span.
    Returns:
        np.array: A binary array of the same length as the input sequence, where 1 indicates a noise token and 0 indicates a non-noise token.
    """
    
    orig_length = length

    num_noise_tokens = int(np.round(mean_span_count * mean_noise_span_length))
    # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_noise_spans = int(np.round(num_noise_tokens / mean_noise_span_length))

    # avoid degeneracy by ensuring positive number of noise spans
    num_noise_spans = max(num_noise_spans, 1)
    num_nonnoise_tokens = length - num_noise_tokens

    # pick the lengths of the noise spans and the non-noise spans
    def _random_segmentation(num_items, num_segments):

        mask_indices = np.arange(num_items - 1) < (num_segments - 1)
        np.random.shuffle(mask_indices)
        first_in_segment = np.pad(mask_indices, [[1, 0]])
        segment_id = np.cumsum(first_in_segment)
        # count length of sub segments assuming that list is sorted
        _, segment_length = np.unique(segment_id, return_counts=True)
        return segment_length

    noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)
    interleaved_span_lengths = np.reshape(
        np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
        [num_noise_spans * 2],
    )
    span_starts = np.cumsum(interleaved_span_lengths)[:-1]
    span_start_indicator = np.zeros((length,), dtype=np.int8)
    span_start_indicator[span_starts] = True
    span_num = np.cumsum(span_start_indicator)
    is_noise = np.equal(span_num % 2, 1)
    return is_noise[:orig_length].astype(np.int8)


if __name__ == "__main__":
    LoggingUtils.setup(LoggingUtils.INFO, Macros.log_file)
    CLI(DatasetCorrupt, as_positional=False)

