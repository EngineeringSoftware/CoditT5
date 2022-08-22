from transformers import RobertaTokenizer, T5ForConditionalGeneration
from pathlib import Path
from tqdm import tqdm
import torch
import torch.utils.data
from torch.utils.data import TensorDataset
from typing import Any, List

from cdt.Macros import Macros
from cdt.coditT5.save_pretrained import save_pretrained

pad_to_max_length = True
MAX_LENGTH = 512


def codeT5_score(
    model_cls: str,
    dataset: str,
    source_file: Path,
    target_file: Path,
    beam_size: int,
    pad_ignored: bool = True,
):
    """Give likelihood score to each reference and write to file."""

    ckpt_dir = Macros.model_dir / model_cls / dataset / "model"
    # first save model
    save_pretrained(model_cls="CodeT5", ckpt_dir=str(ckpt_dir))

    model = T5ForConditionalGeneration.from_pretrained(ckpt_dir)
    if model_cls != "CodeT5" or dataset in ["comment-update", "comment-update-full"]:
        tokenizer = RobertaTokenizer.from_pretrained(
            f"{Macros.model_dir}/codeT5Tokenizer"
        )
    else:
        tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")
    model = model.to("cuda")
    # raw input
    input_sentences = []
    output_sentences = [[] for _ in range(beam_size)]
    likelihood_scores = [[] for _ in range(beam_size)]

    with open(source_file, "r") as sf, open(target_file, "r") as tf:
        for sentence in sf:
            input_sentences.append(sentence.strip())
            for i in range(beam_size):
                for sentence in tf:
                    output_sentences[i].append(sentence.rstrip())
                    break
    # end with

    for j, outputs in enumerate(output_sentences):
        dataset = encode_sentences(tokenizer, input_sentences, outputs)
        dataloader = torch.utils.data.DataLoader(
            TensorDataset(
                dataset["input_ids"], dataset["attention_masks"], dataset["labels"]
            ),
            shuffle=False,
            batch_size=1,
        )

        for batch in tqdm(dataloader):
            input_id, attention_mask, labels = (
                batch[0].to("cuda"),
                batch[1].to("cuda"),
                batch[2].to("cuda"),
            )
            if pad_ignored:
                labels_ignored = torch.tensor(
                    [
                        [(l if l != tokenizer.pad_token_id else -100) for l in label]
                        for label in labels
                    ]
                ).to("cuda")
            else:
                labels_ignored = labels
            output = model(
                input_id,
                labels=labels_ignored,
                attention_mask=attention_mask,
                return_dict=True,
            )
            likelihood_scores[j].append(-1 * output.loss.item())

    beam_search_outputs = []
    # Further process
    data_size = len(input_sentences)
    for di in range(data_size):
        beam_hyps = []
        for bi in range(beam_size):
            score = likelihood_scores[bi][di]
            hypothesis = output_sentences[bi][di]
            beam_hyps.append([hypothesis, score])
        # end for
        beam_search_outputs.append(
            {"id": di, "hyps": beam_hyps,}
        )
    # end for
    return beam_search_outputs


def encode_sentences(
    tokenizer: Any,
    source_sentences: List[str],
    target_sentences: List[str],
    pad_to_max_length=True,
    return_tensors="pt",
):
    """Function that tokenizes a sentence
    Args: tokenizer - the T5tokenizer; source and target sentences are the source and target sentences
    Returns: Dictionary with keys: input_ids, target_ids
    """

    input_ids = []
    attention_masks = []
    target_ids = []

    for sentence in source_sentences:
        encoded_dict = tokenizer(
            sentence.strip(),
            padding="max_length" if pad_to_max_length else "longest",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors=return_tensors,
        )
        input_ids.append(encoded_dict["input_ids"])
        attention_masks.append(encoded_dict["attention_mask"])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    for sentence in target_sentences:
        encoded_dict = tokenizer(
            sentence.strip(),
            padding="max_length" if pad_to_max_length else "longest",
            max_length=MAX_LENGTH,
            truncation=True,
            return_tensors=return_tensors,
        )
        labels = encoded_dict["input_ids"]
        target_ids.append(labels)
    target_ids = torch.cat(target_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "attention_masks": attention_masks,
        "labels": target_ids,
    }

    return batch

