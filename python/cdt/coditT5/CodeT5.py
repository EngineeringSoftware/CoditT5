import transformers
from transformers import (
    RobertaTokenizer,
    T5ForConditionalGeneration,
)
from typing import List, Tuple, Dict, Optional, Union, Sequence
from jsonargparse.typing import Path_dc, Path_drw
import os
from pathlib import Path
from seutil import LoggingUtils
import torch
import torch.utils.data
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import (
    LR_SCHEDULER_REGISTRY,
    OPTIMIZER_REGISTRY,
    instantiate_class,
    SaveConfigCallback,
)
import collections
import numpy as np

from .utils import DefaultLightningCLI, ExampleDataset, post_process_edit_sequences
from cdt.Macros import Macros
from cdt.eval.evaluate import compute_bleu_scores
from cdt.collector.EditSeqProducer import EditSeqProducer
from cdt.coditT5.prediction import PredictionWriter

esp = EditSeqProducer()
logger = LoggingUtils.get_logger(__name__, LoggingUtils.INFO)

MAX_LENGTH = 512


class CodeT5DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: str,
        model: str = "CodeT5",
        batch_size: int = 2,
        eval_batch_size: int = 8,
    ):
        """
        :model_outputs: {model_name: {train: Path, test: Path}}
        """
        super().__init__()

        self.data_dir = Macros.data_dir / model / dataset
        self.dataset = dataset
        self.save_hyperparameters()
        logger.info(f"Data Module params: \n{self.hparams}")

    def setup(self, stage: Optional[str] = None):
        """Load and encode train/valid/test dataset"""

        self.tokenizer = self.trainer.lightning_module.tokenizer
        self.stage = stage
        if stage == "fit" or stage is None:
            # Process training data
            train_source_file = self.data_dir / f"train.{self.dataset}.buggy"
            train_target_file = self.data_dir / f"train.{self.dataset}.fixed"
            self.train_dataset = ExampleDataset(train_source_file, train_target_file)
            # Process validatoin data
            valid_source_file = self.data_dir / f"valid.{self.dataset}.buggy"
            valid_target_file = self.data_dir / f"valid.{self.dataset}.fixed"
            self.valid_dataset = ExampleDataset(valid_source_file, valid_target_file)

        if stage == "test":
            test_source_file = self.data_dir / f"test.{self.dataset}.buggy"
            test_target_file = self.data_dir / f"test.{self.dataset}.fixed"
            logger.info("Start to process testing data...")
            self.test_dataset = ExampleDataset(test_source_file, test_target_file)

        if stage == "validate":
            valid_source_file = self.data_dir / f"valid.{self.dataset}.buggy"
            valid_target_file = self.data_dir / f"valid.{self.dataset}.fixed"
            self.valid_dataset = ExampleDataset(valid_source_file, valid_target_file)

    def tokenizer_collate_fn(
        self, batch_data: List[Tuple[str, str]]
    ) -> Sequence[torch.Tensor]:
        """Customize collate function"""
        source_batch = [t[0] for t in batch_data]
        target_batch = [t[1] for t in batch_data]
        max_length = MAX_LENGTH
        encoded_dict = self.tokenizer(
            source_batch,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoded_dict["input_ids"]
        encoded_dict = self.tokenizer(
            target_batch,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        labels = encoded_dict["input_ids"]
        return (
            input_ids,
            labels,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.hparams.batch_size,
            num_workers=16,
            collate_fn=self.tokenizer_collate_fn,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            shuffle=False,
            batch_size=self.hparams.batch_size,
            num_workers=1,
            collate_fn=self.tokenizer_collate_fn,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.hparams.eval_batch_size,
            num_workers=0,
            collate_fn=self.tokenizer_collate_fn,
        )


class CodeT5Module(pl.LightningModule):

    # Instantiate the model
    def __init__(
        self,
        pretrained_tokenizer: Union[Path_drw, str],
        pretrained_model: Union[Path_drw, str],
        optimizer_init: dict,
        lr_scheduler_init: dict,
        version: str = "",
        output_dir=None,
        beam_size=5,
        num_return_sequences=1,
    ):
        super(CodeT5Module, self).__init__()

        if isinstance(pretrained_tokenizer, Path_drw):
            pretrained_tokenizer = os.path.relpath(
                Path(pretrained_tokenizer.abs_path), Path.cwd()
            )
        if isinstance(pretrained_model, Path_drw):
            pretrained_model = os.path.relpath(
                Path(pretrained_model.abs_path), Path.cwd()
            )

        self.save_hyperparameters()
        self.beam_size = beam_size
        self.num_return_sequences = num_return_sequences

        self.tokenizer = RobertaTokenizer.from_pretrained(
            self.hparams.pretrained_tokenizer
        )

        self.model = T5ForConditionalGeneration.from_pretrained(
            self.hparams.pretrained_model
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        logger.info(f"Model Module params: \n{self.hparams}")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        if "weight_decay" in self.hparams.optimizer_init["init_args"]:
            no_decay = ["bias", "LayerNorm.weight"]
            parameters = [
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.hparams.optimizer_init["init_args"][
                        "weight_decay"
                    ],
                },
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
        else:
            parameters = self.parameters()
        optimizer = instantiate_class(parameters, self.hparams.optimizer_init)
        lr_scheduler = instantiate_class(optimizer, self.hparams.lr_scheduler_init)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }

    def training_step(self, batch: List[torch.Tensor], batch_idx=-1):
        inputs, labels = batch
        attention_masks = ~(inputs == self.tokenizer.pad_token_id)
        outputs = self.model(
            inputs, labels=labels, attention_mask=attention_masks, return_dict=True
        )
        train_loss = outputs.loss
        self.log_dict({"loss/train": train_loss.item()}, on_step=True)

        return train_loss

    def validation_step(self, batch: List[torch.Tensor], batch_idx=-1):
        inputs, labels = batch
        attention_masks = ~(inputs == self.tokenizer.pad_token_id)
        batch_size = inputs.shape[0]
        outputs = self.model(
            inputs, attention_mask=attention_masks, labels=labels, return_dict=True
        )
        val_loss = outputs.loss
        output_sequences = self.model.generate(
            input_ids=inputs,
            attention_mask=attention_masks,
            num_beams=5,
            num_return_sequences=self.num_return_sequences,
            max_length=MAX_LENGTH,
        )
        pred_sequences = []
        target_sequences = []
        for input_ids, output_ids, label in zip(inputs, output_sequences, labels):
            pred = self.detokenize(output_ids)
            target = self.detokenize(label)
            pred_sequences.append(pred)
            target_sequences.append(target)
        _, bleu_score_list = compute_bleu_scores(
            target_sequences, pred_sequences, self.trainer.datamodule.dataset,
        )
        if self.trainer.datamodule.stage == "validate":
            return pred_sequences
        metrics_list = {"bleu/val": bleu_score_list}
        metrics_list["loss/val"] = [val_loss.item()] * batch_size

        # log the prediction of model
        s = ""
        for i in range(batch_size):
            s += f"# Example {i}\n\n"
            s += f"- gold\n```\n{target_sequences[i]}\n```\n\n"
            s += f"- pred\n```\n{pred_sequences[i]}\n```\n\n"
            s += f"- metrics\n\n"
            for k, v in metrics_list.items():
                s += f"{k}: {v[i]}\n"
            s += "\n"

        self.logger.experiment.add_text("examples/val", s, global_step=self.global_step)

        return metrics_list

    def test_step(self, batch: List[torch.Tensor], batch_idx=-1):
        inputs, labels = batch
        attention_masks = ~(inputs == self.tokenizer.pad_token_id)
        batch_size = inputs.shape[0]
        pred_sequences = []

        output_sequences = self.model.generate(
            input_ids=inputs,
            attention_mask=attention_masks,
            num_beams=self.beam_size,
            num_return_sequences=self.num_return_sequences,
            max_length=MAX_LENGTH,
        )

        for output_ids in output_sequences:
            pred = self.detokenize(output_ids)
            pred_sequences.append(pred)

        return pred_sequences

    def validation_epoch_end(self, outputs: Union[List[Dict], List[List[str]]]):
        if self.trainer.datamodule.stage == "validate":
            all_valid_preds = []
            for batch_pred in outputs:
                all_valid_preds.extend(batch_pred)
            output_file = (
                "valid.output.hyp"
                if self.num_return_sequences == 1
                else f"valid.output.{self.num_return_sequences}.hyp"
            )
            with open(f"{self.hparams.output_dir}/{output_file}", "w") as f:
                for pred in all_valid_preds:
                    f.write(f"{pred}\n")
            return
        metrics_list = collections.defaultdict(list)
        for o in outputs:
            for k in o:
                metrics_list[k] += o[k]
        metrics = summarize_metrics(metrics_list)
        self.log_dict(metrics)

    def test_epoch_end(self, outputs: List[List[str]]):
        "Write the predictions to file: output.hyp at the end of test."
        all_test_preds = []
        for batch_pred in outputs:
            all_test_preds.extend(batch_pred)
        output_file = (
            "output.hyp"
            if self.num_return_sequences == 1
            else f"output.{self.num_return_sequences}.hyp"
        )

        # write to file
        with open(f"{self.hparams.output_dir}/{output_file}", "w") as f:
            for pred in all_test_preds:
                f.write(f"{pred}\n")

    def detokenize(
        self, output_ids: torch.Tensor, skip_special_tokens: bool = True
    ) -> str:
        pred = self.tokenizer.convert_tokens_to_string(
            post_process_edit_sequences(
                self.tokenizer.convert_ids_to_tokens(
                    output_ids, skip_special_tokens=skip_special_tokens
                )
            )
        )
        return pred

    def save_pretrained(self, save_dir: Union[str, Path, Path_drw, Path_dc]):
        if isinstance(save_dir, (Path_drw, Path_dc)):
            save_dir = Path(save_dir.abs_path)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)


def summarize_metrics(
    metrics: Dict[str, Union[float, List[float]]],
) -> Dict[str, float]:
    metrics_summary = {}
    for k, v in metrics.items():
        if isinstance(v, list):
            metrics_summary[k] = float(np.mean([float(x) for x in v]))
        else:
            metrics_summary[k] = float(v)
    return metrics_summary


if __name__ == "__main__":
    LoggingUtils.setup(LoggingUtils.INFO, Macros.log_file)

    OPTIMIZER_REGISTRY.register_classes(
        transformers.optimization, torch.optim.Optimizer, override=True
    )
    LR_SCHEDULER_REGISTRY.register_classes(
        transformers.optimization, torch.optim.lr_scheduler._LRScheduler, override=True
    )

    DefaultLightningCLI(
        CodeT5Module,
        CodeT5DataModule,
        save_config_callback=SaveConfigCallback,
        prediction_writer=PredictionWriter,
        optimizers=[(None, "optimizer", "model.optimizer_init")],
        lr_schedulers=[(None, "lr_scheduler", "model.lr_scheduler_init")],
    )
