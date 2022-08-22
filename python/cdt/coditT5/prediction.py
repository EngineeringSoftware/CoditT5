import collections
import concurrent
import os
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, Union, Any

import pytorch_lightning as pl
import seutil as su
from pytorch_lightning.callbacks import BasePredictionWriter
from recordclass import RecordClass

from cdt.eval.evaluate import run_evaluation

logger = su.LoggingUtils.get_logger(__name__, su.LoggingUtils.DEBUG)


class Prediction(RecordClass):
    """Prediction at one data"""

    id: int = -1
    data: str = ""


class PredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: Union[Path, str],
        no_compute_metrics: bool = True,
        dataset: str = "",
        model: str = "",
    ):
        super().__init__(write_interval="epoch")
        self.no_compute_metrics = no_compute_metrics
        self.output_dir = Path(output_dir)
        su.io.mkdir(self.output_dir)
        self.temp_dir = self.output_dir / "temp"
        su.io.mkdir(self.temp_dir)
        self.dataset = dataset
        self.model_name = model

    def write_on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        results: List[List[List[Any]]],
        batch_indices: Optional[Sequence[Sequence[Sequence[int]]]],
    ):
        # Collect preds, and put into a file according to current global rank

        preds: List[str] = []
        for dl_batch_preds in results:
            for batch_preds in dl_batch_preds:
                if isinstance(batch_preds, list):
                    for pred in batch_preds:
                        preds.append(pred)
                else:
                    preds.append(batch_preds)

        su.io.dump(
            self.temp_dir / f"{pl_module.global_rank}.pkl", preds,
        )

        # Wait all processes to finish prediction
        trainer.training_type_plugin.barrier("prediction")

        if pl_module.global_rank == 0:
            id2pred = {}
            for rank in range(trainer.world_size):
                for pred in su.io.load(self.temp_dir / f"{rank}.pkl"):
                    id2pred[pred.id] = pred.data
            if sorted(id2pred.keys()) != list(range(len(id2pred))):
                logger.warning(f"Prediction ids are not continuous")
            preds = [id2pred[i] for i in sorted(id2pred.keys())]

            # Dump predictions
            logger.info("Saving predictions")
            with open(self.output_dir / "output.hyp", "w+") as f:
                for pred in preds:
                    f.write(f"{pred}\n")

            if not self.no_compute_metrics:
                # Compute metrics
                logger.info("Computing and saving metrics")

                run_evaluation(
                    self.output_dir / "output.ref",
                    self.output_dir / "output.hyp",
                    self.output_dir / "output.src",
                    tokenizer="code",
                    dataset=self.dataset,
                    model=self.model_name,
                )

            # Delete temp directory
            su.io.rmdir(self.temp_dir)
