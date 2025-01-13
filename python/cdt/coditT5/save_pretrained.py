from pathlib import Path
from typing import Optional, Union
from jsonargparse import CLI
from jsonargparse.typing import Path_dc, Path_drw
from seutil import LoggingUtils

logger = LoggingUtils.get_logger(__name__, LoggingUtils.INFO)


def locate_ckpt(ckpt_dir: Path) -> Optional[Path]:
    """Locate the checkpoint files in the directory."""
    ckpt_files = list(ckpt_dir.glob("*.ckpt"))
    if len(ckpt_files) == 0:
        ckpt_file = None
        logger.info(f"No checkpoint found in {ckpt_dir}")
    elif len(ckpt_files) == 1:
        ckpt_file = ckpt_files[0]
        logger.info(f"Found one checkpoint in {ckpt_dir}: {ckpt_file.name}")
    else:
        ckpt_files = [f for f in ckpt_files if f.name != "last.ckpt"]
        ckpt_file = sorted(ckpt_files, key=lambda x: x.stat().st_mtime)[-1]
        logger.warning(
            f"Multiple checkpoints found in {ckpt_dir}: {[x.name for x in ckpt_files]}; picking the latest modified: {ckpt_file.name}"
        )
    return ckpt_file


def save_pretrained(
    model_cls: str,
    ckpt_dir: Path_drw,
    ckpt_name: str = None,
    output_dir: Optional[Union[Path_drw, Path_dc]] = None,
):
    """Save the pretrained model from the checkpoint."""
    ckpt_dir = Path_drw(ckpt_dir)
    ckpt_dir = Path(ckpt_dir.abs_path)
    if ckpt_name:
        ckpt_path = ckpt_dir / ckpt_name
    else:
        ckpt_path = locate_ckpt(ckpt_dir)
    if output_dir is not None:
        output_dir = Path(output_dir.abs_path)
    else:
        output_dir = ckpt_dir
    if model_cls == "CodeT5":
        from cdt.coditT5.CodeT5 import CodeT5Module

        model = CodeT5Module.load_from_checkpoint(ckpt_path)
        model.save_pretrained(output_dir)
    else:
        raise ValueError(f"Unknown model class: {model_cls}")


if __name__ == "__main__":
    CLI(save_pretrained, as_positional=False)
