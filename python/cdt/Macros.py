import os
from pathlib import Path


class Macros:
    this_dir: Path = Path(os.path.dirname(os.path.realpath(__file__)))
    python_dir: Path = this_dir.parent
    project_dir: Path = python_dir.parent
    data_dir: Path = project_dir / "data"
    raw_data_dir: Path = project_dir / "raw_data"
    model_dir: Path = project_dir / "models"
    results_dir: Path = project_dir / "results"
    log_file: Path = python_dir / "experiment.log"
    gleu_dir: Path = this_dir / "gleu"

