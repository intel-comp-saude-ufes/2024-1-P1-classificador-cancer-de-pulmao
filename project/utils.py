import os

from glob import glob
from typing import List
from typing_extensions import TypedDict


class DataFolder(TypedDict):
    features: List[str]
    labels: List[int]


def get_paths_and_labels(
    path: str,
    lab_names: List[str],
) -> DataFolder:
    data = DataFolder(
        labels=[],
        features=[],
    )
    lab_cnt = 0
    for lab in lab_names:
        paths_aux = glob(os.path.join(path, lab, "*.jpeg"))
        data["labels"] += [lab_cnt] * len(paths_aux)
        data["features"] += paths_aux
        lab_cnt += 1
    return data
