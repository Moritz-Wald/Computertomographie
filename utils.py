import os
from itertools import product
from pathlib import Path

import numpy as np


def load_vector(file: str | Path) -> np.ndarray:
    vec = list()
    with open(file, 'r') as f:
        for line in f.readlines():
            try:
                vec.append(float(line))
            except ValueError as e:
                if not line:
                    continue
                arr = list(map(float, filter(None, line.strip().split(' '))))
                if arr:
                    vec.append(np.array(arr))
    return np.array(vec, dtype=np.float64)


def write_vector(matrix: np.ndarray, file: str | Path, sep: str | None = '\t', in_cwd: bool = True) -> None:
    with open(Path(os.getcwd()) / file if in_cwd else file, 'a') as f:
        for i, j in product(range(matrix.shape[0]), range(matrix.shape[1])):
            f.write(sep.join(map(str, [i, j, matrix[i, j]])) + '\n')
