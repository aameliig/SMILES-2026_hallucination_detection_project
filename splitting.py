import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """
    Returns 5 different (train, val, test) splits using stratified k‑fold
    The same test set is used for all folds (fixed hold‑out)
    """
    idx_all = np.arange(len(y))
    # Separate a fixed test set
    idx_train_val, idx_test = train_test_split(
        idx_all,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    y_train_val = y[idx_train_val]

    # Create 5 folds from the remaining data
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    splits = []
    for train_idx, val_idx in skf.split(idx_train_val, y_train_val):
        idx_train = idx_train_val[train_idx]
        idx_val = idx_train_val[val_idx]
        splits.append((idx_train, idx_val, idx_test))
    return splits