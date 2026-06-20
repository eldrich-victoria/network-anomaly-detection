import pandas as pd
from typing import List, Dict

class RobustLabelEncoder:
    """
    Custom LabelEncoder that handles unseen categories gracefully
    by mapping them to a default value ('unknown').
    """
    def __init__(self, default_val: str = "unknown") -> None:
        self.default_val = default_val
        self.classes_: List[str] = []
        self.mapping: Dict[str, int] = {}
        self.inverse_mapping: Dict[int, str] = {}

    def fit(self, series: pd.Series) -> "RobustLabelEncoder":
        # Convert series to strings, handle NaNs, and get unique values
        unique_vals = list(series.astype(str).unique())
        
        # Ensure default value is in the classes
        if self.default_val not in unique_vals:
            unique_vals.append(self.default_val)
            
        self.classes_ = sorted(unique_vals)
        self.mapping = {val: idx for idx, val in enumerate(self.classes_)}
        self.inverse_mapping = {idx: val for idx, val in enumerate(self.classes_)}
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        # Convert to string and map values. If value not seen, map to default_val index
        default_idx = self.mapping[self.default_val]
        transformed = series.astype(str).map(lambda x: self.mapping.get(x, default_idx))
        return transformed

    def fit_transform(self, series: pd.Series) -> pd.Series:
        self.fit(series)
        return self.transform(series)
