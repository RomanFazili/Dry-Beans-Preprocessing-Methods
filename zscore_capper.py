from typing import Optional
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ZScoreCapper(BaseEstimator, TransformerMixin):
    """
    This transformer caps extreme values based on z-score thresholds.
    """
    
    def __init__(self, threshold: float = 3.0) -> None:
        self.threshold = threshold
        self.means_: Optional[np.ndarray] = None
        self.stds_: Optional[np.ndarray] = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'ZScoreCapper':
        """Fit the transformer by computing mean and standard deviation."""

        self.means_ = np.mean(X, axis=0)
        self.stds_ = np.std(X, axis=0, ddof=0)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data by capping extreme values based on z-scores."""
        
        if self.means_ is None or self.stds_ is None:
            raise ValueError("ZScoreCapper must be fitted before transform")
        
        z_scores = (X - self.means_) / self.stds_
        
        # Cap z-scores
        z_scores_capped = np.clip(z_scores, -self.threshold, self.threshold)
        
        # Convert back to original scale
        X_capped = (z_scores_capped * self.stds_) + self.means_
        
        return X_capped
