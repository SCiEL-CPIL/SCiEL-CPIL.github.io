"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Precompute feature matrix for a rich function space.
# The evaluator will call equation(x, params) where params has length 10.
# We design 10 features that span a broad set of candidate functions.
# x is uniformly drawn from [-3, 3].

def _build_features(x: np.ndarray) -> np.ndarray:
    """
    Build a feature matrix of shape (n, 10) with diverse basis functions.
    Each column is a candidate feature. We choose features that are smooth
    and cover polynomial, trigonometric, exponential, and rational-like forms.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    F = np.zeros((n, 10), dtype=np.float64)
    
    # Polynomial terms (up to degree 3, centered to avoid large values)
    F[:, 0] = np.ones(n)          # constant
    F[:, 1] = x                   # linear
    F[:, 2] = x ** 2              # quadratic
    F[:, 3] = x ** 3              # cubic
    
    # Trigonometric terms
    F[:, 4] = np.sin(x)           # sine
    F[:, 5] = np.cos(x)           # cosine
    F[:, 6] = np.sin(2.0 * x)     # higher freq sine
    
    # Exponential / decay terms (scaled to be well-behaved on [-3,3])
    F[:, 7] = np.exp(-0.3 * x)    # decaying exponential
    F[:, 8] = np.exp(0.3 * x)     # growing exponential
    
    # A rational-like term: 1/(1 + x^2) which is smooth and bounded
    F[:, 9] = 1.0 / (1.0 + x ** 2)
    
    return F


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form linear combination of 10 pre-defined basis functions.
    params: array of length 10, coefficients for each feature.
    Returns: array of predictions, one per point.
    """
    F = _build_features(x)
    return F @ params
# EVOLVE-BLOCK-END
