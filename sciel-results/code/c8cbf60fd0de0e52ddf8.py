"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = None
_DEADLINE = None

def _ensure_deadline():
    global _START_TIME, _DEADLINE
    if _START_TIME is None:
        _START_TIME = time.time()
        _DEADLINE = _START_TIME + 320.0  # 320s safety margin within 330s limit
    return _DEADLINE


def acquisition(mu, sigma, f_best):
    deadline = _ensure_deadline()
    
    # Upper Confidence Bound (UCB) for minimization.
    # We want to minimize the function, so we prefer candidates with low predicted mean (mu)
    # and high uncertainty (sigma).
    # The score is constructed such that higher values indicate higher priority.
    # Score = -mu + kappa * sigma
    # kappa controls the exploration/exploitation trade-off.
    # A value of 2.5 is a common choice for multimodal functions with limited evaluations.
    kappa = 2.5
    scores = -mu + kappa * sigma
    return scores
# EVOLVE-BLOCK-END
