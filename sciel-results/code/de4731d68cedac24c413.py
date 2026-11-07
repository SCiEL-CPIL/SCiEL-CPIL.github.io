"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track progress across calls within a seed
_step_count = 0
_start_time = None

def acquisition(mu, sigma, f_best):
    global _step_count, _start_time
    
    # Initialize time tracking on first call
    if _start_time is None:
        _start_time = time.time()
        _step_count = 0
    
    elapsed = time.time() - _start_time
    _step_count += 1
    
    # Safety margin: return simple UCB if we're getting close to deadline
    if elapsed > 300:
        scores = -mu + 2.0 * sigma
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return scores
    
    # Ensure inputs are valid
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    if not np.isfinite(f_best):
        f_best = 1e6

    # Structural Change: Pure Adaptive UCB
    # Previous version mixed UCB, EI, Entropy, QAF.
    # This version uses a single UCB score with a beta that decays
    # from a high exploration value to a lower exploitation value.
    
    # Total optimization steps is 25 (30 evals - 5 initial).
    # We assume _step_count starts at 1 for the first acquisition call.
    # So step 1 to 25.
    # Progress: 0 at step 1, 1 at step 25.
    progress = min(1.0, max(0.0, (_step_count - 1) / 24.0))
    
    # Beta schedule:
    # Start with high exploration (beta=8.0) to map the landscape and find the global basin.
    # End with moderate exploitation (beta=1.0) to converge to the minimum.
    # The Beale function has a deep narrow minimum, so we need to be careful not to
    # reduce beta too fast, but also not to wander too much late in the run.
    beta = 8.0 * (1.0 - progress) + 1.0 * progress
    
    scores = -mu + beta * sigma
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    return scores
# EVOLVE-BLOCK-END
