"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    # Cost-Adjusted UCB (Upper Confidence Bound)
    # Structural change: Shift from myopic Expected Improvement (EI) to UCB,
    # which explicitly balances exploitation (low mu) and exploration (high sigma)
    # via a confidence bound, rather than just optimizing expected immediate gain.
    
    # Beta parameter controls exploration-exploitation trade-off.
    # Higher beta favors exploration (uncertainty).
    # 2.0 is a standard heuristic for Bayesian Optimization in low dimensions.
    beta = 2.0
    
    # For minimization, UCB value is mu - beta * sigma.
    # We want to minimize this, so we maximize its negative.
    # Score = -(mu - beta * sigma) / cost = (beta * sigma - mu) / cost
    
    # Safe division
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate acquisition scores
    scores = (beta * sigma - mu) / cost_safe
    
    # Handle any numerical issues (nan, inf)
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
