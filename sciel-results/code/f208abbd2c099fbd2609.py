"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Quantile-based exploration strategy (Top 5% Uncertainty Gate):
    1. Compute empirical percentile ranks of sigma.
    2. Identify candidates in the top 5% of uncertainty.
    3. Filter these candidates to those with predicted mean below the current best (f_best).
    4. If valid candidates exist, score them by negative mean (prioritizing exploitation of promising uncertain regions).
       Candidates outside this set are penalized heavily.
    5. If no valid candidates exist, fallback to a standard UCB score.
    """
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Compute percentile ranks of sigma (0 to 1, higher = more uncertain)
    # np.argsort(np.argsort(x)) gives ranks. 
    # We divide by (n-1) to get [0, 1] range.
    sigma_ranks = np.argsort(np.argsort(sigma)).astype(np.float64) / max(n - 1, 1)
    
    # Threshold for top 5% uncertainty
    tau = 0.95
    
    # Identify candidates in the top uncertainty band
    in_top_band = sigma_ranks >= tau
    
    # Identify candidates with promising means (better than current best)
    # Note: We use strict inequality. If mu == f_best, it's not better.
    below_best = mu < f_best
    
    # Valid candidates: High uncertainty AND promising mean
    valid = in_top_band & below_best
    
    # Initialize scores with a very low value (penalty)
    # We use -1e9 to ensure valid candidates (which have scores ~ -mu) are always preferred.
    # We add a small UCB component to the invalid scores to maintain some order among them,
    # though they are all effectively ignored if valid candidates exist.
    beta_fallback = 1.0
    ucb_scores = -mu + beta_fallback * sigma
    
    scores = np.full(n, -1e9, dtype=np.float64)
    
    if np.any(valid):
        # For valid candidates, we want to select the one with the lowest mean.
        # Score = -mu.
        # To break ties or slightly prefer higher uncertainty within the valid set,
        # we can add a small multiple of the rank.
        # However, the directive emphasizes "selecting points... effectively...".
        # Pure -mu is the most direct interpretation of "lowest mean".
        scores[valid] = -mu[valid]
        
        # For invalid candidates, keep them very low.
        # We can set them to -1e9 + ucb_scores to preserve some relative order,
        # but they will never be selected over valid ones.
        scores[~valid] = -1e9 + ucb_scores[~valid]
    else:
        # Fallback: No candidate is in the top 5% uncertainty AND has mean < f_best.
        # This implies the GP is either very confident (low sigma everywhere) 
        # or the predicted means are all above f_best (exploit phase or overconfident GP).
        # Use standard UCB to explore.
        scores = ucb_scores
        
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
