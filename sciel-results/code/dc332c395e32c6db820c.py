"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Probability of Improvement (PI) with a
    rank-based exploration bonus and an Expected Improvement (EI) exploitation term.
    
    The core structural idea is to use the probability of improvement as the base
    score, which naturally balances exploitation (low mu) and exploration (high sigma)
    in a probabilistically grounded way. We then add a rank-based bonus that
    specifically targets the top quantile of uncertainty, gated by a promise condition
    (mu < f_best), to encourage exploration in regions that are both promising and
    uncertain. Finally, we blend in a small amount of EI to preserve the magnitude
    information that PI lacks.
    
    Parameters:
    mu: np.ndarray, shape (2000,), mean of the GP posterior
    sigma: np.ndarray, shape (2000,), std dev of the GP posterior
    f_best: float, current best observed value (minimum)
    
    Returns:
    scores: np.ndarray, shape (2000,), higher scores indicate higher priority
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    # Avoid division by zero
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute z-score for improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best)
    p_imp = norm.cdf(z)
    p_imp = np.where(np.isfinite(p_imp), p_imp, 0.0)
    p_imp = np.clip(p_imp, 0.0, 1.0)
    
    # Compute Expected Improvement for minimization
    phi_z = norm.pdf(z)
    ei = (f_best - mu) * p_imp + sigma_safe * phi_z
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ei = np.maximum(ei, 0.0)
    
    # Check if we're still within time budget for the rank-based component
    if time.time() < deadline:
        # Compute rank-based uncertainty bonus
        # Get rank of each sigma value (higher rank = more uncertain)
        ranks = np.argsort(np.argsort(sigma_safe))
        
        # Normalize ranks to [0, 1]
        n = len(sigma_safe)
        if n > 1:
            rank_norm = ranks / (n - 1)
        else:
            rank_norm = np.array([0.0])
        
        # Promise gate: only apply bonus if the mean is below current best
        promise_mask = (mu < f_best).astype(float)
        
        # Rank-based bonus: targets top 10% of uncertainty
        # Smoothly increases from 0 at rank 0.85 to 1.0 at rank 1.0
        bonus_weight = np.clip((rank_norm - 0.85) / 0.15, 0.0, 1.0)
        
        # Apply promise gate
        bonus = bonus_weight * promise_mask
        
        # Scale bonus relative to typical EI magnitude
        # Use a fraction of sigma for the bonus magnitude
        bonus_scale = 0.5 * sigma_safe
        
        rank_bonus = bonus * bonus_scale
        
        # Combine: PI base + rank bonus + EI blend
        # PI provides the probabilistic foundation
        # Rank bonus encourages exploration in uncertain, promising regions
        # EI blend preserves magnitude information
        # Weights: 0.5 for PI, 0.3 for rank bonus, 0.2 for EI
        # Normalize EI to be comparable to PI (which is in [0,1])
        # Use a soft scaling: EI / (EI + 1) to bound it
        ei_normalized = ei / (ei + 1.0)
        
        scores = 0.5 * p_imp + 0.3 * rank_bonus / (rank_bonus + 1e-6) + 0.2 * ei_normalized
        
        # Ensure scores are in a reasonable range
        scores = np.where(np.isfinite(scores), scores, 0.0)
        scores = np.maximum(scores, 0.0)
    else:
        # Fallback to a simple blend if time is running out
        scores = 0.5 * p_imp + 0.5 * (ei / (ei + 1.0))
        scores = np.where(np.isfinite(scores), scores, 0.0)
        scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
