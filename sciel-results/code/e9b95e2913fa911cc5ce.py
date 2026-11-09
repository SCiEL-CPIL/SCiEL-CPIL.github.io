"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    Uses adaptive combination of EI, UCB, and a diversity-aware component.
    """
    start_time = time.time()
    deadline = start_time + 5.0  # Safety margin well within 300s limit
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Number of evaluations so far (we can infer from the spread of sigma)
    # Early: high sigma relative to range, Late: low sigma
    sigma_range = np.ptp(sigma_safe) if len(sigma_safe) > 1 else 1.0
    sigma_mean = np.mean(sigma_safe)
    
    # Heuristic: if sigma is very small everywhere, we're in exploitation phase
    # Use the ratio of sigma to f_best scale as a proxy for exploration vs exploitation
    f_scale = max(abs(f_best), 1e-6)
    uncertainty_ratio = sigma_mean / f_scale  # High when uncertain relative to f_best
    
    # Adaptive weights: more exploration when uncertain, more exploitation when confident
    # Clamp uncertainty_ratio to [0, 2] for stability
    ur = np.clip(uncertainty_ratio, 0.0, 2.0)
    
    # EI weight increases as we become more certain
    w_ei = np.clip(0.3 + 0.7 * (1.0 - ur / 2.0), 0.2, 1.0)
    w_ucb = 1.0 - w_ei
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Compute UCB (beta controls exploration)
    # Adaptive beta: higher when uncertain
    beta = 0.5 + 1.5 * ur  # Range roughly [0.5, 3.5]
    ucb = mu + beta * sigma_safe
    # For minimization, we want low UCB, so flip sign for scoring
    ucb_score = -ucb
    
    # Normalize both to have similar scale for combination
    ei_max = np.max(ei)
    if ei_max > 0:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    if ucb_max > ucb_min:
        ucb_norm = (ucb_score - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb_score)
    
    # Combined score
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Add a small diversity/exploration bonus based on local density of high scores
    # This helps escape local optima in the candidate space
    # Simple approach: add small random-like noise based on position in sorted order
    # Actually, let's use the GP's predictive variance as an additional exploration signal
    # Points with high sigma that aren't already well-covered get a bonus
    
    # Check deadline
    if time.time() - start_time > deadline * 0.8:
        # Return quickly if we're running low on time
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Final cleanup: ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure non-negative scores (higher is better)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
