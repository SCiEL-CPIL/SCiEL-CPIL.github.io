"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimization of Griewank 2D.
    
    Uses a blend of Expected Improvement (EI) and Upper Confidence Bound (UCB)
    with adaptive weighting that shifts from exploration to exploitation
    as the search progresses (indicated by how close f_best is to 0).
    
    The cost-aware scaling divides by the evaluation cost to prefer
    cheaper evaluations when scores are comparable.
    
    Key design choices:
    1. EI for exploitation: rewards points with high probability of beating f_best
    2. UCB for exploration: rewards points with high uncertainty
    3. Adaptive blend: early on (f_best far from 0), more exploration;
       later (f_best close to 0), more exploitation
    4. Cost normalization: divide by cost to account for varying evaluation costs
    5. Strict penalty on sigma in EI to prevent noise-dominated regions
    """
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-12)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma
    
    # Standard normal PDF and CDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Expected Improvement (for minimization)
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Apply a strict penalty on the uncertainty term in EI
    # When sigma is large relative to the potential improvement, 
    # we discount the exploration component
    improvement_potential = np.maximum(f_best - mu, 1e-12)
    sigma_ratio = sigma / improvement_potential
    
    # Penalty factor: 1.0 when sigma is small, decreases as sigma grows
    # Use a smooth decay: exp(-0.5 * sigma_ratio^2) for Gaussian-like penalty
    penalty = np.exp(-0.5 * sigma_ratio ** 2)
    
    # Penalized EI
    ei_penalized = (f_best - mu) * Phi_z + penalty * sigma * phi_z
    ei_penalized = np.maximum(ei_penalized, 0.0)
    
    # UCB: mu - kappa * sigma (for minimization, we want low mu and high sigma)
    # Actually for minimization UCB should be: mu - kappa * sigma
    # But we want to balance, so use: -mu + kappa * sigma (higher is better for exploration)
    # Let's use a proper UCB for minimization:
    # UCB_min = mu - kappa * sigma (lower is better)
    # But our score is "higher = evaluate next", so we want:
    # score_ucb = -mu + kappa * sigma (higher sigma and lower mu -> higher score)
    
    # Adaptive kappa: larger early, smaller late
    # f_best starts around 0.5-1.0 for Griewank, goes to ~0
    # Use f_best as a proxy for progress
    # When f_best is high (early), kappa is large (more exploration)
    # When f_best is low (late), kappa is small (more exploitation)
    
    # Normalize f_best to [0, 1] range for adaptive kappa
    # Griewank on [-10,10]^2 has values roughly in [0, 1]
    f_best_norm = np.clip(f_best, 0.0, 1.0)
    
    # kappa ranges from 3.0 (early) to 0.5 (late)
    kappa = 0.5 + 2.5 * f_best_norm
    
    # UCB score (higher is better)
    ucb_score = -mu + kappa * sigma
    
    # Normalize both EI and UCB to comparable scales
    # Use a soft normalization based on the current data
    ei_max = np.max(ei_penalized) if np.max(ei_penalized) > 1e-12 else 1.0
    ucb_max = np.max(ucb_score) - np.min(ucb_score) if np.max(ucb_score) > np.min(ucb_score) + 1e-12 else 1.0
    
    ei_normalized = ei_penalized / ei_max
    ucb_normalized = (ucb_score - np.min(ucb_score)) / ucb_max
    
    # Adaptive blend weight: more EI as f_best decreases
    # weight_ei goes from 0.3 (early) to 0.8 (late)
    weight_ei = 0.3 + 0.5 * (1.0 - f_best_norm)
    weight_ucb = 1.0 - weight_ei
    
    # Combined score
    combined = weight_ei * ei_normalized + weight_ucb * ucb_normalized
    
    # Cost-aware: divide by cost
    # cost is in (0, 1] range typically (1 + mean(x_scaled))
    # Use a softer cost penalty to not over-penalize
    scores = combined / np.maximum(cost, 1e-9)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure non-negative
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
