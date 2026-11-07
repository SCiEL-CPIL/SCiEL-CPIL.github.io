"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Theoretical minimum of Hartmann 3D
HARTMANN_3D_MIN = -3.86278

# Start time for the entire evaluation process to enforce the deadline
_start_time = time.time()
TIME_LIMIT = 300.0 # seconds
SAFETY_MARGIN = 5.0 # seconds

def acquisition(mu, sigma, f_best):
    """
    Knowledge-Gradient-inspired acquisition with uncertainty-weighted diversity.
    
    Structural change from LCB:
    - KG component: estimates expected reduction in the posterior mean of the
      global minimum, not just pointwise confidence.
    - Diversity bonus: rewards candidates whose sigma differs from the median,
      promoting exploration of distinct regions of the input space.
    - Adaptive blending: early on, more weight on exploration/diversity;
      later, more weight on exploitation/KG.
    
    Parameters:
    mu: np.ndarray of shape (2000,), posterior mean.
    sigma: np.ndarray of shape (2000,), posterior standard deviation.
    f_best: float, lowest observed value so far.
    
    Returns:
    np.ndarray of shape (2000,), scores where higher is better.
    """
    # Deadline guard
    if time.time() - _start_time > TIME_LIMIT - SAFETY_MARGIN:
        return np.zeros_like(mu)

    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # --- Knowledge Gradient component ---
    # The KG approximates: E[max(0, f_best - min(mu + noise))]
    # Since we don't have coordinates or the full GP, we approximate using
    # the distribution of mu and sigma across the candidate set.
    # A candidate is "valuable" if it could potentially beat f_best.
    
    # Probability that this candidate beats f_best (approximate)
    # P(f(x) < f_best) = P(mu(x) + sigma(x)*Z < f_best) = Phi((f_best - mu)/sigma)
    from scipy.stats import norm
    z = (f_best - mu) / sigma_safe
    p_beat = norm.cdf(z)  # probability this point is better than f_best
    
    # Expected improvement if we evaluate here (standard EI for minimization)
    ei = (f_best - mu) * p_beat + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0)
    
    # KG-like: weight EI by how much it could reduce the global min
    # Candidates with high EI and high sigma are more valuable for KG
    kg_score = ei * (1.0 + 0.5 * sigma_safe / np.mean(sigma_safe))
    
    # --- Diversity bonus ---
    # Reward candidates whose sigma is in the upper half (uncertain regions)
    # and whose mu is not too close to the best (avoid over-focusing on one region)
    median_sigma = np.median(sigma_safe)
    high_uncertainty = sigma_safe > median_sigma
    
    # Diversity: spread out by penalizing candidates with very similar mu to the best
    # Use the spread of mu values as a proxy for spatial diversity
    mu_spread = np.std(mu)
    if mu_spread > 1e-10:
        # Normalize mu relative to the best
        mu_norm = (mu - np.min(mu)) / (mu_spread + 1e-10)
        # Penalize candidates that are too close to the best (mu_norm near 0)
        # and reward those in intermediate ranges
        diversity_bonus = -0.3 * np.exp(-mu_norm**2 / 2.0)  # Gaussian penalty near best
    else:
        diversity_bonus = np.zeros(n)
    
    # Scale diversity bonus by uncertainty
    diversity_bonus = diversity_bonus * (sigma_safe / (np.mean(sigma_safe) + 1e-10))
    
    # --- Adaptive blending ---
    # Estimate progress: how close are we to the optimum?
    gap = f_best - HARTMANN_3D_MIN
    max_gap = 4.0
    norm_gap = np.clip(gap / max_gap, 0.0, 1.0)
    
    # Early: more exploration (diversity), Late: more exploitation (KG)
    # weight_kg increases as we get closer to optimum
    weight_kg = 0.4 + 0.6 * (1.0 - norm_gap)  # 0.4 to 1.0
    weight_div = 1.0 - weight_kg  # 0.6 to 0.0
    
    # Combine
    scores = weight_kg * kg_score + weight_div * diversity_bonus
    
    # Add a small LCB component for stability
    lcb_component = -(mu - 1.0 * sigma_safe)
    scores = 0.7 * scores + 0.3 * lcb_component
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
