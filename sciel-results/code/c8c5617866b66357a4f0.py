# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        # Fallback: return UCB-like scores quickly
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu + 1.5 * sigma_safe
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Knowledge Gradient (KG) for minimization:
    # KG_i = E[max(0, f_best - min(f_best, f_i))]
    #      = E[max(0, f_best - f_i)]  (since f_best is the current best)
    # This is equivalent to EI for the current best, but we extend it
    # to account for the fact that evaluating at i updates our belief
    # about the global minimum over ALL candidates.
    #
    # Full KG: KG_i = E[f_best - min(f_best, f_i)] where the expectation
    # is over the joint posterior of all candidate values.
    #
    # Approximation: We model the posterior of each candidate value f_j
    # as N(mu_j, sigma_j^2). The minimum over all candidates is
    # min_j f_j. We approximate E[min_j f_j] using the fact that
    # for Gaussian variables, we can use a soft-min approximation.
    #
    # Key insight: KG_i ≈ E[f_best - min(f_best, f_i)]
    # = E[max(0, f_best - f_i)]
    # But a more accurate version accounts for the correlation structure.
    #
    # For computational efficiency with 2000 candidates, we use:
    # 1. EI component (local improvement)
    # 2. A "global minimum reduction" term that estimates how much
    #    evaluating at i would reduce our estimate of the global min
    
    # Compute EI (Expected Improvement) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute the expected minimum over all candidates using soft-min
    # The true minimum is min_j f_j. We approximate E[min_j f_j] using
    # the observation that for a set of Gaussian variables, the expected
    # minimum can be approximated by:
    # E[min_j f_j] ≈ mu_min - sigma_min * c, where c depends on the
    # number of candidates and their variance structure.
    
    # Better approach: Use the fact that if we evaluate at point i,
    # the new best is min(f_best, f_i). The KG is:
    # KG_i = E[f_best - min(f_best, f_i)] = E[max(0, f_best - f_i)]
    # This is exactly EI. But the full KG also considers that after
    # evaluating at i, we update the posterior for all other points.
    
    # Practical KG approximation:
    # 1. EI captures local improvement
    # 2. Add a term that rewards points where the posterior suggests
    #    the global minimum is likely to be lower than f_best
    
    # Estimate the probability that the global minimum (over all 2000
    # candidates) is below f_best. This encourages exploration in
    # regions where the posterior suggests unexplored low values.
    
    # For each candidate j, P(f_j < f_best) = Phi((f_best - mu_j) / sigma_j)
    p_below = norm.cdf((f_best - mu) / sigma_safe)
    
    # The probability that AT LEAST ONE candidate has value < f_best
    # (assuming independence, which is an approximation)
    # P(min_j f_j < f_best) = 1 - prod_j (1 - P(f_j < f_best))
    # = 1 - prod_j (1 - p_below_j)
    
    # For numerical stability, use log-space
    # log(1 - p_below_j) can be very negative when p_below_j is close to 1
    # Use log1p for stability
    log_1_minus_p = np.log1p(-np.clip(p_below, 0, 1 - 1e-15))
    log_prod = np.sum(log_1_minus_p)
    p_min_below = 1.0 - np.exp(log_prod)
    
    # The KG should reward points that are likely to reduce the global minimum.
    # A point i is valuable if:
    # 1. It has high EI (local improvement)
    # 2. It's in a region where the posterior suggests the global min could be lower
    
    # Combine EI with a global exploration term
    # The global term: how much does evaluating at i tell us about the global min?
    # Approximate by: sigma_i * p_below_i (uncertainty times probability of being good)
    
    global_term = sigma_safe * p_below
    
    # Normalize both components
    def normalize(arr):
        mn, mx = np.min(arr), np.max(arr)
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    ei_norm = normalize(ei)
    global_norm = normalize(global_term)
    
    # Adaptive weighting: 
    # - Early on (few points below f_best), explore more (weight global term higher)
    # - Later (many points below f_best), exploit more (weight EI higher)
    frac_below = np.mean(p_below > 0.5)
    
    # Smooth transition
    weight_global = 0.3 + 0.4 * (1.0 - frac_below)
    weight_ei = 1.0 - weight_global
    
    score = weight_ei * ei_norm + weight_global * global_norm
    
    # Add a small UCB component for robustness
    # UCB for minimization: -mu + beta * sigma
    # Use a moderate beta
    ucb = -mu + 1.0 * sigma_safe
    ucb_norm = normalize(ucb)
    
    # Blend in a small UCB component
    score = 0.85 * score + 0.15 * ucb_norm
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
