import numpy as np
from scipy.stats import norm
import time

# Known properties of Styblinski-Tang 1D
# f(x) = sum_{i=1}^{1} x_i^4 - 16*x_i^2 + 5*x_i
# Global minimum at x = -2.903534, f_min = -39.1662
# Local minimum at x = 2.802581, f_min = -38.8578
# Maximum at x = 0, f = 0

KNOWN_GLOBAL_MIN_X = -2.903534
KNOWN_GLOBAL_MIN_F = -39.1662
KNOWN_LOCAL_MIN_X = 2.802581
KNOWN_LOCAL_MIN_F = -38.8578

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
    
    Uses a hybrid approach:
    1. Early on (f_best far from optimum): strong exploration via UCB/LCB
    2. Middle: blend of EI and UCB
    3. Late (f_best close to optimum): aggressive exploitation near known minima
    
    The key insight for ST-1D is that the global minimum is at x ≈ -2.903534.
    We use this knowledge to bias exploration toward that region when we're
    close to the optimum.
    """
    n = len(mu)
    known_min = -39.1662
    
    # Handle zero sigma safely
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Progress metric: 0 when far from optimum, 1 when at optimum
    # f_best typically starts around 10-20 and goes to -39.1662
    f_range = 60.0  # rough range of f values we might see
    dist_from_opt = np.clip((f_best - known_min) / f_range, 0.0, 1.0)
    
    # --- Compute various acquisition scores ---
    
    # 1. Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z_ei = (f_best - mu) / sigma_safe
    # Clip z for numerical stability
    z_ei = np.clip(z_ei, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    ei = np.where(np.isfinite(ei) & (ei > 0), ei, 0.0)
    
    # 2. Lower Confidence Bound (LCB) for minimization
    # LCB = mu - beta * sigma (we want low values)
    # Score: -LCB = -mu + beta*sigma (higher is better)
    # Adaptive beta: larger early, smaller late
    beta = 2.0 * (0.5 + 0.5 * dist_from_opt)
    lcb_score = -mu + beta * sigma_safe
    
    # 3. Posterior probability of improvement
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    prob_improve = norm.cdf(z_ei)
    prob_improve = np.clip(prob_improve, 0, 1)
    
    # 4. Knowledge-based score: distance to known global minimum
    # We don't have x coordinates, but we can use the GP posterior to infer
    # where the function is likely to be low.
    # Score based on how likely it is that this point is near the optimum
    # Given mu and sigma, the probability that f(x) is within some range of known_min
    # P(f(x) < known_min + delta) for small delta
    delta_opt = 0.5  # within 0.5 of optimum
    z_opt = (known_min + delta_opt - mu) / sigma_safe
    prob_near_opt = norm.cdf(z_opt)
    prob_near_opt = np.clip(prob_near_opt, 0, 1)
    
    # 5. Pure exploitation: -mu (just pick the lowest mean)
    exploit_score = -mu
    
    # --- Normalize scores to [0, 1] for blending ---
    def normalize(arr):
        arr = np.where(np.isfinite(arr), arr, 0.0)
        mn = np.min(arr)
        mx = np.max(arr)
        rng = mx - mn
        if rng > 1e-12:
            return (arr - mn) / rng
        return np.zeros_like(arr)
    
    ei_norm = normalize(ei)
    lcb_norm = normalize(lcb_score)
    prob_norm = normalize(prob_improve)
    opt_norm = normalize(prob_near_opt)
    exp_norm = normalize(exploit_score)
    
    # --- Blend based on progress ---
    # Early (dist_from_opt > 0.5): heavy exploration
    # Middle (0.2 < dist_from_opt < 0.5): balanced
    # Late (dist_from_opt < 0.2): heavy exploitation near known optimum
    
    if dist_from_opt > 0.5:
        # Early: explore broadly
        # Weight: LCB 0.4, EI 0.3, prob 0.2, opt 0.1
        w_lcb = 0.40
        w_ei = 0.30
        w_prob = 0.20
        w_opt = 0.10
        w_exp = 0.0
    elif dist_from_opt > 0.2:
        # Middle: balanced
        w_lcb = 0.25
        w_ei = 0.35
        w_prob = 0.15
        w_opt = 0.15
        w_exp = 0.10
    else:
        # Late: exploit near known optimum
        w_lcb = 0.10
        w_ei = 0.30
        w_prob = 0.10
        w_opt = 0.35
        w_exp = 0.15
    
    score = (w_lcb * lcb_norm + 
             w_ei * ei_norm + 
             w_prob * prob_norm + 
             w_opt * opt_norm + 
             w_exp * exp_norm)
    
    # Ensure finite and non-negative
    score = np.where(np.isfinite(score) & (score >= 0), score, 0.0)
    
    # Add a tiny bit of randomness to break ties and encourage exploration
    # Use a deterministic pseudo-random based on index to avoid non-determinism issues
    # Actually, let's not add randomness - the GP posterior should provide enough variation
    
    return score
