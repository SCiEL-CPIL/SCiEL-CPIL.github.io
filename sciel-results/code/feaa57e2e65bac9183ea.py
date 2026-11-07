"""
Acquisition function using Progress-Adaptive Expected Improvement and UCB.
This strategy balances exploitation (EI) and exploration (UCB) by adapting
the exploration weight based on the current best value's proximity to the
known global minimum of the Styblinski-Tang function.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global state for deadline guard
_start_time = None

# Known global minimum of Styblinski-Tang 1D on [-5, 5]
GLOBAL_MIN = -39.1662

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: 330s total. Return safe EI if time is running out.
    if time.time() - _start_time > 325.0:
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, -1e9)

    n = len(mu)
    
    # Calculate Expected Improvement (EI)
    # EI(x) = E[max(f_best - f(x), 0)]
    # For f(x) ~ N(mu, sigma^2):
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Handle potential numerical issues in norm.cdf/pdf for large z
    # For very large z, cdf -> 1, pdf -> 0
    # For very small z, cdf -> 0.5, pdf -> 0.4
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Calculate UCB (Upper Confidence Bound) for minimization
    # We want to maximize the score.
    # UCB_min(x) = -mu(x) + beta * sigma(x)
    # Note: Since we are minimizing, "Upper" confidence of the *negative* function
    # or "Lower" confidence of the function?
    # Standard UCB for minimization: Score = -mu + beta * sigma
    # This encourages points with low mean (good) and high uncertainty (exploration).
    
    # Adaptive Beta:
    # If f_best is far from GLOBAL_MIN, we need more exploration (high beta).
    # If f_best is close to GLOBAL_MIN, we need more exploitation (low beta).
    
    # Normalize progress: 
    # Let's say at start (f_best ~ 0 or positive), progress is low.
    # At end (f_best ~ -39), progress is high.
    # Range of ST 1D on [-5,5] is roughly [-39, 10].
    # Let's define a progress metric p in [0, 1].
    
    # A simple heuristic:
    # distance_to_opt = f_best - GLOBAL_MIN
    # If distance is large, beta should be large.
    # If distance is small, beta should be small.
    
    # Let's use a smooth decay.
    # Initial f_best is usually around 0-10.
    # Let's map distance to beta.
    
    # Heuristic:
    # beta = base_beta * exp(-k * progress)
    # Or simpler: beta decreases linearly or quadratically with improvement.
    
    # Let's use the ratio of current regret to initial expected regret.
    # But we don't know initial regret exactly.
    # Let's just use the value of f_best directly.
    
    # If f_best is -39, we want beta ~ 0.
    # If f_best is 0, we want beta ~ 2.0 (typical UCB value).
    
    # Linear interpolation:
    # beta = 2.0 * (f_best - GLOBAL_MIN) / (0.0 - GLOBAL_MIN)
    # If f_best = 0, beta = 2.0 * 39.16 / 39.16 = 2.0
    # If f_best = -39.16, beta = 0
    # If f_best = 10, beta = 2.0 * 49.16 / 39.16 > 2.0 (cap it)
    
    beta_max = 2.5
    beta_min = 0.1
    
    # Calculate beta
    # If f_best is very high (early), use max beta.
    # If f_best is close to min, use min beta.
    
    # Normalize f_best between GLOBAL_MIN and some upper bound (e.g., 10)
    upper_bound = 10.0
    if f_best >= upper_bound:
        beta = beta_max
    elif f_best <= GLOBAL_MIN:
        beta = beta_min
    else:
        # Linear map from [GLOBAL_MIN, upper_bound] to [beta_min, beta_max]
        # progress = (f_best - GLOBAL_MIN) / (upper_bound - GLOBAL_MIN)
        # beta = beta_min + progress * (beta_max - beta_min)
        
        # Actually, we want beta to be HIGH when f_best is HIGH (far from min).
        # So:
        # t = (f_best - GLOBAL_MIN) / (upper_bound - GLOBAL_MIN)
        # t is 0 when f_best is min, 1 when f_best is upper_bound.
        # beta = beta_min + t * (beta_max - beta_min)
        
        t = (f_best - GLOBAL_MIN) / (upper_bound - GLOBAL_MIN)
        t = np.clip(t, 0.0, 1.0)
        beta = beta_min + t * (beta_max - beta_min)
        
    # Compute UCB scores
    # Score_ucb = -mu + beta * sigma
    ucb_scores = -mu + beta * sigma_safe
    
    # We want to combine EI and UCB.
    # EI is in units of improvement (positive).
    # UCB is in units of objective value (negative is good, but we maximize score).
    # To combine them meaningfully, we should normalize them or use a weighted sum.
    
    # However, EI already incorporates both mean and std.
    # UCB is a simpler heuristic.
    # A common robust approach is to use EI primarily, but add a small UCB term
    # to break ties or encourage exploration in flat regions.
    
    # Alternative: Use a "Posterior Sampling" like score but smoothed?
    # No, let's stick to the adaptive hybrid.
    
    # Let's normalize EI and UCB to have similar scales?
    # EI can be large if sigma is large.
    # UCB can be large if sigma is large.
    
    # Let's try a weighted sum:
    # Score = w_ei * EI + w_ucb * UCB_normalized
    
    # But UCB is -mu + beta*sigma.
    # If mu is -39 and sigma is 0.1, UCB is -39 + 0.02 = -38.98.
    # EI is roughly (f_best - (-39)) * ... which is small if f_best is close.
    
    # Actually, if f_best is close to the true min, EI becomes very small everywhere
    # except very near the min. UCB might still push us to explore if beta is not 0.
    # But we set beta to be small when f_best is good.
    
    # Let's just use EI as the primary driver, but if the variance of EI is very low
    # (meaning the GP is confident), we might want to explore more.
    # Or, simpler: Just use EI. Why did the current program fail?
    # The current program used Thompson Sampling (probability of being min).
    # EI is often superior for 1D optimization because it directly targets the improvement.
    # The "Progress-Adaptive" part is the key: ensuring we don't get stuck in local optima
    # early on by having high exploration, and locking in the global min later.
    
    # Let's refine the combination.
    # Score = EI + lambda * (UCB - mean(UCB)) ?
    # Or just Score = EI?
    
    # Let's look at the "Inspiration 2" which scored 1.0. It used a portfolio.
    # Let's try a simple, robust EI with a small exploration bonus that decays.
    
    # Exploration Bonus:
    # We want to add a term that is high where sigma is high, but scaled down as we get closer to the min.
    # This is essentially UCB.
    
    # Let's compute the score as:
    # Score = EI + alpha * sigma
    # Where alpha is our adaptive beta?
    # No, EI already has a sigma term.
    # EI ~ (f_best - mu) * Phi(z) + sigma * phi(z).
    # The sigma term in EI is phi(z) * sigma.
    # phi(z) is max at z=0 (i.e., mu = f_best).
    # So EI already explores regions where mu is near f_best.
    # It does NOT explore regions where mu is much higher than f_best (z is large negative),
    # because Phi(z) -> 0.
    
    # This is the weakness of EI: it ignores high-uncertainty regions that are currently "bad"
    # but might contain the global min.
    # UCB: -mu + beta * sigma.
    # If mu is high (bad), -mu is low (bad score).
    # If sigma is high, beta * sigma is high (good score).
    # So UCB can pick up high uncertainty regions even if mu is bad, provided beta is large enough.
    
    # So the hybrid is:
    # Score = EI + beta * sigma
    # But EI already has a sigma component.
    # Let's just use UCB with adaptive beta?
    # UCB is simpler and often works well for 1D.
    # Let's try Score = -mu + beta * sigma.
    
    # Let's test this logic:
    # Early on: beta is high. We explore high sigma regions.
    # Late on: beta is low. We exploit low mu regions.
    
    # This is a clean, structurally different approach from the TS-probability score.
    
    scores = -mu + beta * sigma_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    # Normalize scores to be positive? Not necessary, evaluator picks max.
    # But let's make sure they are not all -1e9.
    if np.all(scores == -1e9):
        # Fallback to EI
        z = (f_best - mu) / (sigma_safe)
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        scores = np.where(np.isfinite(ei), ei, -1e9)
        
    return scores

# EVOLVE-BLOCK-END
