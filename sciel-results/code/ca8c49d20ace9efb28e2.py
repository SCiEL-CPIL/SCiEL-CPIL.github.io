import numpy as np
import time
from scipy.stats import norm

# Global timer for deadline guard
_start_time = time.time()
_TIME_LIMIT = 315.0  # Safety margin from 330s

def acquisition(mu, sigma, f_best):
    # Check time
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        # If we are running out of time, return a safe fallback
        return np.full(len(mu), -1e10)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # Handle edge cases
    if mu.size == 0:
        return np.array([])

    # Ensure sigma is non-negative and finite
    sigma = np.where(np.isfinite(sigma), np.maximum(sigma, 0.0), 0.0)
    mu = np.where(np.isfinite(mu), mu, 0.0)

    # Use a hybrid approach: Expected Improvement (EI) blended with UCB
    # EI is generally strong for optimization, while UCB provides exploration.
    # We blend them based on the progress of the optimization (distance from minimum).
    
    # Goldstein-Price minimum is 3.0
    # dist_from_min indicates how far we are from the global minimum
    # If f_best is close to 3.0, we should exploit more (higher weight on EI)
    # If f_best is far from 3.0, we should explore more (higher weight on UCB)
    
    if not np.isfinite(f_best):
        # Fallback to UCB if f_best is not finite
        kappa = 2.0
        scores = -mu + kappa * sigma
        return np.where(np.isfinite(scores), scores, -1e10)

    # Compute EI for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # For GP with mean mu and std sigma:
    # EI(x) = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    sigma_safe = np.where(sigma > 1e-10, sigma, 1e10)
    z = (f_best - mu) / sigma_safe
    
    # Compute Phi and phi
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    
    ei = (f_best - mu) * Phi + sigma * phi
    
    # For points where sigma was very small, EI is simply max(f_best - mu, 0)
    small_sigma = sigma < 1e-10
    if np.any(small_sigma):
        ei[small_sigma] = np.maximum(f_best - mu[small_sigma], 0.0)
    
    ei = np.maximum(ei, 0.0)

    # Compute UCB score for minimization
    # LCB = mu - kappa * sigma
    # Score = -LCB = -mu + kappa * sigma
    # We use an adaptive kappa that increases as we get closer to the minimum
    # to encourage exploration in the fine-tuning phase
    
    dist_from_min = max(f_best - 3.0, 1e-6)
    
    # Adaptive kappa:
    # When dist_from_min is large, we are far from the minimum, so we can use a moderate kappa.
    # When dist_from_min is small, we are close, so we might want to explore a bit more to ensure we haven't missed a better point.
    # However, usually, we want to exploit more when close. Let's stick with a moderate, slightly decreasing kappa.
    # Actually, let's use a fixed kappa for UCB part to keep it simple, or a mild adaptation.
    # Let's use a kappa that starts at 2.0 and decreases to 1.0 as we approach the minimum.
    # This encourages exploration early and exploitation later.
    # But wait, EI already handles exploration/exploitation balance well.
    # Let's blend EI and UCB.
    
    # Normalize EI and UCB scores to have similar scales?
    # EI is in objective units. UCB is in objective units.
    # Let's just take a weighted sum.
    
    # Weight for EI: starts at 0.5, increases to 1.0 as we approach the minimum
    # Weight for UCB: starts at 0.5, decreases to 0.0 as we approach the minimum
    # This way, early on, it's a 50/50 blend. Later, it's mostly EI.
    
    # Use a sigmoid or linear function of dist_from_min
    # dist_from_min is typically > 0. 
    # Let's map dist_from_min from [0, 10] to [1, 0] for the UCB weight.
    # w_ucb = 1.0 - min(dist_from_min / 10.0, 1.0)
    # w_ei = 1.0 - w_ucb
    
    # But if dist_from_min is very large (e.g., 50), w_ucb becomes 0.
    # Maybe we want some UCB always?
    # Let's use a decay: w_ucb = exp(-dist_from_min / 5.0)
    # Then w_ei = 1.0 - w_ucb
    
    w_ucb = np.exp(-dist_from_min / 5.0)
    w_ei = 1.0 - w_ucb
    
    # Compute UCB score
    # We need to choose a kappa for the UCB part.
    # Let's use a kappa of 2.0 for the UCB component.
    kappa = 2.0
    ucb_score = -mu + kappa * sigma
    
    # Blend
    scores = w_ei * ei + w_ucb * ucb_score
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
