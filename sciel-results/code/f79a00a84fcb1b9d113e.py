import numpy as np
import time
from scipy.stats import norm

_T0 = time.time()
_DEADLINE = _T0 + 325.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function that dynamically blends EI and UCB.
    
    Strategy:
    1. Calculate EI for minimization.
    2. Calculate UCB (with a tuned beta).
    3. Normalize both to comparable scales.
    4. Blend them based on the relative uncertainty (sigma) vs the gap to best.
       - If the best is very good (f_best close to 0) and uncertainty is low, favor exploitation (EI).
       - If uncertainty is high relative to the gap, favor exploration (UCB).
    """
    if time.time() > _DEADLINE:
        return np.zeros_like(mu)

    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-10)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-10)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, np.inf)

    # 1. Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is CDF and phi is PDF of standard normal
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues in exp
    z_clipped = np.clip(z, -50, 50)
    
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # 2. Upper Confidence Bound (UCB) for minimization
    # We want to minimize mu - beta * sigma
    # Score to maximize: beta * sigma - mu
    # Let's use a beta that is adaptive. 
    # A common heuristic is beta proportional to log(n) or similar, but here we don't have n directly.
    # We can use a fixed beta or one based on the scale of the problem.
    # Given the domain [-5, 5] and sphere function, values are in [0, 50].
    # Let's try beta = 1.5 or 2.0. The prompt mentions inspiration 1/2 scored high with EI/UCB.
    # Let's use a slightly higher beta to encourage exploration initially, 
    # but the blend will handle the balance.
    beta = 2.0
    ucb_scores = beta * sigma - mu
    ucb_scores = np.where(np.isfinite(ucb_scores), ucb_scores, -np.inf)

    # 3. Normalize scores to [0, 1] range for blending
    # EI is already non-negative.
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    if ei_max > ei_min:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ei)

    ucb_min = np.min(ucb_scores)
    ucb_max = np.max(ucb_scores)
    if ucb_max > ucb_min:
        ucb_norm = (ucb_scores - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb_scores)

    # 4. Dynamic Weighting
    # Determine how much to trust exploration vs exploitation.
    # If the average sigma is high relative to the range of mu, we are uncertain -> more UCB.
    # If the average sigma is low, we are confident -> more EI.
    
    # A simple heuristic: 
    # weight_ucb = 0.5 + 0.5 * tanh(mean_sigma / (mean_sigma + 1.0)) 
    # This transitions from 0.5 (balanced) to 1.0 (pure UCB) as sigma increases.
    # However, we also want to consider f_best. If f_best is very low (close to 0), 
    # we should be more conservative (more EI).
    
    mean_sigma = np.mean(sigma)
    
    # Scale factor: typical sigma in this problem might be small once converged.
    # Let's say if mean_sigma > 1.0, we are still exploring a lot.
    # If mean_sigma < 0.1, we are exploiting.
    
    # Let's use a sigmoid-like transition based on mean_sigma.
    # x = mean_sigma / 1.0
    # w_ucb = 1 / (1 + exp(-x))
    # At x=0 (sigma=1), w=0.5. At x=2 (sigma=2), w~0.88. At x=-2 (sigma=0.01), w~0.12.
    
    x = mean_sigma / 1.0
    w_ucb = 1.0 / (1.0 + np.exp(-x))
    
    # Also, if f_best is very small (close to 0), we should reduce exploration weight.
    # If f_best < 0.1, we are close to the optimum.
    # Let's dampen w_ucb if f_best is very small.
    if f_best < 0.1:
        # Strongly favor exploitation
        w_ucb *= 0.5
    elif f_best < 0.5:
        w_ucb *= 0.8

    w_ei = 1.0 - w_ucb

    # Ensure weights are within [0, 1]
    w_ucb = np.clip(w_ucb, 0.0, 1.0)
    w_ei = np.clip(w_ei, 0.0, 1.0)

    # Blend
    scores = w_ei * ei_norm + w_ucb * ucb_norm

    # Rescale back to a reasonable magnitude for the optimizer
    # The original scores (EI and UCB) had different magnitudes.
    # By normalizing, we lost the absolute scale.
    # Let's map the normalized blend [0, 1] back to a scale comparable to the original scores.
    # A simple way: multiply by the max of the original max scores.
    max_original = max(np.max(ei), np.max(ucb_scores))
    if max_original > 0:
        scores = scores * max_original
    else:
        scores = scores

    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
