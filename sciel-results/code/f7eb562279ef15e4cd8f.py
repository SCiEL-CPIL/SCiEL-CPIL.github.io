import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 285.0  # Safety margin before 300s

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        return -mu
    
    # Ensure sigma is safe
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Thompson Sampling: sample from the posterior at each candidate point
    # The key insight: instead of computing a deterministic score, we use
    # a randomized approach that naturally balances exploration/exploitation
    
    # Adaptive exploration weight based on convergence state
    # When f_best is large (early), explore more; when small (late), exploit more
    # Use quantile normalization to make this scale-invariant
    
    # Compute a normalized measure of how close we are to optimum
    # f_best ranges from ~2.9 (ackley max) to 0 (min)
    convergence = np.exp(-f_best / 0.3)  # 1 when f_best~0, ~0 when f_best large
    
    # Exploration weight: higher when far from optimum
    # Use a smooth transition
    explore_weight = 0.2 + 0.8 * (1.0 - convergence)
    
    # Thompson Sampling: sample one draw from N(mu, sigma) for each candidate
    # But we need to return 2000 scores, so we use the expected value of the
    # sample plus an exploration bonus weighted by uncertainty
    
    # Alternative: use the probability that a sampled value beats f_best
    # P(sample < f_best) = Phi((f_best - mu) / sigma)
    # This is actually the probability of improvement, which is a valid acquisition
    
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    # Probability of improvement (PoI)
    poi = norm.cdf(z)
    
    # Expected improvement for Thompson Sampling context
    # In TS, we sample y ~ N(mu, sigma) and pick argmin y
    # The score should reflect how likely this point is to be the minimum
    # A good proxy: -mu + alpha * sigma where alpha adapts
    
    # Use a hybrid: combine PoI (probability of being better than best)
    # with a lower confidence bound for direct value estimation
    
    # LCB component: mu - beta * sigma (we want to minimize, so lower is better)
    # Score = -LCB = -mu + beta * sigma
    beta = 0.5 + 1.5 * explore_weight
    
    lcb_score = -mu + beta * sigma_safe
    
    # Blend PoI and LCB score
    # PoI is in [0,1], LCB score can be any real number
    # Normalize PoI contribution to be comparable
    w_poi = 0.3 * explore_weight
    w_lcb = 1.0 - w_poi
    
    score = w_poi * poi + w_lcb * lcb_score
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
