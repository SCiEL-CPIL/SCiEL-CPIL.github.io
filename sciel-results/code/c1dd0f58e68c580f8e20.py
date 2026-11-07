import numpy as np
import time

# Global time tracking
_start_time = time.time()
TIME_LIMIT = 300.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB (Upper Confidence Bound) acquisition function.
    
    Score = (mu + beta * sigma) / cost
    
    For minimization, we want low mu and high sigma (uncertainty).
    Standard UCB for maximization is mu + beta * sigma.
    For minimization, we can use -mu + beta * sigma or (f_best - mu) + beta * sigma.
    However, the directive suggests UCB(x) = mu(x) + beta * sigma(x).
    Since we are minimizing, a high "score" should correspond to a point that is likely to be good (low value) or uncertain.
    A common adaptation for minimization is to maximize: (f_best - mu) + beta * sigma.
    This encourages points with low predicted mean (improvement) and high uncertainty.
    """
    # Check time
    current_time = time.time()
    if current_time - _start_time > TIME_LIMIT - 5:
        # Emergency fallback: favor low mu and low cost
        return -mu / cost

    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-8)

    # Beta for exploration
    # The directive suggests beta = 2.0 as a starting point.
    # In GP-UCB, beta is often adaptive, but fixed is simpler and often robust.
    # For minimization, we want to explore where sigma is high.
    # The term (f_best - mu) handles exploitation.
    # The term beta * sigma handles exploration.
    
    beta = 2.0
    
    # UCB score for minimization:
    # We want to select x that minimizes f(x).
    # A good proxy is maximizing the potential improvement: (f_best - mu) + beta * sigma.
    # This is equivalent to EI but without the non-linearity of the normal CDF/PDF,
    # which can saturate.
    
    ucb_score = (f_best - mu) + beta * sigma
    
    # Cost-aware score
    # We want higher scores to be selected.
    # High Potential Improvement / Low Cost = High Score.
    score = ucb_score / cost
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    
    return score
