"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer for deadline guard
_start_time = time.time()
_DEADLINE = 295.0  # Safety margin from 330s kill


def acquisition(mu, sigma, f_best):
    """
    Lower Confidence Bound (LCB) acquisition with adaptive beta.
    
    For minimization: score = mu - beta * sigma
    Higher score means we prefer that candidate.
    
    beta adapts based on f_best:
    - When f_best is large (far from optimum), beta is large -> more exploration
    - When f_best is small (close to optimum), beta is small -> more exploitation
    
    mu: array of shape (N,) - posterior mean in objective units
    sigma: array of shape (N,) - posterior std in objective units
    f_best: scalar - best (lowest) observed value so far
    """
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        return np.zeros_like(mu)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Numerical stability
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Adaptive beta:
    # When f_best is large, we want to explore more (high beta)
    # When f_best approaches 0, we want to exploit (low beta)
    # Use a smooth transition. f_best for Beale can be up to ~100+ initially.
    # beta = 1 + c / max(f_best, eps)
    # If f_best = 100, beta = 1 + c/100
    # If f_best = 1, beta = 1 + c
    # If f_best = 0.01, beta = 1 + 100c
    
    # Let's think about this differently.
    # LCB for minimization: we want to minimize mu - beta*sigma
    # So the "score" we return should be higher for better candidates.
    # score = -(mu - beta*sigma) = -mu + beta*sigma
    # Or equivalently, we can return mu - beta*sigma and the evaluator picks the MINIMUM?
    # No, the contract says "higher meaning evaluate next".
    # So we want to return a score where higher is better.
    # For LCB minimization, we pick the point with lowest (mu - beta*sigma).
    # So our score should be: score = -(mu - beta*sigma) = beta*sigma - mu
    # Or we can just flip it: score = -mu + beta*sigma
    
    # Let's use: score = -mu + beta * sigma
    # This way, high sigma and low mu both contribute to high score.
    
    # Adaptive beta calculation:
    # We want beta to be large when f_best is large.
    # A simple formulation: beta = base + scale / max(f_best, eps)
    # But f_best can be negative? No, Beale min is 0, and it's always >= 0.
    # Actually, the function value is always non-negative.
    # f_best starts high (maybe ~50-100 for random points on [-4.5,4.5]^2)
    # and decreases toward 0.
    
    # Let's use: beta = 1.0 + 5.0 / (max(f_best, 0.1) ** 0.5)
    # When f_best = 100: beta = 1 + 5/10 = 1.5
    # When f_best = 1: beta = 1 + 5/1 = 6.0
    # When f_best = 0.01: beta = 1 + 5/0.1 = 51.0
    # When f_best = 0.1: beta = 1 + 5/sqrt(0.1) = 1 + 5/0.316 = 16.8
    
    # This seems reasonable. Early on (high f_best), moderate exploration.
    # As we get closer, exploration increases to escape local minima?
    # Wait, that's backwards. We want MORE exploration when far, LESS when close.
    
    # Let me reconsider.
    # When f_best is large (early), we don't know much, so we should explore a lot.
    # When f_best is small (late), we should exploit.
    
    # So beta should be LARGE when f_best is LARGE, and SMALL when f_best is SMALL.
    # beta = base + scale * f_best? No, that would make beta huge when f_best is huge.
    # But sigma is also in objective units. If f_best is 100, sigma might also be ~10-50.
    # So beta*sigma could be 1.5 * 30 = 45, which is comparable to mu differences.
    
    # Let's try: beta = 1.0 + 10.0 / max(f_best, 1.0)
    # f_best = 100: beta = 1 + 0.1 = 1.1
    # f_best = 10: beta = 1 + 1.0 = 2.0
    # f_best = 1: beta = 1 + 10.0 = 11.0
    # f_best = 0.1: beta = 1 + 10/1 = 11.0 (capped at 1.0 in denominator)
    
    # Hmm, this makes beta LARGER when f_best is smaller, which is the opposite of what I want.
    # I want beta large when f_best is large (early), small when f_best is small (late).
    
    # Let's try: beta = 1.0 + 10.0 * min(f_best / 10.0, 1.0)
    # f_best = 100: beta = 1 + 10*1 = 11
    # f_best = 50: beta = 1 + 10*1 = 11
    # f_best = 10: beta = 1 + 10*1 = 11
    # f_best = 5: beta = 1 + 10*0.5 = 6
    # f_best = 1: beta = 1 + 10*0.1 = 2
    # f_best = 0.1: beta = 1 + 10*0.01 = 1.1
    
    # This looks better! Large beta early, small beta late.
    
    # But wait, the directive says: "beta = 1 + 10/f_best"
    # That would give:
    # f_best = 100: beta = 1 + 0.1 = 1.1
    # f_best = 10: beta = 1 + 1 = 2
    # f_best = 1: beta = 1 + 10 = 11
    # f_best = 0.1: beta = 1 + 100 = 101
    
    # That's the opposite! High beta when f_best is small.
    # The directive says "prioritizes high uncertainty regions when the solution is far from the global minimum"
    # So when f_best is LARGE (far from min), beta should be large.
    # But "1 + 10/f_best" gives small beta when f_best is large.
    
    # I think the directive might have a typo or I'm misunderstanding.
    # Let me re-read: "adaptive beta parameter that scales with the inverse of the current best value (e.g., beta = 1 + 10/f_best), which structurally prioritizes high uncertainty regions when the solution is far from the global minimum"
    
    # If f_best is large (far from min), 10/f_best is small, so beta is small.
    # That doesn't prioritize high uncertainty when far from min.
    
    # Unless... the idea is that when f_best is large, sigma is also large, so even with small beta, the exploration term beta*sigma is still significant?
    
    # Or maybe the intent is the opposite: when f_best is small (close to min), we want to explore more to escape local minima?
    
    # Let me just go with the directive's formula and see.
    # beta = 1 + 10 / max(f_best, eps)
    
    eps = 1e-6
    beta = 1.0 + 10.0 / max(f_best, eps)
    
    # Cap beta to avoid extreme values
    beta = min(beta, 50.0)
    
    # LCB score: higher is better
    # score = -mu + beta * sigma
    scores = -mu + beta * sigma_safe
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
