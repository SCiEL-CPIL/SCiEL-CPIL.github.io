"""Rewritten acquisition for Bayesian optimization of Branin 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: a blend of Expected Improvement and a decaying UCB term.
    
    The UCB term provides exploration that decays over the course of the run,
    while EI provides exploitation. We estimate the step count from the magnitude
    of sigma relative to f_best, or use a fixed decay schedule based on the
    number of distinct points (approximated by the spread of mu).
    
    Since we don't know the exact step number, we use a heuristic: the variance
    of the GP posterior decreases as we add more points. We can estimate the
    'age' of the optimization by looking at how small sigma is relative to
    the scale of the problem.
    
    For Branin 2D on [-5,10]x[0,15], the function values range roughly from 0.4 to 300.
    A reasonable UCB beta that decays from ~2.0 to ~0.5 over 30 steps.
    
    We estimate the step by observing the typical magnitude of sigma. 
    Initially, sigma is large (unexplored regions). Later, sigma is small.
    Let's normalize sigma by the scale of the problem.
    
    Actually, a simpler and more robust approach: use a fixed beta that works well
    for the given number of evaluations (30 total, 5 initial, so 25 steps).
    
    A common choice is beta_t = log(t) or similar. But since we don't have t,
    let's use a hybrid: EI + beta * sigma, where beta is chosen to balance
    exploration and exploitation.
    
    From the knowledge: "pure decaying UCB" scored 2.576. Let's try a blend.
    
    Let's use: score = EI + beta * sigma
    where beta is estimated from the data. A reasonable heuristic:
    beta = max(0.5, 3.0 * exp(-mean(sigma) / (f_best + 1e-9)))
    
    This way, when sigma is large (early), beta is small (more EI).
    When sigma is small (late), beta is larger (more UCB to escape local minima).
    Wait, that's backwards. Early on, we want more exploration (larger beta).
    Late on, we want more exploitation (smaller beta).
    
    Let's invert: beta = max(0.1, 2.0 * exp(-mean(sigma) / (f_best + 1e-9)))
    When sigma is large (early), exp is small, so beta is small? No, that's wrong.
    
    Let's think again. Early on, sigma is large. We want large beta.
    Late on, sigma is small. We want small beta.
    So beta should be proportional to sigma? No, that's what UCB does.
    
    Let's just use a fixed beta. The knowledge says decaying UCB worked well.
    Let's use beta = 1.0 as a starting point, which is a common choice.
    
    Actually, let's use the number of points as a proxy. We can estimate it
    by looking at the entropy of the posterior, but that's complex.
    
    Let's try: score = EI + 1.0 * sigma
    This is a standard UCB-EI blend.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB term with a fixed beta
    beta = 1.0
    ucb = beta * sigma_safe
    
    # Blend: EI + UCB
    score = ei + ucb
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
