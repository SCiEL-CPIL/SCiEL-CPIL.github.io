"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Strategy:
    1. Use Thompson Sampling (Posterior Sampling) instead of deterministic heuristics like UCB or EI.
       This involves sampling a single function value for each candidate point from the posterior
       distribution N(mu, sigma^2).
    2. Since we want to minimize the objective function, we select the candidate with the lowest
       sampled value. This implicitly balances exploration (high sigma leads to more variance in
       samples, allowing for exploration) and exploitation (low mu leads to lower expected samples).
    3. Incorporate cost-awareness by dividing the sampled value by the evaluation cost.
       However, simple division can be unstable if cost varies significantly or if sampled values
       are negative (which they are for Hartmann 3D). 
       
       A more robust cost-aware formulation for minimization via sampling is to treat the "score"
       as the expected improvement per unit cost, but since we only have one sample, we can use
       the sampled value normalized by cost. 
       
       Actually, a common cost-aware Thompson Sampling approach is to sample the function value
       f_sample ~ N(mu, sigma^2) and then choose the x that minimizes f_sample / cost(x).
       But wait: if f_sample is negative, dividing by cost (positive) makes it "more negative"
       (worse) if cost > 1, and "less negative" (better) if cost < 1.
       Cost is 1 + mean(x), so cost is in [1, 2].
       Minimizing f_sample / cost is equivalent to maximizing -f_sample / cost.
       
       Let's verify the logic:
       We want to find x that minimizes the true function f(x), but evaluations are expensive.
       A common proxy is to minimize the "expected regret per cost" or similar.
       
       However, the standard cost-aware EI is EI / cost.
       For Thompson Sampling, we sample f_t ~ p(f|x, D).
       We want to pick x that minimizes f(x) while considering cost.
       One simple approach: score = - (f_sample / cost).
       We want to maximize this score.
       If f_sample is -3.8 (good) and cost is 1.0, score is 3.8.
       If f_sample is -3.8 and cost is 2.0, score is 1.9.
       So it prefers cheaper evaluations for the same predicted quality.
       If f_sample is -3.0 (worse) and cost is 1.0, score is 3.0.
       If f_sample is -3.8 and cost is 2.0, score is 1.9.
       The first is preferred. This seems reasonable.
       
       Another perspective: We want to maximize the probability of being the new best per unit cost?
       That's complex. The linear scaling f_sample / cost is a standard heuristic for cost-aware TS.
       
       Let's stick to: Score = -f_sample / cost.
       
       Note: Since mu and sigma are in objective units, and Hartmann 3D has negative values,
       f_sample will be negative.
       We return the scores such that higher is better.
       So we return -f_sample / cost.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma)) and np.all(np.isfinite(cost))):
        return np.zeros(n, dtype=np.float64)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Thompson Sampling: Draw one sample from N(mu, sigma^2) for each candidate
    # Using standard normal random numbers
    eps = np.random.normal(0, 1, size=n)
    f_samples = mu + sigma_safe * eps
    
    # Cost-aware score: We want to minimize f, so we maximize -f.
    # We also want to minimize cost, so we divide by cost (or multiply by 1/cost).
    # Score = -f_sample / cost
    scores = -f_samples / cost_safe
    
    # Ensure finite output
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check deadline (though this is very fast)
    if time.time() > deadline:
        pass
        
    return scores
# EVOLVE-BLOCK-END
