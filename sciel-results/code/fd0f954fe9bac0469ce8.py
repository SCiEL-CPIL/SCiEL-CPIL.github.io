"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global state
_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    global _start_time, _call_count
    
    # Initialize time tracking
    if _start_time is None:
        _start_time = time.time()
    
    # Safety guard
    current_time = time.time()
    elapsed = current_time - _start_time
    
    # Basic finite check
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    _call_count += 1
    
    # Strategy: Cost-Adjusted Expected Hypervolume Improvement (EHVI) proxy
    # 
    # The current program uses standard EI with a mild cost penalty.
    # A different approach is to use a "Cost-Effective" metric that explicitly
    # balances the probability of finding a significantly better point against
    # the cost, but in a way that is robust to the scale of f_best.
    #
    # Let's try a "Cost-Adjusted UCB" with an adaptive beta.
    # UCB = -mu + beta * sigma
    # We want to maximize this, but penalize cost.
    # Score = UCB / cost
    #
    # However, UCB can be very aggressive.
    # Let's try a "Cost-Adjusted EI" but with a different normalization.
    #
    # Alternative Structural Change: Use a "Posterior Sampling" (Thompson Sampling) approach
    # directly on the acquisition scores, but with a twist: we sample from the posterior
    # and then pick the point that minimizes the expected cost-adjusted regret.
    #
    # Actually, let's look at the "Three Hump Camel" structure again.
    # The global minimum is at (0,0) in original space, which is (0.5, 0.5) in scaled space.
    # The local minimum is at (-0.3, 0.3) in original space, which is (0.47, 0.53) in scaled space.
    # The cost function is 1 + mean(x_scaled).
    # Cost at (0.5, 0.5) = 1 + 0.5 = 1.5.
    # Cost at (0.47, 0.53) = 1 + 0.5 = 1.5.
    # The costs are almost identical. The main issue is distinguishing the two minima.
    #
    # The GP should be able to distinguish them if it has enough points.
    # The problem with EI is that it might get stuck in the local minimum if it finds it first.
    #
    # A different structural change: Use "Probability of Improvement" (POI) with a
    # cost-aware twist. POI is P(f(x) < f_best - xi).
    # If we set xi to be a small positive value, we require a significant improvement.
    #
    # Let's try a "Cost-Adjusted POI".
    # POI = Phi((f_best - mu) / sigma)
    # Score = POI / (1 + lambda * cost)
    #
    # But POI doesn't account for the magnitude of improvement.
    #
    # Let's try "Expected Improvement" but with a "hard" cost penalty that kicks in
    # only when the EI is small.
    #
    # Actually, the most promising structural change is to use a "Bayesian Optimization
    # with Cost" (BOC) framework where the objective is to maximize the expected
    # improvement per unit cost.
    #
    # Score = EI / cost
    #
    # This is simple and directly addresses the cost constraint.
    # The current program uses EI / (1 + lambda * cost), which is a soft penalty.
    # Let's try the hard penalty: EI / cost.
    #
    # However, EI can be zero or negative (though we usually take max(EI, 0)).
    # If EI is 0, the score is 0.
    #
    # Let's also add an exploration term to avoid getting stuck.
    # Score = (EI + epsilon * sigma) / cost
    #
    # This encourages exploration when EI is low.
    #
    # Let's try this:
    # 1. Compute EI.
    # 2. Add a small exploration bonus proportional to sigma.
    # 3. Divide by cost.
    
    # Compute EI
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Exploration bonus
    # The scale of sigma depends on the problem.
    # Let's use a small multiple of the mean sigma.
    mean_sigma = np.mean(sigma)
    exploration_bonus = 0.1 * mean_sigma * (sigma / mean_sigma)  # This is just 0.1 * sigma
    
    # Score = (EI + exploration_bonus) / cost
    # To avoid division by zero, ensure cost is at least 1e-9
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = (ei + exploration_bonus) / cost_safe
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # If we are very close to timeout, return a simple score to be safe
    if elapsed > 325.0:
        return -mu / cost_safe
    
    return scores
# EVOLVE-BLOCK-END
