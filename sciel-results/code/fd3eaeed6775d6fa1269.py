import numpy as np
from scipy.stats import norm
import time

# Global state for tracking elapsed time and potentially for UCB-like structures
# However, since we only have mu, sigma, f_best, cost, we must be stateless or use very lightweight state.
# The directive suggests a structural change: Multi-Armed Bandit (UCB1) over spatial partitions.
# But we don't have access to previous evaluations' x-coordinates or values directly, only the GP outputs (mu, sigma) for 2000 candidates.
# The GP already aggregates the history.
# A pure UCB over a grid would require knowing which grid cells have been visited and their average reward.
# We don't have the history of x and f(x). We only have the current GP surrogate.
# Therefore, a direct implementation of UCB over spatial partitions based on *history* is not possible with the given signature.
# We must approximate the "bandit" aspect using the GP uncertainty and mean.
#
# The directive says: "Replace the deterministic acquisition logic with a Multi-Armed Bandit (UCB1) over spatial partitions... rather than local GP uncertainty".
# This implies we should discretize the domain and assign scores based on cell-level statistics.
# But again, we lack the cell-level history.
#
# Let's re-read carefully: "mu, sigma and cost have shape (2000,). Return 2000 finite scores... No coordinates are passed."
# This is a critical constraint. We do NOT have the x-coordinates of the 2000 candidates.
# Wait, the task says: "The evaluator scores 2000 new uniform candidates with acquisition".
# Usually, in such BO loops, the `acquisition` function is called with the GP predictions for a set of candidate points.
# If coordinates are not passed, we cannot map them to a grid.
#
# Let's look at the "INSPIRATIONS". They use `time`.
# Maybe the "structural change" is not a full grid UCB because it's impossible without x.
# Perhaps the "UCB" idea is interpreted differently: treating each candidate as an arm? No, that's just PI/EI.
#
# Let's reconsider the "No coordinates are passed" constraint.
# If I cannot access x, I cannot do spatial partitioning.
# Is it possible that the "cost" vector encodes spatial information?
# cost = 1 + mean(x_scaled).
# cost is in [1, 2].
# cost is a scalar per candidate. It does not give full x.
#
# So, a true UCB over spatial partitions is **impossible** with the given signature `acquisition(mu, sigma, f_best, cost)`.
# The directive might be a generic suggestion that doesn't fully apply due to the interface constraints, OR I am missing something.
# Let's look at the "Knowledge from this run": "Hybrid EI/UCB balances exploitation (EI) and exploration (UCB)... Measured: 1 -> 1.041."
# This hybrid likely uses the GP's sigma as the exploration term (like UCB) and mu as the exploitation term.
# UCB formula: mu + kappa * sigma.
# Since we are minimizing, we want low mu and high sigma (uncertainty).
# Standard UCB for minimization: score = -mu + kappa * sigma.
# Or, in terms of "regret reduction" or "improvement":
# PI: P(f(x) < f_best) = Phi((f_best - mu) / sigma).
# EI: E[f_best - f(x)]_+ = (f_best - mu) * Phi(z) + sigma * phi(z).
#
# The current program uses PI - cost_penalty. Score 0.8985.
# Parent (1.0246) used something else.
# Inspiration 1 (1.0471) likely used EI or a hybrid.
#
# Let's try a **Hybrid Acquisition** that combines Expected Improvement (EI) and a UCB-like term, adjusted for cost.
# The "UCB" part in BO context usually refers to the exploration bonus `kappa * sigma`.
# So, Score = EI + kappa * sigma - cost_penalty.
# Or simply: Score = -mu + kappa * sigma - cost_penalty.
#
# Let's analyze the cost.
# cost = 1 + 0.5 * ||x||_1 (if scaled to [0,1] and mean is average of coords? "mean(x scaled to [0,1])").
# If x in [0,10]^4, scaled to [0,1] is x/10.
# cost = 1 + mean(x/10) = 1 + (1/4) * sum(x_i)/10 = 1 + sum(x_i)/40.
# Min cost is 1 (at x=0). Max cost is 2 (at x=10).
#
# The goal is to minimize Shekel 4d.
# Shekel 4d minimum is -10.5364.
#
# Let's try to improve upon the current PI-based approach by using EI, which is generally superior to PI because it considers the *magnitude* of improvement, not just the probability.
#
# Proposed Structural Change:
# 1. Compute EI.
# 2. Add an exploration bonus based on sigma (UCB-like).
# 3. Apply a cost-aware penalty.
# 4. Use a deadline guard.
#
# Why this might jump:
# PI ignores how much better a point is. A point with 50% chance of improving by 0.01 is scored same as 50% chance of improving by 1.0.
# EI captures this.
# Adding `kappa * sigma` encourages exploring high-uncertainty regions, which is crucial for multi-modal functions like Shekel.
# The current program has NO explicit exploration term (PI is bounded by 1, and doesn't grow with sigma if mu is fixed).
#
# Let's refine the score:
# z = (f_best - mu) / sigma
# EI = (f_best - mu) * Phi(z) + sigma * phi(z)
# Exploration = kappa * sigma
# Cost_Penalty = lambda * (cost - 1)
#
# Score = EI + Exploration - Cost_Penalty
#
# We need to tune kappa and lambda.
# f_best is in objective units. Shekel values are around -10.
# sigma is in objective units.
# EI is in objective units.
# Cost penalty is in [0, 1] range (since cost is [1,2]).
# We need to scale the cost penalty to be comparable to EI.
# If EI is typically around 0.1-1.0, and cost penalty is 0-1, lambda=1 might be too strong or too weak.
# Let's assume EI is the primary driver.
#
# Also, consider that `f_best` changes.
#
# Deadline Guard:
# The function runs many times. We should track time.
# However, the function is called per evaluation step.
# "Each evaluation must finish within 300 s."
# "The evaluator KILLS the program after 330s wall-clock."
# This implies the *total* time for all evaluations is 330s?
# "While total cost is below 30... evaluator scores 2000... and evaluates the best one."
# Total cost 30. Initial 5 points.
# Cost per eval is 1-2.
# So number of evals is roughly (30-5)/1.5 ~ 16 evals.
# 16 * 300s = 4800s. This is way more than 330s.
# So the 330s is a hard wall-clock limit for the *entire* script execution?
# "The evaluator KILLS the program after 330s wall-clock."
# This likely means the total time for the BO loop is 330s.
# We have ~16 iterations. 330/16 ~ 20s per iteration.
# Each iteration calls `acquisition` once (for 2000 points).
# So `acquisition` has ~20s to run.
# NumPy operations on 2000 points are milliseconds.
# So time is not a bottleneck for the acquisition function itself.
# The bottleneck is likely the GP training/prediction, which is done by the evaluator, not us.
# We just receive mu, sigma.
# So the deadline guard might be overkill for the acquisition function itself, but good practice.
#
# Let's focus on the score formula.
#
# Previous score 0.8985 used: PI - (cost - 1).
# Let's try: EI + 0.1 * sigma - 0.5 * (cost - 1).
# Or maybe just EI is enough?
#
# Let's look at the "Hybrid EI/UCB" knowledge.
# "Hybrid EI/UCB balances exploitation (EI) and exploration (UCB)... Measured: 1 -> 1.041."
# This suggests EI + UCB term is good.
#
# Let's implement:
# 1. Safe sigma.
# 2. z-score.
# 3. EI calculation.
# 4. UCB exploration term: kappa * sigma.
# 5. Cost penalty: lambda * (cost - 1).
# 6. Combine: Score = EI + kappa * sigma - lambda * (cost - 1).
#
# Tuning:
# If f_best is -10, and we find a point with mu=-10.5, sigma=0.1.
# z = (-10 - (-10.5))/0.1 = 5.
# Phi(5) ~ 1.
# EI = 0.5 * 1 + 0.1 * ~0 = 0.5.
# Exploration = 0.1 * 0.1 = 0.01.
# Cost penalty: if cost=1.5, penalty = 0.5 * 0.5 = 0.25.
# Score = 0.5 + 0.01 - 0.25 = 0.26.
#
# If we have a point with mu=-9, sigma=1.0.
# z = (-10 - (-9))/1 = -1.
# Phi(-1) = 0.16.
# EI = (-1) * 0.16 + 1 * 0.24 = 0.08.
# Exploration = 0.1 * 1.0 = 0.1.
# Cost penalty: if cost=1.2, penalty = 0.5 * 0.2 = 0.1.
# Score = 0.08 + 0.1 - 0.1 = 0.08.
#
# The first point is better.
#
# What if we have a point with mu=-10.1, sigma=0.5.
# z = 0.1/0.5 = 0.2.
# Phi(0.2) = 0.58.
# EI = 0.1 * 0.58 + 0.5 * 0.39 = 0.058 + 0.195 = 0.25.
# Exploration = 0.1 * 0.5 = 0.05.
# Cost penalty: if cost=1.1, penalty = 0.5 * 0.1 = 0.05.
# Score = 0.25 + 0.05 - 0.05 = 0.25.
#
# The first point (mu=-10.5, sigma=0.1) still wins.
#
# Let's try to make the cost penalty adaptive or stronger?
# If we are close to the optimum, we should be careful with cost.
#
# Let's try a different structural change: **Posterior Sampling (Thompson Sampling)** approximation?
# No, we only have mu and sigma.
#
# Let's stick to the Hybrid EI/UCB.
#
# One issue: `f_best` is a scalar. `mu`, `sigma`, `cost` are arrays.
#
# Code structure:
# import numpy as np
# from scipy.stats import norm
# import time
#
# start_time = time.time()
#
# def acquisition(mu, sigma, f_best, cost):
#     # Deadline guard
#     # if time.time() - start_time > 320: return zeros
#     
#     sigma_safe = np.maximum(sigma, 1e-9)
#     z = (f_best - mu) / sigma_safe
#     
#     # Expected Improvement
#     phi = norm.pdf(z)
#     Phi = norm.cdf(z)
#     ei = (f_best - mu) * Phi + sigma_safe * phi
#     
#     # UCB Exploration Term
#     kappa = 0.1  # Tuning parameter
#     ucb = kappa * sigma_safe
#     
#     # Cost Penalty
#     lambda_cost = 0.5  # Tuning parameter
#     penalty = lambda_cost * (cost - 1.0)
#     
#     scores = ei + ucb - penalty
#     
#     # Ensure finite
#     scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
#     
#     return scores

# Let's refine the parameters.
# If EI is small, UCB might dominate.
# If cost is high, penalty might dominate.
#
# The previous score 0.8985 was PI - (cost-1).
# PI is bounded [0,1].
# EI is unbounded (theoretically, but practically limited by f_best - min).
# Shekel min is -10.5. f_best starts around -5?
# So EI can be large.
#
# Let's try kappa=0.5, lambda=1.0.
#
# Also, consider that `cost` is provided.
#
# Let's add the deadline guard properly.
# The function is called multiple times.
# We can use a global variable for start time.

import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = 320.0  # Safety margin from 330s

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    if time.time() - _START_TIME > _DEADLINE:
        return np.zeros_like(mu)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement (EI)
    # E[max(0, f_best - f(x))]
    # = (f_best - mu) * P(f(x) < f_best) + sigma * pdf(z)
    # Note: Since we are minimizing, improvement is f_best - f(x).
    # f(x) ~ N(mu, sigma^2)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # E[f(x) | f(x) < f_best] = mu - sigma * pdf(z) / cdf(z)
    # E[improvement] = f_best * Phi(z) - (mu * Phi(z) - sigma * pdf(z))
    # = (f_best - mu) * Phi(z) + sigma * pdf(z)
    
    pdf_z = norm.pdf(z)
    cdf_z = norm.cdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # UCB Exploration Term
    # Encourages high uncertainty
    kappa = 0.1
    ucb = kappa * sigma_safe
    
    # Cost Penalty
    # Penalize higher cost
    # cost is in [1, 2]
    # Scale penalty to be comparable to EI
    # If EI is ~1, penalty ~1 is significant.
    lambda_cost = 0.5
    penalty = lambda_cost * (cost - 1.0)
    
    scores = ei + ucb - penalty
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
