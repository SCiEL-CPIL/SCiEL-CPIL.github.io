import numpy as np
import time

# Module-level state for deadline guard and persistent bandit state
_start_time = None
# We maintain a set of "arms" (candidate points) and their statistics.
# Since we don't receive x coordinates, we must track the *indices* of the candidates
# from the 2000 uniform grid provided by the evaluator.
# However, the evaluator provides 2000 *new* uniform candidates every time.
# This means we cannot maintain a fixed set of arms across calls in the traditional MAB sense
# unless we map the continuous space to a discrete grid or use the indices directly.
#
# Wait, the task says: "the evaluator scores 2000 new uniform candidates with acquisition".
# This implies the candidate set changes every step. We cannot maintain a fixed arm history
# based on indices because index 5 in step 1 is a different point than index 5 in step 2.
#
# Therefore, true persistent MAB on a fixed grid is not possible with this interface.
# We must adapt the strategy.
#
# Alternative Structural Change: **Bayesian Optimization with Explicit Cost-Adjusted Lower Confidence Bound (LCB) 
# using a Non-Linear Cost Penalty and Adaptive Temperature.**
#
# The current program uses a linear cost division: score = (-mu + beta*sigma) / cost.
# A different design family is **Probability of Improvement (PoI) adjusted for cost** 
# or **Expected Improvement (EI) adjusted for cost**, but the prompt says "different design...".
#
# Let's look at Inspiration 2: Cost-Adjusted Thompson Sampling.
# Since we can't persist arms, we can simulate Thompson Sampling by drawing a sample from the posterior 
# *inside* the acquisition function for each of the 2000 points? No, that's just UCB with a random beta.
#
# Let's try a **Cost-Sensitive Probability of Improvement (PoI)**.
# PoI(x) = P(f(x) < f_best) = Phi((f_best - mu(x)) / sigma(x)).
# This is a probability in [0, 1]. It naturally handles the scale of improvement.
# To make it cost-aware, we can use a **Cost-Adjusted PoI**:
# score = PoI(x) / (cost(x) ^ alpha)
# Or better, a **Cost-Adjusted Expected Improvement**:
# EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma.
# Cost-Aware EI: score = EI(x) / (cost(x) + epsilon).
#
# The current program is UCB. EI is a different acquisition function family.
# EI is generally more robust than UCB because it accounts for the magnitude of the improvement, 
# not just the bound. UCB can be overly optimistic in high-variance regions where the expected 
# improvement is actually small. EI balances the probability of improvement with the expected gain.
#
# Additionally, we can introduce a **dynamic temperature** or **exploration rate** that decays 
# more aggressively based on the *cost* spent so far, not just the gap.
#
# Let's implement **Cost-Aware Expected Improvement (CA-EI)** with a **non-linear cost penalty**.
# Instead of dividing by cost (which can be unstable if cost is small), we can use 
# score = EI(x) * (1 / (1 + cost(x))). This is smoother.
#
# To make it structurally different from the current UCB, we switch to EI.
# To make it better, we add a **look-ahead** or **multi-step** consideration? 
# No, keep it simple but different.
#
# Another option: **Knowledge Gradient (KG)**.
# KG estimates the expected value of information from evaluating a point.
# It is computationally expensive to compute exactly, but we can approximate it.
# However, with 2000 points, we need speed.
#
# Let's stick to **Cost-Aware EI** but with a **adaptive beta-like scaling on the EI formula itself**?
# No, EI doesn't have a beta.
#
# Let's try **Cost-Aware Probability of Improvement (PoI)**.
# PoI is simpler and often works well. 
# score = PoI(x) / cost(x).
#
# But the biggest structural change might be **using the cost in the GP model itself**?
# No, we only get mu, sigma, f_best, cost. We can't change the GP.
#
# Let's look at the "Inspiration 2" again: Cost-Adjusted Thompson Sampling.
# How to do TS without persistent arms?
# We can draw a single sample from the posterior for the *entire function*? 
# No, we only have mu and sigma at 2000 points.
# We can draw a sample `y_i = mu_i + sigma_i * z_i` where `z_i` are correlated? 
# We don't have the covariance matrix, only the marginals. 
# So we can't sample the full function. We can only sample independent values at each point.
# That is effectively UCB with a random beta: `score = (-mu + beta_i * sigma) / cost` where `beta_i` is random.
# This is "Thompson Sampling" on the marginal distributions.
#
# Let's implement **Marginal Thompson Sampling with Cost Adjustment**.
# 1. Generate random multipliers `z` for each of the 2000 points (standard normal).
# 2. Compute sampled objective: `f_sample = mu + sigma * z`.
# 3. For minimization, we want low `f_sample`.
# 4. Score: `-f_sample / cost`.
#
# Why is this different from UCB?
# UCB: `-mu + beta * sigma`. This is a deterministic bound.
# TS: `-mu + sigma * z`. This is a random sample.
# TS allows for "exploration" by sometimes sampling low values in high-variance regions, 
# but it also allows for "exploitation" by sampling near the mean in low-variance regions.
# The randomness is injected every step.
#
# To make it cost-aware, we divide by cost.
#
# One improvement: Use a **deterministic seed** for the random numbers based on the step count 
# or a global counter to ensure reproducibility? 
# The prompt says "five seeded instances". The evaluator runs the function 5 times for 5 instances.
# If we use `np.random.seed()`, it will be different for each call unless we control it.
# If we use a global counter, the sequence of `z` will be deterministic across calls within a run.
# This is good for stability.
#
# Let's implement **Cost-Aware Marginal Thompson Sampling**.

_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    global _start_time, _call_count
    
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 320:
        # Fallback: simple exploitation
        mu_safe = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
        cost_safe = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=-1e6)
        cost_safe = np.maximum(cost_safe, 1e-6)
        return -mu_safe / cost_safe

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Ensure finite values
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=-1e6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=-1e6)
    
    # Avoid division by zero
    cost = np.maximum(cost, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    
    # Increment call count for deterministic random seed
    _call_count += 1
    
    # Use a deterministic random number generator seeded by the call count
    # This ensures that the "Thompson Sampling" noise is consistent and reproducible
    # within a single evaluation run, which helps the optimizer converge.
    rng = np.random.default_rng(_call_count)
    
    # Draw standard normal samples for each candidate point
    # z has shape (2000,)
    z = rng.standard_normal(mu.shape)
    
    # Compute the sampled objective value
    # f_sample = mu + sigma * z
    # For minimization, we want to find the minimum of f_sample
    # So the "score" should be high for low f_sample
    f_sample = mu + sigma * z
    
    # Cost-aware score: We want to maximize the "value" per cost.
    # Value of a point is its negative function value (since we minimize).
    # score = -f_sample / cost
    score = -f_sample / cost
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    return score
