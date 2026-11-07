"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: The evaluator kills after 330s. We must return well before.
# The acquisition function is called repeatedly. We don't know the exact call count,
# but typically it's called until cost >= 30.
# With 5 initial points (cost ~5), and average cost per eval ~1.5 (since x in [0,1]^2, 
# mean(x) is ~0.5, so cost ~1.5), we have (30-5)/1.5 ~ 16 evaluations.
# Each call to acquisition scores 2000 points.
# The heavy lifting is in the GP evaluation in the harness, not here.
# Here, we just compute scores.
# However, if we do internal optimization, it might be slow.
# Let's stick to a fast, vectorized approach but with a structural change:
# Instead of simple EI/cost, we use a "Batch EI" approximation via greedy selection
# with a penalty for cost and a diversity bonus, but computed on the 2000 candidates.
# Wait, the function signature returns scores for 2000 candidates.
# The harness picks the argmax.
# If I want to do "batch" selection, I would need to return scores that encourage
# selecting a diverse set. But the harness only picks ONE (the best score).
# So "batch" in the return value doesn't directly work unless the harness supports it.
# The task says: "evaluates the best one". So it's a single-point selection loop.
# Therefore, I cannot return a batch. I must return a single-point acquisition score.

# So, what is a "structurally different" approach for single-point selection?
# 1. Expected Improvement (EI) - current.
# 2. Upper Confidence Bound (UCB) - failed in inspirations.
# 3. Probability of Improvement (PI) - less exploitative.
# 4. Knowledge Gradient (KG) - approximated by EI.
# 5. Cost-aware PI or EI with different cost modeling.
# 6. Thompson Sampling - requires sampling from GP, which is not passed (only mu, sigma).
#    We can approximate TS by sampling from N(mu, sigma^2) and picking the min?
#    But we only have mu and sigma for 2000 points. We can sample one value for each.
#    Score = -sampled_value / cost.
#    This is a valid structural change: from deterministic expected improvement to
#    stochastic sampling (Thompson Sampling approximation).

# Let's try Thompson Sampling approximation.
# For each candidate i:
#   sample y_i ~ N(mu_i, sigma_i^2)
#   score_i = -y_i / cost_i
# This balances exploration (high sigma -> high variance in y) and exploitation (low mu).
# It is fundamentally different from EI which integrates over the improvement.

# However, TS can be too exploratory if sigma is large.
# A hybrid: blend EI and TS? No, keep it pure for the structural change.

# Alternative: "Expected Improvement with Cost Penalty" but using a different cost model?
# The cost is 1 + mean(x_scaled).
# Current: EI / cost.
# What if we use: EI - lambda * cost? No, cost is per eval, EI is expected gain.
# Units don't match. EI/cost is gain per unit cost.

# Let's look at the "Levy 2d" function. It is multimodal.
# GP with Matern-5/2 and length scale 0.2 on [0,1]^2.
# The minimizer is moved to a uniform point in the inner 80%.
# This means the global minimum is somewhere in [0.1, 0.9]^2 roughly.

# Is there a way to use the structure of the 2000 candidates?
# Maybe a local search around the current best?
# But we don't have coordinates. We only have mu, sigma, cost.
# We can't do local search without coordinates.

# So, Thompson Sampling is the most distinct valid structural change.
# Let's implement Thompson Sampling approximation.
# To make it robust, we can use a single random seed per call? No, that would be deterministic.
# We should use random numbers. But the harness might call it many times.
# If we use random numbers, the scores will fluctuate.
# This is inherent to TS.

# One risk: If sigma is very small (converged), TS behaves like greedy.
# If sigma is large, TS explores.

# Let's add a time guard just in case.
_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    # Check time
    if time.time() - _start_time > 290:
        # Fallback to simple EI/cost if we are running out of time
        z = (f_best - mu) / (sigma + 1e-9)
        return ((f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)) / cost

    # Thompson Sampling Approximation
    # Sample from N(mu, sigma^2)
    # We need a random generator. To ensure reproducibility or just randomness?
    # The task doesn't specify determinism. Standard practice is to use global RNG or local.
    # Using global RNG means different calls yield different samples.
    
    # To avoid extreme outliers from the normal distribution affecting the "best" pick
    # in a bad way, we could clip the samples, but that distorts TS.
    # Let's stick to pure TS.
    
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Sample
    # Note: np.random.normal is fast for 2000 points.
    samples = np.random.normal(loc=mu, scale=sigma_safe)
    
    # Score: We want to minimize the function value.
    # So higher score = lower sampled value.
    # Cost awareness: divide by cost.
    # score = -samples / cost
    
    # However, if cost varies significantly, dividing by cost might be too aggressive.
    # The current baseline uses EI/cost.
    # Let's try -samples / cost.
    
    # Handle division by zero or very small cost
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = -samples / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
