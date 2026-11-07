"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# We need to be fast. The acquisition function is called repeatedly.
# The inputs are mu, sigma, f_best, cost with shape (2000,).
# We return 2000 finite scores.

# The task is minimizing powell 4d on [-4, 5]^4.
# The cost is 1 + mean(x scaled to [0, 1]).
# x is in [-4, 5]. Scaled to [0, 1] means (x - (-4)) / (5 - (-4)) = (x+4)/9.
# So cost = 1 + mean((x+4)/9) over the 4 dimensions?
# "mean(x scaled to [0, 1])" likely means the average of the 4 scaled coordinates.
# Let's assume cost is given as an input array of shape (2000,).
# Wait, the prompt says "cost has shape (2000,)". So we don't need to compute it.
# We just use the provided cost.

# The current program uses Expected Improvement (EI) per unit cost.
# EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma.
# Score = EI / cost.

# To improve, we can try:
# 1. Expected Improvement (EI) is standard.
# 2. Probability of Improvement (PI): Phi(z).
# 3. Lower Confidence Bound (LCB): mu - beta * sigma.
#    LCB per cost: (mu - beta * sigma) / cost.
#    But we want to minimize, so lower LCB is better.
#    So we want to MAXIMIZE -LCB / cost? Or minimize LCB * cost?
#    The prompt says "higher = evaluate next".
#    So if we use LCB, we want high scores for points likely to be low.
#    Score = -(mu - beta * sigma) / cost = (beta * sigma - mu) / cost.
#    This is similar to EI but linear in sigma.

# Let's look at the "cost-aware" aspect.
# EI/cost is a common heuristic.
# Maybe we can use a more robust acquisition function.
# Or maybe tune the beta in LCB.

# Another idea: Use the UCB (Upper Confidence Bound) for minimization?
# No, LCB is for minimization.
# LCB(x) = mu(x) - beta * sigma(x).
# We want to find points with low LCB.
# So we want to maximize -LCB(x) / cost(x) ?
# Or maybe just -LCB(x) * cost(x)^(-gamma)?

# Let's stick with EI/cost but maybe add a small exploration term or tune it.
# Actually, EI can be zero or negative if mu > f_best.
# If mu > f_best, EI is negative (or close to 0 if we use max(0, EI)).
# Standard EI is max(0, EI).
# The current code doesn't use max(0, ...).
# If mu > f_best, z < 0.
# (f_best - mu) is negative.
# norm.cdf(z) is small.
# norm.pdf(z) is positive.
# The sum can be negative.
# Dividing by cost (positive) gives negative score.
# This effectively deprioritizes points where we expect the value to be worse than f_best.
# This is good.

# However, EI can be unstable when sigma is very small.
# If sigma -> 0, z -> +/- inf.
# If mu < f_best, z -> +inf. EI -> f_best - mu.
# If mu > f_best, z -> -inf. EI -> 0.
# This is well-behaved.

# What if we use a different acquisition function?
# Let's try LCB with an adaptive beta.
# Or maybe a combination.

# Let's consider the "powell 4d" function.
# It has a specific structure.
# Maybe the GP model is not perfect.
# Cost-aware EI is a strong baseline.

# Let's try to optimize the computation speed.
# norm.cdf and norm.pdf can be slow.
# We can use scipy.special.erf for cdf and a direct formula for pdf.
# norm.cdf(z) = 0.5 * (1 + erf(z / sqrt(2)))
# norm.pdf(z) = exp(-z^2 / 2) / sqrt(2 * pi)

# Let's rewrite the acquisition function with faster math.

def acquisition(mu, sigma, f_best, cost):
    # Handle edge cases
    sigma = np.maximum(sigma, 1e-9)
    
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Use faster approximations or direct scipy calls if fast enough.
    # scipy.special is usually faster than scipy.stats.
    from scipy.special import erf, exp, sqrt, pi
    
    # cdf = 0.5 * (1 + erf(z / sqrt(2)))
    cdf = 0.5 * (1.0 + erf(z / 1.4142135623730951))
    
    # pdf = exp(-z^2 / 2) / sqrt(2 * pi)
    pdf = exp(-0.5 * z * z) / 2.5066282746310002
    
    # EI = (f_best - mu) * cdf + sigma * pdf
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Score = EI / cost
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    scores = ei / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
