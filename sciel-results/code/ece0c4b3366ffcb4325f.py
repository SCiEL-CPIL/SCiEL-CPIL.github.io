"""Rewritten program with a stronger acquisition function."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# We can cache some constants to speed up computation
# But the main improvement should be in the acquisition function itself.

# Let's try a hybrid approach:
# 1. Use Expected Improvement (EI) as the base
# 2. But also consider the probability of improvement
# 3. Or use Upper Confidence Bound (UCB) with a time-varying beta

# Actually, let me think about what might be better than EI:
# - UCB with proper exploration
# - Probability of Improvement (PI) - too greedy
# - Knowledge Gradient - too complex
# - Local Penalized EI - penalize points near already-evaluated points

# Since we don't have coordinates, we can't do local penalization directly.
# But we can use the sigma information to encourage exploration.

# Let's try a modified EI that accounts for the scale of sigma:
# The standard EI might be too conservative or too aggressive depending on the scale.

# Another idea: use a mixture of EI and UCB
# score = EI + beta * sigma

# Or use the probability of improvement weighted by the expected improvement

# Let me try: score = (f_best - mu) * Phi(z) + sigma * phi(z) + gamma * sigma * Phi(-z)
# where the last term encourages exploration when there's uncertainty and low probability of improvement

# Actually, let me reconsider. The standard EI is:
# EI = (f_best - mu) * Phi(z) + sigma * phi(z)  where z = (f_best - mu) / sigma

# This is a well-established acquisition function. Let me think about what could be better.

# One issue with EI is that it can be too focused on exploitation.
# UCB with beta = log(t) / t or similar might balance exploration better.

# But since we don't have the step number, we need to use sigma to guide exploration.

# Let me try: score = (f_best - mu) * Phi(z) + sigma * phi(z) + alpha * sigma * (1 - Phi(z))
# where alpha is a small constant to encourage exploration in uncertain regions

# Or simply: use EI but with a small epsilon to avoid numerical issues, and clip the scores

# Actually, let me try a different approach: use the expected maximum improvement
# or a variant that better balances exploration and exploitation.

# Let me try: score = max(0, f_best - mu - kappa * sigma)
# This is a variant of UCB that's more aggressive in exploration.

# Or: score = (f_best - mu) / (sigma + epsilon)  # This is like a z-score, higher when more likely to improve

# Let me think about this differently. The task is to minimize a function with minimum value -2.53218.
# f_best starts from 5 random points and improves over 25 more evaluations.

# A good acquisition function should:
# 1. Exploit regions where mu is low (likely to have low values)
# 2. Explore regions where sigma is high (uncertain)
# 3. Avoid regions where we're unlikely to improve (mu >> f_best)

# Let me try a combination:
# score = (f_best - mu) * Phi(z) + sigma * phi(z)  # EI
# But also add a small exploration bonus: + beta * sigma * Phi(-z)

# where Phi(-z) is the probability that the true value is above f_best (i.e., no improvement)
# This encourages exploring uncertain regions where there's a chance of finding a better value.

# Let me implement this with a small beta.

def acquisition(mu, sigma, f_best):
    """
    Acquisition function that combines Expected Improvement with an exploration bonus.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best (lowest) observed value so far
    
    Returns:
        array of shape (2000,) - acquisition scores (higher is better)
    """
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration bonus: encourage exploring regions with high uncertainty
    # where there's a reasonable chance of finding a better value
    # Phi(-z) is the probability that the true value is above f_best
    # We want to explore where this probability is moderate (not too high, not too low)
    # Use a weighted term that peaks at moderate uncertainty
    exploration = sigma_safe * norm.pdf(z) * 0.1  # Small bonus proportional to uncertainty
    
    # Combine EI and exploration bonus
    scores = ei + exploration
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
