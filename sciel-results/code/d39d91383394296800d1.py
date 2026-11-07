"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Constants for the acquisition function parameters
# We will optimize these parameters against the task's objective.
# However, since we don't have access to the training data during runtime,
# we rely on a robust default strategy that balances exploration and exploitation.
# Common choices:
# - Expected Improvement (EI): z = (f_best - mu) / sigma, score = (f_best - mu) * Phi(z) + sigma * phi(z)
# - Upper Confidence Bound (UCB): score = -mu + kappa * sigma (for minimization, we want to minimize mu but maximize uncertainty)
#   Actually, for minimization, UCB is typically -mu + kappa * sigma.
#   Let's stick to EI as it's generally strong, but maybe add a small exploration bonus or adjust the temperature.
# 
# The directive says "MODEL-AND-OPTIMIZE approach: parameterize the answer and fit or optimize the parameters directly against the task's objective."
# Since we cannot fit parameters online without significant overhead and we have a strict time limit, we will implement a parameterized acquisition
# that uses a small set of hyperparameters. We can choose good defaults based on literature for Beale function.
# Beale function is known to have multiple local minima. EI tends to be exploitative. 
# Perhaps a combination of EI and a small UCB term? Or just standard EI?
# The current program uses standard EI and scores 1.0. 
# To improve, we might try:
# 1. Slight modification to EI, e.g., adding a small constant to sigma to avoid division by zero issues or to encourage exploration more.
# 2. Using a different acquisition function like Probability of Improvement (PI) is less efficient.
# 3. Using a hybrid: EI + lambda * sigma. This encourages exploring points with high uncertainty even if they don't have high expected improvement.
#
# Let's implement a hybrid EI/UCB: score = EI + alpha * sigma
# We need to choose alpha. A common choice is small, e.g., 0.1 or 0.01.
# Alternatively, we can use the fact that the evaluator uses 30 evaluations total.
# In the early stages, exploration is key. In later stages, exploitation.
# But we don't know the step number.
#
# Let's try a simple optimization: we will compute EI as before, but we will also compute a "confidence" term.
# Actually, the most robust improvement over pure EI for multi-modal functions like Beale is often to add a small amount of random exploration or use a more aggressive exploration strategy early on.
# However, since we can't change the loop logic, we must make the acquisition function itself smarter.
#
# Let's try: score = EI + beta * sigma
# where beta is a small positive constant. This makes the acquisition function less prone to collapsing to a single region.
# Let's set beta = 0.1 * (f_best - np.min(mu)) / (np.std(mu) + 1e-9) to normalize the exploration term?
# Or just a fixed beta. Let's try beta = 0.05.
#
# Also, we should ensure numerical stability.
#
# Another idea: Use the fact that the function is bounded. 
# Let's stick to a robust EI with a small exploration bonus.

def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Dynamic exploration weight based on median uncertainty
    # Higher uncertainty (early stages) leads to higher exploration weight
    med_sigma = np.median(sigma_safe)
    beta = 0.05 + 0.1 * med_sigma
    
    exploration_bonus = beta * sigma_safe
    
    # Small penalty for points predicted to be much worse than current best
    # This helps avoid wasting evaluations on clearly bad regions
    penalty = 0.01 * np.maximum(0, mu - f_best)
    
    scores = ei + exploration_bonus - penalty
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
