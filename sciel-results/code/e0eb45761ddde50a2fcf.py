import numpy as np
from scipy.stats import norm
import time

# We use a deterministic, non-random acquisition that combines:
# 1. A "Thompson-like" score using the mean of the posterior (optimistic)
# 2. An exploration term based on uncertainty
# But crucially, we use a time-aware strategy:
# - Early on (few evals), we explore more aggressively (high uncertainty weight)
# - Later, we exploit more (higher mean weight)
# 
# Since we don't know the exact step, we can infer it from the number of
# distinct f_best values or just use a fixed schedule based on typical behavior.
# However, the simplest robust approach without step count is to use
# a hybrid that naturally balances: 
#   Score = -(mu - kappa * sigma)  [UCB for minimization]
# but with a dynamic kappa that decreases as f_best approaches 0.
#
# Actually, the Beale function has a minimum of 0. As f_best gets closer to 0,
# we should exploit more. Let's use a dynamic kappa based on f_best.
#
# Alternative: Pure UCB with a well-chosen kappa.
# The current program blends EI and UCB. Let's try pure UCB with adaptive kappa.
#
# Strategy: 
#   kappa = base_kappa * (1 + f_best)  # when f_best is large, explore more
#   But f_best can be negative? No, Beale min is 0, but values can be > 0.
#   Actually, f_best starts high and decreases.
#   Let's use: kappa = 1.0 + 0.5 * max(0, f_best - 1.0)
#   This gives kappa=1.0 when f_best <= 1, and increases as f_best is larger.
#
# Wait, the Beale function is:
# f(x,y) = (1.5 - 2x + x^2)(3.0 - 3y + y^2)^2 + 10
# The minimum is 0 at (3, 2). But the domain is [-4.5, 4.5]^2.
# At (3,2): (1.5-6+9)(3-6+4)^2 + 10 = (4.5)(1)^2 + 10 = 14.5? 
# Let me recalculate: 1.5 - 2*3 + 3^2 = 1.5 - 6 + 9 = 4.5
# 3 - 3*2 + 2^2 = 3 - 6 + 4 = 1
# So f(3,2) = 4.5 * 1 + 10 = 14.5? That's not 0.
#
# Actually, the standard Beale function is:
# f(x,y) = (1.5 - 2x + x^2)(3.0 - 3y + y^2)^2 + 10
# and its minimum is at (3, 2) with value 10? No, let me check.
#
# The Beale function is often defined as:
# f(x,y) = (1.5 - 2x + x^2)(3.0 - 3y + y^2)^2 + 10
# The global minimum is at (3, 2) with f(3,2) = (1.5-6+9)(3-6+4)^2 + 10 = 4.5*1+10=14.5
# But the problem says "minimum value 0". 
#
# Perhaps the function is defined without the +10, or shifted.
# If the minimum is 0, then the function might be:
# f(x,y) = (1.5 - 2x + x^2)(3.0 - 3y + y^2)^2
# which has minimum 0 at (3,2).
#
# Regardless, the acquisition function doesn't need to know the exact function.
# It just needs to balance exploration and exploitation.
#
# Let's try a pure UCB with a fixed kappa that works well for 2D problems.
# A common choice is kappa = 2.0 or 3.0.
#
# But the current program already uses UCB with kappa=2.0 blended with EI.
# Let's try a different approach: use the posterior mean and variance to
# compute a "probability of improvement" (POI) score, which is:
#   POI(x) = P(f(x) < f_best) = Phi((f_best - mu(x)) / sigma(x))
# This is a simple and effective acquisition function.
#
# Or, we can use a weighted combination of POI and UCB.
#
# Let's try: Score = POI + 0.5 * UCB_score
# where UCB_score = -mu + kappa*sigma
#
# Actually, let's try something simpler: just use UCB with a higher kappa
# to encourage more exploration early on.
#
# Strategy: Pure UCB with kappa = 3.0
# This is materially different from the current blend of EI and UCB.

def acquisition(mu, sigma, f_best):
    # Handle potential numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Pure UCB for minimization: -mu + kappa * sigma
    # Use a higher kappa to encourage exploration
    kappa = 3.0
    score = -mu + kappa * sigma_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
