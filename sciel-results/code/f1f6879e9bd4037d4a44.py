"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Thompson Sampling: sample one draw from the GP posterior.
    # The draw is mu - sigma * z where z ~ N(0,1) (minimization: lower is better).
    # We penalize by cost to prefer cheaper evaluations when function values are similar.
    # Use a fixed random state for reproducibility within a single call, but the
    # stochasticity across calls (different GP fits) provides the diversity.
    
    # Generate a single standard normal draw
    z = np.random.standard_normal(mu.shape[0])
    
    # Sampled function value: for minimization, we want low values
    # GP posterior: f ~ N(mu, sigma^2), so a sample is mu + sigma * z
    # But we want to MINIMIZE, so we score by -sampled_value (higher score = better)
    sampled = mu + sigma * z
    
    # Cost-aware: penalize high-cost candidates. The penalty should be scaled
    # relative to the function scale. Use f_best as a reference scale.
    # A candidate with sampled value v and cost c gets score: -v - lambda * c
    # where lambda controls the cost sensitivity.
    # We want the penalty to be meaningful relative to function values.
    # Scale lambda by |f_best| or a typical function value scale.
    
    # Use a cost penalty that is adaptive: if f_best is close to 0 (good),
    # be more sensitive to cost; otherwise, focus on function value.
    # Typical rosenbrock values in [-2,2]^2 range from 0 to ~65536.
    # f_best starts high and decreases.
    
    # Simple approach: penalize cost with a fixed weight relative to function scale.
    # The cost ranges from 1 to 2 (since mean(x scaled to [0,1]) is in [0,1]).
    # So cost penalty is at most 1 unit. If function values are large, this is negligible.
    # If function values are small (near optimum), this matters.
    
    # Use: score = -sampled / (cost + epsilon)
    # This makes expensive candidates less attractive proportionally.
    eps = 1e-6
    score = -sampled / (cost + eps)
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
