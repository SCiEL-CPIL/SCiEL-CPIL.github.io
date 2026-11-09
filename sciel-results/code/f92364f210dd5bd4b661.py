"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline guarding
_start_time = None
_DEADLINE = 330.0  # Hard limit is 330s, keep a safety margin
_MARGIN = 5.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Args:
        mu: Mean of the GP (shape 2000,)
        sigma: Std of the GP (shape 2000,)
        f_best: Best observed value (scalar)
        cost: Cost of evaluating each candidate (shape 2000,)
        
    Returns:
        scores: Acquisition scores (shape 2000,), higher is better
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    # Check deadline
    if time.time() > _DEADLINE - _MARGIN:
        # Fallback: return random scores or zero to ensure termination
        # Returning zeros means no preference, which might be safer than crashing
        return np.zeros_like(mu)

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Thompson Sampling:
    # We want to minimize f. 
    # The posterior is Normal(mu, sigma^2).
    # We sample a value y from N(mu, sigma^2).
    # The "reward" for evaluating this point is the negative of the function value (since we minimize).
    # However, we have a cost.
    # We want to maximize (Expected Improvement - Cost) or similar.
    # In TS, we sample the function value y.
    # The utility of evaluating x is roughly: -y - lambda * cost.
    # But we don't know y until we evaluate.
    # Standard TS for minimization: Sample y ~ N(mu, sigma^2). 
    # The score to maximize is usually just -y if cost is uniform.
    # With cost, we treat the "effective value" as y + lambda * cost.
    # We want to minimize this effective value.
    # So the score (to maximize) is -(y + lambda * cost) = -y - lambda * cost.
    
    # What is lambda?
    # The cost is 1 + mean(x_scaled). Range [1, 2].
    # The function values are standardized.
    # If we ignore cost, we might pick expensive points.
    # A reasonable heuristic: The cost variation is small (1 to 2).
    # The sigma is the uncertainty.
    # Let's use a fixed lambda. 
    # If lambda is too high, we exploit low-cost regions (low x).
    # If lambda is too low, we explore high-cost regions (high x) if they have high uncertainty.
    # Since the minimum is shifted uniformly, low x doesn't inherently guarantee a better minimum, 
    # but the cost is strictly higher for high x.
    # Therefore, we should prefer low x for the same GP posterior.
    
    # Let's sample y
    # To avoid generating 2000 random numbers every call (which is fast anyway), 
    # we can use the fact that N(mu, sigma) = mu + sigma * N(0, 1).
    
    # We need a single sample for all candidates to maintain the correlation structure 
    # implied by the GP? 
    # Actually, Thompson Sampling in BO typically samples a function from the posterior.
    # Here, we are given mu and sigma for 2000 points. 
    # If we sample independent z_i ~ N(0,1) for each point, we lose the spatial correlation 
    # between the sampled values. 
    # However, generating a full GP sample is not possible without the kernel matrix and observed data.
    # The prompt says "mu and sigma ... are in objective units". It does not provide the covariance matrix.
    # Thus, we must assume the provided mu/sigma are sufficient for a local approximation.
    # Independent sampling per point is a common approximation when only pointwise posterior params are available.
    # This is effectively "Independent Thompson Sampling".
    
    # Let's refine the cost penalty.
    # cost is in [1, 2].
    # If we set lambda = 0.5, the cost penalty is in [0.5, 1.0].
    # This is a significant shift relative to the function values (standardized, mean 0, std 1).
    # This will strongly bias towards low-cost (low x) candidates.
    # Is this good?
    # The minimizer is uniform in the inner 80%.
    # Low x is [0, 1]. Inner 80% is [1, 9].
    # So the minimizer is never in the lowest cost region [0, 1].
    # However, the function might be smooth.
    # If we bias too much to low x, we might miss the global minimum in [1, 9].
    # But we also have the GP mean mu. If the GP detects a good region in high x, mu will be low.
    # The cost penalty is a constant shift per x.
    
    # Let's try a moderate lambda.
    # If lambda is 0, we ignore cost.
    # If lambda is 1, we penalize heavily.
    
    # Let's use a dynamic lambda based on the current best cost? 
    # No, let's keep it simple.
    # Let's try lambda = 0.1. The cost difference between min and max cost is 1.0.
    # So the penalty range is [0.1, 0.2]. This is small compared to the function variance (1.0).
    # This allows the GP to dominate, but breaks ties in favor of low cost.
    
    # Actually, let's look at the "Expected Improvement" analogy.
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z).
    # This is non-negative.
    # TS score = -y - lambda * cost.
    # If we want to mimic EI-like behavior, we might want to center the samples.
    # But TS is generally robust.
    
    # Let's generate the sample.
    # Using a fixed seed for the random number generator inside the function would make it deterministic 
    # across calls with the same inputs, which is bad for exploration diversity if the GP doesn't change much.
    # However, the GP *does* change every iteration because we add a new point.
    # So mu and sigma change.
    # Using a global RNG is fine.
    
    # To ensure we don't have issues with reproducibility or specific RNG state, 
    # we use np.random.randn.
    
    z = np.random.randn(mu.shape)
    y = mu + sigma * z
    
    # Cost penalty
    lambda_cost = 0.1
    scores = -(y + lambda_cost * cost)
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
