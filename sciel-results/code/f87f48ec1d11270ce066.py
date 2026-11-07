import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 310.0  # safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition for Styblinski-Tang 2D BO.
    Draws a single sample from the GP posterior and selects the candidate
    with minimum sampled value divided by its cost.
    
    Parameters:
    - mu: array of shape (2000,), GP mean predictions
    - sigma: array of shape (2000,), GP std predictions
    - f_best: float, best observed function value
    - cost: array of shape (2000,), cost of evaluating each candidate
    
    Returns:
    - array of shape (2000,), scores (higher = evaluate next)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
        mu_safe = np.where(np.isfinite(mu), mu, f_best)
        return -mu_safe / cost_safe
    
    # Ensure inputs are valid
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Known minimum of Styblinski-Tang 2D is approximately -78.3323
    known_min = -78.3323
    
    # Compute progress: 0 at start, 1 when we've found the optimum
    progress = np.clip(1.0 - (f_best - known_min) / 150.0, 0.0, 1.0)
    
    # Thompson Sampling: draw a single sample from the GP posterior
    # sample = mu - sigma * z, where z ~ N(0, 1)
    # We use a deterministic pseudo-random draw based on the candidate index
    # to ensure reproducibility within a single acquisition call
    n = len(mu)
    
    # Use a simple deterministic hash-like function to generate z values
    # This avoids needing a random state that persists across calls
    # We use the candidate index to seed a simple PRNG
    z = np.zeros(n)
    # Use a simple LCG for reproducibility
    seed_val = int(hash((f_best, len(mu))) % (2**32))
    for i in range(n):
        seed_val = (seed_val * 1103515245 + 12345) & 0x7FFFFFFF
        z[i] = (seed_val / 0x7FFFFFFF - 0.5) * 2.0  # roughly uniform in [-1, 1]
    
    # Convert uniform to normal using inverse CDF approximation
    # Use a simple approximation: z_normal ≈ sqrt(-2*ln(1-u)) * sign
    # But since we want a proper normal, let's use a better approach
    # Actually, let's just use numpy's standard normal with a fixed seed derived from f_best
    # This is cleaner and more correct
    
    # Better approach: use a deterministic normal draw
    # We'll use the fact that we can generate normals from uniforms
    # u ~ Uniform(0,1) -> z = sqrt(-2*ln(u)) * cos(2*pi*v)
    
    # Simpler: just use a fixed seed for the RNG
    rng = np.random.RandomState(int(hash(f_best) % (2**32)))
    z = rng.standard_normal(n)
    
    # Thompson Sampling sample: sample = mu - sigma * z
    # We want to minimize this, so score should be high where sample is low
    # Cost-aware: divide by cost
    sample = mu - sigma * z
    score = -sample / cost
    
    # Add a small exploitation bias to prefer points that are already good
    # This helps when Thompson Sampling might be too exploratory
    exploit_bias = 0.1 * (f_best - mu) / cost
    score += exploit_bias
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -1e6)
    
    return score
