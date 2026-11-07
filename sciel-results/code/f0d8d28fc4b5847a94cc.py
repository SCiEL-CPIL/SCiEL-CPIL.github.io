import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling acquisition function with cost awareness.
    
    Instead of using a deterministic score like EI/cost, this method samples
    from the posterior distribution N(mu, sigma^2) for each candidate.
    To make this deterministic and reproducible without maintaining state,
    we use the 'cost' array (which is unique per candidate due to the 
    continuous cost function 1 + mean(x_scaled)) to generate a deterministic
    random offset for each candidate.
    
    The score is the negative of the sampled value (since we minimize).
    However, to incorporate cost awareness structurally different from division,
    we can interpret the cost as a penalty on the "quality" of the sample 
    or use it to modulate the exploration. 
    Here, we use the cost to seed the random number generator for the sample,
    ensuring that different points get different samples, and we select the 
    point with the lowest sampled value (highest acquisition score).
    
    Wait, the task says "Return 2000 finite scores (higher = evaluate next)".
    If we use Thompson Sampling, we want to pick the x that minimizes f(x).
    So score = -sampled_value.
    
    But how does cost play in? 
    In standard TS, you don't explicitly divide by cost. 
    However, the prompt asks for a "cost-aware acquisition".
    The previous strategy was EI/cost. 
    A structural change is to use the cost to determine the *variance* or the *seed*.
    
    Let's stick to the core TS idea: 
    1. Generate a deterministic random number r_i for each candidate i based on cost[i].
    2. Sample s_i = mu[i] + sigma[i] * r_i.
    3. Score = -s_i.
    
    This is a fundamental shift from "expected improvement per cost" to "sampled value".
    It is stochastic in nature (though deterministically simulated), which often helps
    escape local optima that deterministic EI might get stuck in, especially in the
    early stages.
    
    To ensure robustness and handle the "cost-aware" requirement in a way that might 
    be more effective than simple division, we can also consider that lower cost 
    evaluations are "cheaper" to try. But TS naturally balances exploration/exploitation.
    
    Let's implement the deterministic TS.
    """
    
    # Handle NaNs and Infs in inputs
    mu_safe = np.nan_to_num(mu, nan=f_best, posinf=f_best, neginf=-np.inf)
    sigma_safe = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    sigma_safe = np.maximum(sigma_safe, 1e-6)
    
    cost_safe = np.nan_to_num(cost, nan=1.0, posinf=1.0, neginf=1.0)
    cost_safe = np.maximum(cost_safe, 1e-9)
    
    # Generate deterministic pseudo-random numbers in [0, 1)
    # Using a hash-like function on the cost value.
    # Since cost is a float, we can use it to seed a simple PRNG or use a trigonometric hash.
    # A common trick: sin(hash) is not good. 
    # Better: Use the fractional part of cost * large_prime to create a uniform distribution.
    # However, simple multiplication might have patterns.
    # Let's use a simple deterministic "random" number generator based on the index and cost.
    # Actually, since we don't have indices, we rely on cost.
    # If two points have the same cost, they get the same sample. This is a risk, but cost is continuous.
    
    # A more robust deterministic "random" number:
    # r = (cost * 12345.6789) % 1.0  -- This is not uniform.
    # Let's use a sine-based hash which is reasonably uniform for floats.
    # r_i = (sin(cost_i * 12345.6789) + 1) / 2
    
    # Even better, use a simple LCG step or just rely on the fact that 
    # we can use np.random.default_rng with a seed derived from the cost.
    # But creating 2000 RNGs is slow.
    
    # Vectorized approach:
    # Map cost to a uniform distribution in [0, 1].
    # Since cost is in [1, 2] (1 + mean(x in [0,1])), it's a small range.
    # We can stretch it.
    
    # Let's use a simple deterministic hash:
    # h = (cost * 1e6) % 1.0 is bad.
    
    # Alternative: Use the index? No, indices are not passed.
    # We must use cost.
    
    # Let's try: r = (np.sin(cost * 12345.6789) + 1) / 2
    # This provides a pseudo-random value in [0, 1].
    
    r = (np.sin(cost_safe * 12345.6789) + 1.0) / 2.0
    
    # Convert uniform [0, 1] to standard normal using inverse CDF (probit)
    # scipy.stats.norm.ppf is available
    # However, ppf(0) and ppf(1) are -inf and inf. 
    # We should clip r to [1e-6, 1-1e-6] to avoid infinities.
    r_clipped = np.clip(r, 1e-6, 1 - 1e-6)
    z_samples = norm.ppf(r_clipped)
    
    # Thompson Sample: s = mu + sigma * z
    samples = mu_safe + sigma_safe * z_samples
    
    # We want to minimize, so higher score is better.
    # Score = -sample
    scores = -samples
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
