import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Thompson Sampling with cost-awareness
    # Sample latent values from the GP predictive distribution
    # and select based on ratio of sampled value to cost
    
    start_time = time.time()
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-10)
    cost = np.maximum(cost, 1e-10)
    
    # Sample from the GP predictive distribution: f ~ N(mu, sigma^2)
    # Use a fixed seed per call for reproducibility within a single acquisition call
    # But we want stochastic exploration, so use random samples
    n_samples = len(mu)
    
    # Generate standard normal random numbers
    rng = np.random.default_rng()
    z = rng.standard_normal(n_samples)
    
    # Sampled function values
    f_samples = mu + sigma * z
    
    # For minimization, we want to find low values
    # Thompson sampling for minimization: pick the candidate with the lowest sampled value
    # But we need cost-awareness
    
    # Cost-aware Thompson sampling:
    # We want to balance low function values with low cost
    # Use a ratio or weighted combination
    
    # Approach 1: Minimize (f_sample + lambda * cost)
    # Approach 2: Use f_sample / cost (lower is better for minimization with cost)
    # Approach 3: Use a probabilistic criterion
    
    # For minimization with cost, a natural choice is to minimize expected cost-adjusted value
    # Let's use: score = -f_sample / cost (higher is better for selection)
    # Or equivalently, we want to maximize -f_sample / cost
    
    # But we need to be careful: f_best is the current best (lowest observed)
    # We should focus on points that might beat f_best
    
    # Better approach: Use the expected improvement formulation but with Thompson sampling
    # EI for minimization: E[max(f_best - f, 0)]
    # With Thompson sampling, we can compute: max(f_best - f_sample, 0) / cost
    
    # Let's use: score = max(f_best - f_sample, 0) / cost
    # This gives high scores to points that likely improve upon f_best at low cost
    
    improvement = np.maximum(f_best - f_samples, 0.0)
    score = improvement / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Safety margin check
    elapsed = time.time() - start_time
    if elapsed > 290:  # Well under the 300s limit
        # Return a safe fallback: just use EI / sqrt(cost)
        sigma_safe = np.maximum(sigma, 1e-8)
        z_ei = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
        return ei / np.sqrt(cost)
    
    return score
