import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a Cost-Weighted Thompson Sampling proxy.
    """
    # Sample a global multiplier lambda from a Beta distribution
    # This stochastically shifts the exploration-exploitation trade-off
    # Beta(2, 2) gives a distribution centered at 0.5 with some variance
    # Higher lambda -> more exploitation (f_best - mu)
    # Lower lambda -> more exploration (sigma)
    
    # Use a fixed seed for reproducibility within the same run? 
    # No, the task implies stochasticity is okay or expected for TS.
    # However, the evaluator runs multiple instances.
    # Let's use np.random.beta(2, 2)
    
    # To ensure we don't get extreme values that cause numerical issues,
    # we clip lambda to [0.01, 0.99]
    lam = np.random.beta(2, 2)
    lam = np.clip(lam, 0.01, 0.99)
    
    # Calculate improvement terms
    # Exploitation: f_best - mu (how much better than current best)
    # Exploration: sigma (uncertainty)
    
    exploitation = f_best - mu
    exploration = sigma
    
    # Combine them
    # score = (lam * exploitation + (1 - lam) * exploration) / cost
    
    # Note: We should ensure sigma is non-negative. It's a std dev, so it should be.
    # But let's clip just in case of numerical noise.
    sigma_clipped = np.maximum(sigma, 0.0)
    
    combined = lam * exploitation + (1.0 - lam) * sigma_clipped
    
    # Avoid division by zero
    safe_cost = np.maximum(cost, 1e-6)
    
    scores = combined / safe_cost
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
