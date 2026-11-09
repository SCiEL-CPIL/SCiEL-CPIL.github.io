"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Uses a hybrid approach:
    1. Compute Expected Improvement (EI) scores.
    2. Add a diversity/exploration bonus based on local density of high-EI points.
    3. Ensure numerical stability and finite outputs.
    """
    start_time = time.time()
    deadline = 325.0  # 5 seconds safety margin before 330s limit
    
    n = len(mu)
    
    # Clip sigma to avoid division by zero or extremely small values
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Compute EI: (f_best - mu) * Phi(z) + sigma * phi(z)
    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative (can be slightly negative due to numerical issues)
    ei = np.maximum(ei, 0.0)
    
    # Add a small exploration bonus based on uncertainty
    # This encourages exploring regions with high sigma
    exploration_bonus = sigma_safe * 0.1
    
    # Combine EI and exploration
    scores = ei + exploration_bonus
    
    # Check deadline
    if time.time() > deadline:
        # If we're running out of time, just return the basic scores
        # Ensure finite values
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Add a diversity bonus: penalize points that are too close to other high-score points
    # Use a simple local density estimate
    # For efficiency, only consider top-K points for diversity calculation
    k = min(200, n)
    
    # Get indices of top-k EI scores
    top_k_indices = np.argpartition(ei, -k)[-k:]
    
    # Compute pairwise distances among top-k points
    # Since we don't have coordinates, we use the index as a proxy for position
    # This is a heuristic: points with similar mu/sigma patterns are "close"
    
    # Alternative: use the normalized mu and sigma to create a feature space
    mu_norm = (mu - np.min(mu)) / (np.max(mu) - np.min(mu) + 1e-10)
    sigma_norm = (sigma - np.min(sigma)) / (np.max(sigma) - np.min(sigma) + 1e-10)
    
    # Create a 2D feature vector for each point
    features = np.column_stack([mu_norm, sigma_norm])
    
    # Compute distances from each point to the top-k points
    top_k_features = features[top_k_indices]
    
    # For each point, compute the minimum distance to any top-k point
    # This is O(n*k) which is manageable for n=2000, k=200
    min_distances = np.full(n, np.inf)
    
    # Process in chunks to manage memory
    chunk_size = 500
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        chunk_features = features[i:end]
        
        # Compute distances from chunk to top-k points
        # Shape: (chunk_size, k)
        dists = np.linalg.norm(chunk_features[:, np.newaxis, :] - top_k_features[np.newaxis, :, :], axis=2)
        
        # Minimum distance to any top-k point
        min_dists = np.min(dists, axis=1)
        min_distances[i:end] = min_dists
    
    # Check deadline
    if time.time() > deadline:
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Diversity bonus: points far from other high-EI points get a bonus
    # Normalize distances to [0, 1]
    max_dist = np.max(min_distances) + 1e-10
    diversity_bonus = (min_distances / max_dist) * 0.5
    
    # Final score
    scores = scores + diversity_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure scores are non-negative
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
