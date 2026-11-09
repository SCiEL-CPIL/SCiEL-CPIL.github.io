"""E. coli growth rate model using kernel regression with adaptive basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using adaptive kernel expansion.
    
    Model: mu = sum_k w_k * RBF(x; c_k, sigma_k) + bias
    
    We use 3 RBF centers with shared width, plus linear terms for extrapolation.
    The 10 parameters are:
    p0: weight for RBF center 1
    p1: weight for RBF center 2  
    p2: weight for RBF center 3
    p3: RBF width (sigma) - shared across centers
    p4: center 1 coordinate (b)
    p5: center 2 coordinate (b)
    p6: center 3 coordinate (b)
    p7: linear coefficient for b
    p8: linear coefficient for s (normalized)
    p9: bias
    """
    # Extract params
    w1, w2, w3 = params[0], params[1], params[2]
    sigma = np.abs(params[3]) + 0.1  # Ensure positive width
    c1, c2, c3 = params[4], params[5], params[6]
    lin_b = params[7]
    lin_s = params[8]
    bias = params[9]
    
    # Normalize inputs to reasonable range for stable computation
    # b: [1, 5] -> normalize to [0, 1]
    # s: [1, 40] -> normalize to [0, 1]  
    # temp: [1, 12] -> normalize to [0, 1]
    # pH: [1, 5] -> normalize to [0, 1]
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 39.0
    temp_norm = (temp - 1.0) / 11.0
    ph_norm = (pH - 1.0) / 4.0
    
    # Stack normalized features: [b_norm, s_norm, temp_norm, ph_norm]
    # Use all 4 dimensions for RBF distance
    features = np.stack([b_norm, s_norm, temp_norm, ph_norm], axis=-1)
    
    # RBF centers in 4D space (learned b-coordinates, others fixed at midpoints)
    # Centers: (c_k, 0.5, 0.5, 0.5) - we only learn b-coordinate of centers
    center1 = np.array([c1, 0.5, 0.5, 0.5])
    center2 = np.array([c2, 0.5, 0.5, 0.5])
    center3 = np.array([c3, 0.5, 0.5, 0.5])
    
    # Compute RBF terms: exp(-||x - c||^2 / (2 * sigma^2))
    def rbf(x, center, sigma):
        diff = x - center
        dist_sq = np.sum(diff ** 2, axis=-1)
        return np.exp(-dist_sq / (2.0 * sigma ** 2))
    
    rbf1 = rbf(features, center1, sigma)
    rbf2 = rbf(features, center2, sigma)
    rbf3 = rbf(features, center3, sigma)
    
    # Linear terms for better extrapolation
    lin_term = lin_b * b_norm + lin_s * s_norm
    
    # Combine
    result = w1 * rbf1 + w2 * rbf2 + w3 * rbf3 + lin_term + bias
    
    # Ensure finite output
    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return result
# EVOLVE-BLOCK-END
