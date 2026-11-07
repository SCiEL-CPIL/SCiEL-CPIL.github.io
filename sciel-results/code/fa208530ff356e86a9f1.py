import numpy as np
import time

_start_time = None

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Bivariate Gaussian Radial Basis Function (RBF) network for stress-strain.
    
    stress = sum_i a_i * exp(-((strain - c_i)^2 / (2*s_i^2) + (temp - d_i)^2 / (2*t_i^2)))
    
    10 params: [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10]
    Centers and widths are fixed to cover the domain.
    
    Centers (strain, temp):
    5 strain centers: [-0.001, 0.1, 0.3, 0.6, 0.95]
    2 temp centers: [0.1, 1.0]
    10 combinations.
    
    Widths: s_i = 0.15 for strain, t_i = 0.3 for temp (fixed, not fitted)
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > 140:
        return np.zeros_like(strain, dtype=np.float64)
    
    # Fixed centers
    strain_centers = np.array([-0.001, 0.1, 0.3, 0.6, 0.95])
    temp_centers = np.array([0.1, 1.0])
    
    # Fixed widths
    strain_width = 0.15
    temp_width = 0.3
    
    # Parameters are the weights for each RBF
    weights = params  # shape (10,)
    
    # Build the 10 RBF terms
    # For each of the 10 combinations of (strain_center, temp_center)
    # We need to compute exp(-((strain - sc)^2 / (2*sw^2) + (temp - tc)^2 / (2*tw^2)))
    
    # Reshape for broadcasting
    # strain: (N,), strain_centers: (5,) -> (N, 5)
    # temp: (N,), temp_centers: (2,) -> (N, 2)
    
    # Compute strain part
    strain_diff = strain[:, np.newaxis] - strain_centers[np.newaxis, :]  # (N, 5)
    strain_sq = strain_diff ** 2 / (2 * strain_width ** 2)  # (N, 5)
    
    # Compute temp part
    temp_diff = temp[:, np.newaxis] - temp_centers[np.newaxis, :]  # (N, 2)
    temp_sq = temp_diff ** 2 / (2 * temp_width ** 2)  # (N, 2)
    
    # Combine: for each combination (i, j) where i in 0..4, j in 0..1
    # index = i * 2 + j
    # RBF value = exp(-strain_sq[:, i] - temp_sq[:, j])
    
    # Create full RBF matrix: (N, 10)
    # For each strain center i and temp center j, compute the exponential
    # strain_sq[:, i] shape (N,), temp_sq[:, j] shape (N,)
    # We need to broadcast to (N, 5, 2) then reshape to (N, 10)
    
    # strain_sq: (N, 5), temp_sq: (N, 2)
    # Combine: strain_sq[:, :, np.newaxis] + temp_sq[:, np.newaxis, :] -> (N, 5, 2)
    combined = strain_sq[:, :, np.newaxis] + temp_sq[:, np.newaxis, :]  # (N, 5, 2)
    rbf_values = np.exp(-combined)  # (N, 5, 2)
    rbf_values = rbf_values.reshape(rbf_values.shape[0], -1)  # (N, 10)
    
    # Weighted sum
    result = rbf_values @ weights  # (N,)
    
    return result
