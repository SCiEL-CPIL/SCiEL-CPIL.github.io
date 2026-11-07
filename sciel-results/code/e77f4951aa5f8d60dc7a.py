"""
Strategy: Fixed-Center Gaussian RBF with Linear Parameters.

The previous strategy used a polynomial basis. This strategy uses a Radial Basis Function (RBF) basis with FIXED centers and widths.
The 10 parameters are the amplitudes (weights) of the RBFs.
Because the model is linear in the parameters (y = sum(w_i * phi_i(x))), the MSE loss function is a convex quadratic function of the parameters.
This allows the external BFGS optimizer to reliably find the global minimum, unlike non-linear RBF models where centers/widths are optimized.

We select 10 centers in the 4D input space (b, s, temp, pH) that cover the range and capture potential interactions.
The inputs are normalized to [-1, 1] to ensure the Gaussian exponents are well-scaled and gradients are stable.

Centers chosen to cover the domain:
1. (0, 0, 0, 0)       : Center of the domain
2. (1, 0, 0, 0)       : High b
3. (-1, 0, 0, 0)      : Low b
4. (0, 1, 0, 0)       : High s
5. (0, -1, 0, 0)      : Low s
6. (0, 0, 1, 0)       : High temp
7. (0, 0, -1, 0)      : Low temp
8. (0, 0, 0, 1)       : High pH
9. (0, 0, 0, -1)      : Low pH
10. (0.5, 0.5, 0.5, 0.5) : Diagonal interaction point

Widths are fixed to provide local sensitivity.
"""
import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using Fixed-Center Gaussian RBFs.
    
    Args:
        b: nutrient concentration (approx [1, 5])
        s: substrate concentration (approx [1, 5])
        temp: temperature (approx [1, 40])
        pH: pH level (approx [2, 12])
        params: array of 10 amplitudes for the RBFs.
        
    Returns:
        Array of predicted growth rates.
    """
    # Normalize inputs to roughly [-1, 1]
    # b: [1, 5] -> center 3, half-range 2
    b_n = (b - 3.0) / 2.0
    # s: [1, 5] -> center 3, half-range 2
    s_n = (s - 3.0) / 2.0
    # temp: [1, 40] -> center 20.5, half-range 19.5
    t_n = (temp - 20.5) / 19.5
    # pH: [2, 12] -> center 7, half-range 5
    ph_n = (pH - 7.0) / 5.0
    
    # Stack normalized inputs into a matrix of shape (N, 4)
    # N is the number of samples
    X = np.column_stack((b_n, s_n, t_n, ph_n))
    
    # Define fixed centers for the 10 RBFs
    # Each center is a 4D point
    centers = np.array([
        [ 0.0,  0.0,  0.0,  0.0],  # 0: Center
        [ 1.0,  0.0,  0.0,  0.0],  # 1: High b
        [-1.0,  0.0,  0.0,  0.0],  # 2: Low b
        [ 0.0,  1.0,  0.0,  0.0],  # 3: High s
        [ 0.0, -1.0,  0.0,  0.0],  # 4: Low s
        [ 0.0,  0.0,  1.0,  0.0],  # 5: High temp
        [ 0.0,  0.0, -1.0,  0.0],  # 6: Low temp
        [ 0.0,  0.0,  0.0,  1.0],  # 7: High pH
        [ 0.0,  0.0,  0.0, -1.0],  # 8: Low pH
        [ 0.5,  0.5,  0.5,  0.5]   # 9: Mixed high
    ])
    
    # Fixed widths (sigma) for the Gaussians.
    # A width of ~0.5-1.0 in normalized space allows local fitting without being too noisy.
    # Using a uniform width for simplicity, or slightly varied if needed.
    # Let's use sigma = 0.8 for all, which corresponds to a decay length of ~0.8 in normalized units.
    sigma = 0.8
    sigma_sq = sigma * sigma
    
    # Calculate squared distances from each sample to each center
    # X shape: (N, 4)
    # centers shape: (10, 4)
    # We want (N, 10) distances.
    
    # Efficient calculation using broadcasting
    # X[:, np.newaxis, :] has shape (N, 1, 4)
    # centers[np.newaxis, :, :] has shape (1, 10, 4)
    # diff shape: (N, 10, 4)
    diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
    
    # Squared distances: (N, 10)
    sq_dist = np.sum(diff ** 2, axis=2)
    
    # Gaussian values: exp(-sq_dist / (2 * sigma^2))
    # To avoid underflow/overflow issues, we can compute directly.
    # Since inputs are normalized to [-1, 1], max sq_dist is around 16.
    # exp(-16 / (2*0.64)) = exp(-12.5) ~ 3e-6, which is fine.
    rbf_values = np.exp(-sq_dist / (2.0 * sigma_sq))
    
    # rbf_values shape: (N, 10)
    # params shape: (10,)
    
    # Prediction is the dot product of rbf_values and params
    # result shape: (N,)
    result = rbf_values @ params
    
    return result
