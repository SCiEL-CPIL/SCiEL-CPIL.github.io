import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using a Radial Basis Function (RBF) model with 
    biologically informed centers.
    
    The inputs are normalized to [0, 1].
    10 Gaussian basis functions are defined with fixed centers chosen to cover 
    the likely optimal growth region (T~37C, pH~7) and boundaries.
    The 10 parameters are the linear weights for these basis functions.
    This is a convex problem in the weights, ensuring BFGS finds the global 
    minimum for the given centers.
    """
    # Normalize inputs to [0, 1] based on known ranges
    # b: [1, 5] -> [0, 1]
    # s: [1, 5] -> [0, 1]
    # temp: [1, 40] -> [0, 1]
    # pH: [2, 12] -> [0, 1]
    
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    temp_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0
    
    # Define 10 fixed centers in normalized space
    # Focus on Optimal T (~37C -> 0.92) and Optimal pH (~7 -> 0.5)
    centers = np.array([
        [1.0, 1.0, 0.92, 0.5],  # High b, High s, Opt T, Opt pH
        [0.5, 0.5, 0.92, 0.5],  # Mid b, Mid s, Opt T, Opt pH
        [0.0, 0.0, 0.92, 0.5],  # Low b, Low s, Opt T, Opt pH
        [1.0, 1.0, 0.50, 0.5],  # High b, High s, Mid T, Opt pH
        [0.5, 0.5, 0.50, 0.5],  # Mid b, Mid s, Mid T, Opt pH
        [1.0, 1.0, 0.92, 0.0],  # High b, High s, Opt T, Low pH
        [1.0, 1.0, 0.92, 1.0],  # High b, High s, Opt T, High pH
        [0.5, 0.5, 0.92, 0.0],  # Mid b, Mid s, Opt T, Low pH
        [0.5, 0.5, 0.92, 1.0],  # Mid b, Mid s, Opt T, High pH
        [1.0, 1.0, 1.00, 0.5]   # High b, High s, Max T, Opt pH
    ])
    
    # Gaussian width (sigma) in normalized space
    sigma = 0.25
    sigma_sq = 2.0 * sigma * sigma
    
    # Compute distances from each input point to each center
    # Inputs are (N, 4), Centers are (10, 4)
    # We want (N, 10) distances squared
    
    # Expand dimensions for broadcasting: (N, 1, 4) and (1, 10, 4)
    x = np.stack([b_norm, s_norm, temp_norm, ph_norm], axis=-1)
    c = centers[None, :, :]
    
    diff = x[:, None, :] - c
    dist_sq = np.sum(diff ** 2, axis=-1) # Shape (N, 10)
    
    # Compute Gaussian basis values
    # phi = exp(-dist_sq / sigma_sq)
    phi = np.exp(-dist_sq / sigma_sq)
    
    # Weighted sum
    # params is (10,)
    y = np.sum(phi * params[None, :], axis=1)
    
    return y
