import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    params = np.asarray(params, dtype=float)
    
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)))
    elif len(params) > 10:
        params = params[:10]
    
    # Softplus basis: sum_{i=1}^{10} w_i * softplus(s_i * (x - c_i))
    # Centers c_i and slopes s_i are fixed on a symmetric grid in [-2, 2]
    # Only weights w_i (params[0:10]) are free parameters
    # This is linear-in-weights, so BFGS finds global minimum
    
    x_1d = x.flatten()
    
    # Fixed centers: symmetric grid in [-2, 2]
    # 10 centers spread evenly
    centers = np.linspace(-2.0, 2.0, 10)
    
    # Fixed slopes: vary to capture different scales of variation
    # Mix of small and large slopes for multi-scale approximation
    slopes = np.array([1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 0.5, -0.5, 4.0, -4.0])
    
    # Compute softplus basis functions
    # softplus(z) = log(1 + exp(z)), numerically stable version
    z = slopes * (x_1d[:, np.newaxis] - centers[np.newaxis, :])  # shape: (N, 10)
    
    # Numerically stable softplus: log(1 + exp(z)) = max(0, z) + log(1 + exp(-|z|))
    softplus_z = np.maximum(0, z) + np.log1p(np.exp(-np.abs(z)))
    
    # Linear combination with weights
    weights = params[:10]
    result = softplus_z @ weights  # shape: (N,)
    
    # Reshape to match input shape
    result = result.reshape(x.shape)
    
    return result
