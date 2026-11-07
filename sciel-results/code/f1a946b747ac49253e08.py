import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    p = np.asarray(params, dtype=float)
    
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)), mode='constant')
    else:
        p = p[:10]
        
    # Fixed centers for RBF
    centers = np.array([
        [-1.0, -1.0],
        [-1.0,  0.0],
        [-1.0,  1.0],
        [ 0.0, -1.0],
        [ 0.0,  0.0],
        [ 0.0,  1.0],
        [ 1.0, -1.0],
        [ 1.0,  0.0],
        [ 1.0,  1.0]
    ])
    
    # Calculate distances
    # x and y are arrays of shape (N,)
    # centers are (9, 2)
    # We need distance from each point (x[i], y[i]) to each center
    
    # Expand dimensions
    x_col = x[:, np.newaxis]
    y_col = y[:, np.newaxis]
    
    cx = centers[:, 0]
    cy = centers[:, 1]
    
    dx = x_col - cx
    dy = y_col - cy
    
    dist_sq = dx**2 + dy**2
    
    # RBF: exp(-dist_sq / (2 * sigma^2))
    sigma = 1.0
    rbf_values = np.exp(-dist_sq / (2 * sigma**2))
    
    # rbf_values shape: (N, 9)
    
    # Add bias column
    bias = np.ones((rfb_values.shape[0], 1))
    basis = np.hstack([bias, rbf_values])
    
    # p[0] is bias, p[1:10] are RBF weights
    pred = basis @ p
    
    return pred
