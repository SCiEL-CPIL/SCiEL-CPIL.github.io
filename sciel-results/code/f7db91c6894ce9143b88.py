import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts the target function using a Radial Basis Function (RBF) approximation.
    
    10 fixed centers are placed in the input domain [0.5, 2]^3.
    The 10 parameters scale the amplitude of each RBF.
    This allows the model to capture local non-polynomial structures.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    if x.size == 0:
        return np.array([], dtype=np.float64)
    
    if params.size < 10:
        params = np.concatenate([params, np.zeros(10 - params.size)])
    
    # Define 10 centers in [0.5, 2]^3
    # Using a subset of a 3x3x3 grid
    # Grid values: 0.5, 1.25, 2.0
    # We pick 10 points that are well-distributed.
    # 8 corners + center + one more?
    # Let's use:
    centers = np.array([
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 2.0],
        [0.5, 2.0, 0.5],
        [0.5, 2.0, 2.0],
        [2.0, 0.5, 0.5],
        [2.0, 0.5, 2.0],
        [2.0, 2.0, 0.5],
        [2.0, 2.0, 2.0],
        [1.25, 1.25, 1.25],
        [1.25, 0.5, 1.25]
    ])
    
    sigma = 0.5
    sigma_sq = sigma * sigma
    
    pred = np.zeros_like(x)
    
    for i in range(10):
        cx, cy, cz = centers[i]
        # Calculate squared distance
        dx = x - cx
        dy = y - cy
        dz = z - cz
        dist_sq = dx**2 + dy**2 + dz**2
        
        # Gaussian RBF
        rbf = np.exp(-dist_sq / (2.0 * sigma_sq))
        
        pred += params[i] * rbf
        
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
