import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Strategy: RBF Network with fixed centers and variable widths/heights
    # We use 5 RBFs. 
    # params[0:5] are the weights (heights) for the 5 RBFs
    # params[5:10] are the inverse squared widths (1/(2*sigma^2)) for the 5 RBFs
    # We fix the centers to a grid covering the input domain [0.5, 2]^3.
    
    # Fixed centers: A 2x2x2 grid would need 8 centers, but we only have 10 params.
    # Let's use 5 centers. A reasonable set:
    # 1. Center of the domain: (1.25, 1.25, 1.25)
    # 2. Corners or edges to capture boundary effects?
    # Let's pick 5 distinct points in [0.5, 2]^3.
    # c1 = (1.25, 1.25, 1.25)
    # c2 = (0.5, 0.5, 0.5)
    # c3 = (2.0, 2.0, 2.0)
    # c4 = (0.5, 2.0, 1.25)
    # c5 = (2.0, 0.5, 1.25)
    
    centers = np.array([
        [1.25, 1.25, 1.25],
        [0.5, 0.5, 0.5],
        [2.0, 2.0, 2.0],
        [0.5, 2.0, 1.25],
        [2.0, 0.5, 1.25]
    ])
    
    # Number of points
    n = x.shape[0]
    
    # Initialize prediction array
    pred = np.zeros(n)
    
    # Parameters:
    # p[0:5] -> weights (heights)
    # p[5:10] -> inverse squared widths (lambda). RBF = exp(-lambda * dist^2)
    # We need to ensure lambda is positive to avoid explosion. 
    # BFGS might try negative values. 
    # If lambda < 0, exp(-lambda * d^2) = exp(|lambda| * d^2) which grows exponentially.
    # This could cause overflow. 
    # To be safe, we can use lambda = p[5:10] ** 2 or similar, but that changes the optimization landscape.
    # Alternatively, we can clip lambda to a small positive value.
    # Let's use lambda = p[5:10] * p[5:10] to ensure positivity and smoothness.
    # Wait, if p[5:10] is large, lambda is huge, RBF becomes very narrow (delta function).
    # If p[5:10] is small, lambda is small, RBF becomes very broad.
    # This is a reasonable parameterization.
    
    weights = params[0:5]
    inv_widths_sq = params[5:10] * params[5:10]  # Ensure positive
    
    # Compute distances from each point to each center
    # x, y, z are arrays of length n.
    # centers are (5, 3)
    
    # We can loop over the 5 centers for efficiency and clarity
    for i in range(5):
        cx, cy, cz = centers[i]
        # Calculate squared Euclidean distance
        dx = x - cx
        dy = y - cy
        dz = z - cz
        dist_sq = dx*dx + dy*dy + dz*dz
        
        # Calculate RBF value
        # If inv_widths_sq is 0, exp(0) = 1.
        # If inv_widths_sq is large, exp(-large * dist_sq) -> 0 for dist_sq > 0.
        rbf_val = np.exp(-inv_widths_sq[i] * dist_sq)
        
        pred += weights[i] * rbf_val
        
    return pred
