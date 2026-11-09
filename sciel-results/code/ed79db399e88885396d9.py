"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Strategy: Fixed Gaussian RBF basis with 10 centers.
    # Centers are chosen to cover the domain [0.5, 2.0]^3.
    # We use a 2x2x2 grid of centers.
    # This provides 8 local bases. We need 10 params.
    # The remaining 2 params can be used for a global linear/constant term 
    # or additional centers. Let's use 8 RBFs + 1 constant + 1 global linear term?
    # Or just 10 RBFs? 
    # The prompt suggests "10 fixed, evenly spaced centers".
    # A 2x2x2 grid has 8 points. 
    # Let's create a 2x2x2 grid and add 2 more centers, perhaps at the face centers or corners?
    # Actually, a simple 2x2x2 grid is robust. Let's stick to 8 RBFs and use 
    # the last 2 params for a low-order polynomial fallback (constant and linear x) 
    # to ensure global smoothness and handle cases where RBFs might struggle with 
    # large-scale trends, or just use 10 RBFs.
    
    # Let's try 10 RBFs.
    # Grid:
    # x: 0.8, 1.7 (2 points)
    # y: 0.8, 1.7 (2 points)
    # z: 0.8, 1.7 (2 points)
    # That's 8.
    # Add 2 more? Maybe at (1.25, 1.25, 0.8) and (1.25, 1.25, 1.7)?
    
    # Let's define the centers explicitly.
    # To be "evenly spaced", a 2x2x2 grid is the most symmetric for 8.
    # For 10, let's use the 8 corners of a sub-box and 2 centers of opposite faces.
    
    c_x = [0.8, 1.7]
    c_y = [0.8, 1.7]
    c_z = [0.8, 1.7]
    
    centers = []
    for cx in c_x:
        for cy in c_y:
            for cz in c_z:
                centers.append((cx, cy, cz))
                
    # Add 2 more centers to make 10.
    # Let's add the center of the domain in x,y but varying z?
    # Or just duplicate logic? No, distinct centers are better.
    # Let's add (1.25, 1.25, 0.8) and (1.25, 1.25, 1.7).
    centers.append((1.25, 1.25, 0.8))
    centers.append((1.25, 1.25, 1.7))
    
    # Precompute centers as array for vectorization
    # centers is a list of 10 tuples.
    C = np.array(centers) # shape (10, 3)
    
    # Input arrays
    # x, y, z are 1D arrays of length N (200 train/test)
    # We need to compute distance from each point (x_i, y_i, z_i) to each center c_j.
    
    # X_points shape: (N, 3)
    X = np.column_stack((x, y, z))
    
    # Distances squared
    # diff shape: (N, 10, 3)
    # We can broadcast: X[:, np.newaxis, :] - C[np.newaxis, :, :]
    diff = X[:, np.newaxis, :] - C[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2) # shape (N, 10)
    
    # Gaussian Kernel: exp(-gamma * dist_sq)
    # Choose gamma. Domain size is 1.5.
    # If gamma is too small, kernels overlap too much (ill-conditioned).
    # If gamma is too large, they are too local (discontinuous-ish).
    # Typical choice: gamma = 1 / (2 * sigma^2).
    # Let sigma ~ 0.4 (covers ~1/4 of the range).
    # gamma = 1 / (2 * 0.16) = 1 / 0.32 = 3.125.
    # Let's try gamma = 2.0 or 3.0.
    # Since inputs are in [0.5, 2], a distance of 0.5 is significant.
    # exp(-2 * 0.25) = exp(-0.5) ~ 0.6.
    # exp(-3 * 0.25) = exp(-0.75) ~ 0.47.
    # Let's use gamma = 2.5.
    
    gamma = 2.5
    K = np.exp(-gamma * dist_sq) # shape (N, 10)
    
    # Linear combination
    # params shape (10,)
    # result shape (N,)
    result = K @ params
    
    return result
# EVOLVE-BLOCK-END
