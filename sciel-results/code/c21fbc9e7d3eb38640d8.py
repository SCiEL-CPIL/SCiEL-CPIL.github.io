import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Gaussian Radial Basis Function (RBF) network.
    
    We define 10 fixed centers on a 3x3 grid plus the origin:
    Centers:
    (-1, -1), (-1, 0), (-1, 1)
    ( 0, -1), ( 0, 0), ( 0, 1)
    ( 1, -1), ( 1, 0), ( 1, 1)
    ( 0, 0)  -- duplicate? No, let's pick a different 10th center.
    
    Actually, a 3x3 grid has 9 points. We need 10.
    Let's add the origin (0,0) if it's not already there?
    The 3x3 grid with coordinates in {-1, 0, 1} includes (0,0).
    So the 9 points are:
    (-1,-1), (-1,0), (-1,1)
    ( 0,-1), ( 0,0), ( 0,1)
    ( 1,-1), ( 1,0), ( 1,1)
    
    We need one more center. Let's add (0.5, 0.5) or something?
    Or maybe a 4x3 grid? No, that's 12.
    Let's stick to 9 grid points + 1 origin? But origin is already there.
    Let's use 9 grid points + 1 center at (0, 0) is redundant.
    
    Alternative: Use 10 centers from a regular grid or random fixed positions.
    Let's use:
    (-1, -1), (-1, 1)
    ( 1, -1), ( 1, 1)
    (-0.5, -0.5), (-0.5, 0.5)
    ( 0.5, -0.5), ( 0.5, 0.5)
    ( 0, 0)
    ( 0, 0.5)  <- Arbitrary 10th?
    
    Better: Just use the 9 grid points and add one more at (0,0) is bad.
    Let's use a 4x3 grid?
    x in {-1, -1/3, 1/3, 1}, y in {-1, 0, 1}. That's 12.
    
    Let's just use 10 fixed centers that cover the domain [-1, 1]x[-1, 1].
    Centers:
    1. (-1.0, -1.0)
    2. (-1.0,  1.0)
    3. ( 1.0, -1.0)
    4. ( 1.0,  1.0)
    5. (-0.5, -0.5)
    6. (-0.5,  0.5)
    7. ( 0.5, -0.5)
    8. ( 0.5,  0.5)
    9. ( 0.0,  0.0)
    10. ( 0.0,  1.0)  # Just to have 10 distinct points
    
    Actually, symmetry might be better.
    Let's use:
    9 points from 3x3 grid: {-1, 0, 1} x {-1, 0, 1}
    And the 10th point at (0, 0) is already included.
    So I will use the 9 grid points and the 10th parameter will be multiplied by a global constant 1?
    No, the params are linear weights for the basis functions.
    
    Let's define 10 basis functions:
    phi_i(x, y) = exp(-|| (x, y) - c_i ||^2 / (2 * sigma^2))
    
    If I use 9 centers, I only have 9 basis functions. I have 10 params.
    I can add a 10th basis function which is a constant 1 (or a very wide RBF).
    Let's add a constant 1 as the 10th basis function. This acts as a global bias.
    
    Basis:
    1-9: RBFs at the 3x3 grid points.
    10: Constant 1.
    
    Sigma (bandwidth) is fixed. Since the domain is [-1, 1], a sigma of ~0.5-0.7 seems reasonable.
    Let's pick sigma = 0.6.
    
    The function is linear in params, so BFGS will converge quickly.
    """
    
    # Define centers
    # 3x3 grid
    xs = np.array([-1.0, 0.0, 1.0])
    ys = np.array([-1.0, 0.0, 1.0])
    
    centers = []
    for cx in xs:
        for cy in ys:
            centers.append((cx, cy))
    
    # 10th center: Let's use a wide RBF or just a constant?
    # The prompt suggests "10 fixed, spatially distributed centers".
    # Let's add (0, 0) again? No.
    # Let's add (0.5, 0.5)? It's already in the list? No, (0.5, 0.5) is NOT in {-1,0,1}x{-1,0,1}.
    # Wait, my grid is {-1, 0, 1}.
    # So (0.5, 0.5) is a good 10th center.
    centers.append((0.5, 0.5))
    
    centers = np.array(centers) # Shape (10, 2)
    
    # Compute squared distances from each point (x, y) to each center
    # x, y are 1D arrays of length N
    # centers are (10, 2)
    
    # Distances: || (x, y) - c_i ||^2 = (x - cx_i)^2 + (y - cy_i)^2
    # x[:, None] has shape (N, 1)
    # centers[:, 0] has shape (10,)
    # diff_x = x[:, None] - centers[:, 0][None, :] -> (N, 10)
    # diff_y = y[:, None] - centers[:, 1][None, :] -> (N, 10)
    
    diff_x = x[:, None] - centers[:, 0][None, :]
    diff_y = y[:, None] - centers[:, 1][None, :]
    
    dist_sq = diff_x**2 + diff_y**2
    
    # Fixed bandwidth
    sigma = 0.6
    sigma_sq = sigma * sigma
    
    # RBF values: exp(-dist_sq / (2 * sigma_sq))
    # Shape (N, 10)
    rbf_values = np.exp(-dist_sq / (2.0 * sigma_sq))
    
    # The 10th parameter is for the 10th center (0.5, 0.5).
    # This is a valid RBF.
    
    # Linear combination
    # rbf_values @ params
    # (N, 10) @ (10,) -> (N,)
    
    f = rbf_values @ params
    
    return f
