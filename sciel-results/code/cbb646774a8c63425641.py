import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Fixed Center Gaussian RBFs with Optimized Amplitudes.
    
    Instead of optimizing the centers of the RBFs (which creates a highly non-convex 
    landscape and causes BFGS to get stuck in local minima), we fix the centers to 
    a regular grid covering the input domain [0.5, 2.0]^3. We then optimize only 
    the amplitudes of these fixed RBFs.
    
    We have 10 parameters. We will use 10 fixed centers.
    To ensure coverage, we distribute these centers across the volume.
    
    A simple 2x2x2 grid gives 8 centers. We can add 2 more for better resolution 
    or use a different distribution. Let's use a 2x2x2 grid plus the center of the 
    domain and one corner, or simply a 10-point Halton-like or grid distribution.
    
    Actually, a 2x2x2 grid covers the corners and mid-edges? No, 2x2x2 is just corners.
    Let's use a 3x3x3 grid? That's 27 points, too many.
    
    Let's use 10 points:
    1. Center of domain (1.25, 1.25, 1.25)
    2. (0.5, 0.5, 0.5)
    3. (2.0, 0.5, 0.5)
    4. (0.5, 2.0, 0.5)
    5. (0.5, 0.5, 2.0)
    6. (2.0, 2.0, 0.5)
    7. (2.0, 0.5, 2.0)
    8. (0.5, 2.0, 2.0)
    9. (2.0, 2.0, 2.0)
    10. (1.25, 0.5, 1.25) -- An extra point on the edge to break symmetry?
    
    This set of 10 fixed centers provides a global basis. The optimization problem becomes:
    min_p sum_i (y_i - sum_j p_j * K(x_i, c_j))^2
    This is a linear least squares problem in p (if we ignore the fixed structure).
    BFGS should converge very quickly and reliably to the optimal amplitudes for this 
    fixed basis, avoiding the local minima traps of moving centers.
    """
    
    start_time = time.time()
    
    # Define 10 fixed centers in [0.5, 2.0]^3
    # We want good spatial coverage.
    # Corners:
    # (0.5, 0.5, 0.5), (2.0, 0.5, 0.5), (0.5, 2.0, 0.5), (0.5, 0.5, 2.0)
    # (2.0, 2.0, 0.5), (2.0, 0.5, 2.0), (0.5, 2.0, 2.0), (2.0, 2.0, 2.0)
    # Center:
    # (1.25, 1.25, 1.25)
    # One more:
    # (1.25, 0.5, 1.25) or similar.
    
    centers = np.array([
        [0.5, 0.5, 0.5],
        [2.0, 0.5, 0.5],
        [0.5, 2.0, 0.5],
        [0.5, 0.5, 2.0],
        [2.0, 2.0, 0.5],
        [2.0, 0.5, 2.0],
        [0.5, 2.0, 2.0],
        [2.0, 2.0, 2.0],
        [1.25, 1.25, 1.25],
        [1.25, 0.5, 1.25]
    ])
    
    # Width of the Gaussians. 
    # The domain size is 1.5. A width of ~0.5-0.7 allows overlap but maintains locality.
    # Let's fix sigma = 0.6.
    sigma = 0.6
    sigma2 = 2 * sigma * sigma
    
    # Compute distances from input points to each center
    # x, y, z are arrays of length N
    # centers is 10x3
    # We want to compute N x 10 matrix of squared distances
    
    # x is (N,), y is (N,), z is (N,)
    # centers[:, 0] is (10,)
    
    # Vectorized distance calculation
    # dx = x[:, None] - centers[None, :, 0] -> (N, 10)
    # This might be memory heavy for large N, but N=200 is small.
    
    dx = x[:, None] - centers[None, :, 0]
    dy = y[:, None] - centers[None, :, 1]
    dz = z[:, None] - centers[None, :, 2]
    
    sq_dist = dx**2 + dy**2 + dz**2
    
    # Gaussian kernel values
    # Clip exponent to avoid underflow/overflow issues, though exp(-large) is just 0
    exponents = -sq_dist / sigma2
    exponents = np.clip(exponents, -50, 0)
    
    K = np.exp(exponents)  # Shape (N, 10)
    
    # Prediction is K @ params
    # params is (10,)
    pred = K @ params
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 110:
         # Return current best (which is just the computed pred)
         # If we were doing iterative refinement, we'd stop here.
         # Since this is a single closed-form evaluation, we just return.
         pass
         
    return pred
