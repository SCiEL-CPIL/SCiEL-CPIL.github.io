"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin before 120s limit


def _elapsed_ok():
    return (time.time() - _START_TIME) < TIME_LIMIT


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Radial Basis Function (RBF) network with 10 Gaussian kernels.
    
    The 10 params are interpreted as 5 center points in 2D space:
    params = [c1x, c1y, c2x, c2y, c3x, c3y, c4x, c4y, c5x, c5y]
    
    For each input point (x_i, y_i), we compute the distance to each center,
    apply a Gaussian kernel, and sum the weighted outputs.
    
    Since BFGS fits the params, it will adjust the center locations to best
    fit the data. We use a fixed width for the Gaussian to keep the problem
    well-conditioned.
    
    The model is:
    f(x, y) = sum_{k=1}^{5} exp(-||[x,y] - c_k||^2 / (2*sigma^2))
    
    where c_k = (params[2*k], params[2*k+1]) and sigma is a fixed bandwidth.
    
    To make BFGS converge better, we add a small constant offset term that
    is also fitted. Actually, we have exactly 10 params, so let's use:
    - 5 centers (10 params)
    - The amplitude of each kernel is fixed to 1 (or we could use a scaled version)
    
    Wait, with amplitude fixed to 1, the model is quite restrictive. Let me think...
    
    Alternative: use 10 params as 5 centers, but also allow the overall scale to be
    captured. Actually, BFGS will fit the centers, and the sum of Gaussians will
    approximate the function. The amplitude being 1 might be too restrictive.
    
    Let's try a different approach: use the 10 params to define a more flexible model.
    
    New idea: Use 10 params as coefficients for a set of pre-defined RBF centers.
    But the centers need to be fixed or data-dependent.
    
    Actually, the cleanest approach: 
    - Fix 5 centers at well-chosen locations (e.g., grid points in [-1,1]^2)
    - Use the 10 params as: 5 amplitudes + 5 additional terms (or 10 amplitudes for 10 centers)
    
    Let me use 10 fixed centers and 10 amplitude params:
    Centers at a 3x3 grid plus 1 more, or a 5x2 grid, etc.
    
    Better: Use 10 centers on a regular grid in [-1, 1]^2, and the 10 params
    are the amplitudes of the Gaussian kernels centered at these locations.
    """
    n = len(x)
    
    # Define 10 fixed centers in [-1, 1]^2
    # Use a 3x3 grid (9 points) plus the origin, or a 5x2 grid
    # Let's use a 3x3 grid: x in {-1, 0, 1}, y in {-1, 0, 1}
    # That's 9 centers. Add one more at (0, 0) for a total of 10? No, (0,0) is already there.
    # Use a 4x3 grid or just 10 well-spaced points.
    
    # Let's use 10 centers: 5 on x-axis, 5 on y-axis, or a 2x5 grid
    # Actually, a 3x3 grid gives 9, let's add one at (0.5, 0.5)
    
    centers = np.array([
        [-1.0, -1.0],
        [0.0, -1.0],
        [1.0, -1.0],
        [-1.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [-1.0, 1.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5]
    ])
    
    # Compute distances from each input point to each center
    # x: (n,), y: (n,), centers: (10, 2)
    # dist: (n, 10)
    dx = x[:, np.newaxis] - centers[0, 0]  # This is wrong, let me redo
    dx = x[:, np.newaxis] - centers[:, 0][np.newaxis, :]  # (n, 10)
    dy = y[:, np.newaxis] - centers[:, 1][np.newaxis, :]  # (n, 10)
    dist_sq = dx ** 2 + dy ** 2  # (n, 10)
    
    # Gaussian kernel with fixed bandwidth
    sigma = 0.4  # Bandwidth parameter
    kernels = np.exp(-dist_sq / (2 * sigma ** 2))  # (n, 10)
    
    # params are the amplitudes for each kernel
    pred = kernels @ params
    
    # Ensure finite values
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return pred
# EVOLVE-BLOCK-END
