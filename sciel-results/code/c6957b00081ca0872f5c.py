# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Uses a fixed set of 10 Gaussian Radial Basis Functions (RBFs).
    The centers are strategically placed to cover the input domain [0.5, 2]^3.
    The 10 parameters are the weights for these basis functions.
    This allows the linear model to approximate smooth non-polynomial functions 
    more flexibly than a fixed polynomial basis.
    """
    p = params
    
    # Define 10 centers in [0.5, 2]^3
    # We use a mix of center, corners, and face centers to ensure coverage.
    # Domain is [0.5, 2.0]. Midpoint is 1.25.
    # Let's use a 2x2x2 grid of points plus the center and one more?
    # Actually, 10 points.
    # Points:
    # 0: Center (1.25, 1.25, 1.25)
    # 1-8: Corners of a smaller box or the domain corners?
    # Let's use the 8 corners of the domain [0.5, 2.0] and the center, and one more.
    # Corners:
    # (0.5, 0.5, 0.5), (0.5, 0.5, 2.0), (0.5, 2.0, 0.5), (0.5, 2.0, 2.0)
    # (2.0, 0.5, 0.5), (2.0, 0.5, 2.0), (2.0, 2.0, 0.5), (2.0, 2.0, 2.0)
    # Center: (1.25, 1.25, 1.25)
    # 10th: Maybe (1.25, 1.25, 0.5) or similar? Or just duplicate center?
    # Better: Use a slightly wider width to ensure overlap.
    
    centers = np.array([
        [1.25, 1.25, 1.25], # 0: Center
        [0.5,  0.5,  0.5],  # 1
        [0.5,  0.5,  2.0],  # 2
        [0.5,  2.0,  0.5],  # 3
        [0.5,  2.0,  2.0],  # 4
        [2.0,  0.5,  0.5],  # 5
        [2.0,  0.5,  2.0],  # 6
        [2.0,  2.0,  0.5],  # 7
        [2.0,  2.0,  2.0],  # 8
        [1.25, 1.25, 0.5]   # 9: Extra point to break symmetry/cover bottom face
    ])
    
    # Width (sigma). 
    # Domain size is 1.5. 
    # If sigma is too small, the basis is local and underfits.
    # If sigma is too large, it's like a polynomial.
    # Let's try sigma = 0.8.
    sigma = 0.8
    
    # Compute RBFs
    # x, y, z are arrays.
    # We can vectorize this.
    # Shape of x, y, z: (N,)
    # Shape of centers: (10, 3)
    
    # Compute squared distances
    # dist_sq[i, j] = (x[i]-c[j,0])^2 + (y[i]-c[j,1])^2 + (z[i]-c[j,2])^2
    # Using broadcasting:
    # x[:, np.newaxis] - centers[np.newaxis, :, 0]
    
    dx = x[:, np.newaxis] - centers[:, 0]
    dy = y[:, np.newaxis] - centers[:, 1]
    dz = z[:, np.newaxis] - centers[:, 2]
    
    dist_sq = dx**2 + dy**2 + dz**2
    
    # Gaussian: exp(-dist_sq / (2 * sigma^2))
    # To avoid underflow/overflow issues, we just compute it.
    # 2 * sigma^2 = 2 * 0.64 = 1.28
    rbf_values = np.exp(-dist_sq / 1.28)
    
    # rbf_values shape: (N, 10)
    # params shape: (10,)
    # pred = rbf_values @ params
    
    pred = rbf_values @ p
    
    return pred
# EVOLVE-BLOCK-END
