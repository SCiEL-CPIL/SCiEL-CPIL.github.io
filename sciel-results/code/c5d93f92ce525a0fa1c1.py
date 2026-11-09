import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure inputs are arrays
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    
    # Extract parameters
    p = params[:10]
    
    # Define a set of local basis functions (bumps) centered at different points in the domain [0.5, 2.0]
    # We use 7 bumps to cover the space, plus 3 linear terms.
    # Centers chosen to be somewhat evenly distributed but not overlapping too much.
    # Domain is [0.5, 2.0].
    
    # Centers for the bumps (u, v, w)
    centers = np.array([
        [0.7, 0.7, 0.7],
        [1.25, 1.25, 1.25],
        [1.8, 1.8, 1.8],
        [0.7, 1.25, 1.8],
        [1.8, 0.7, 1.25],
        [1.25, 1.8, 0.7],
        [1.25, 0.7, 1.8]
    ])
    
    # Width of the bumps (sigma)
    # A narrower width allows for more localized fitting.
    # If sigma is too small, the optimizer might struggle to find the right centers if they were free,
    # but here centers are fixed. A moderate width helps with conditioning.
    sigma = 0.4
    
    # Compute the 7 bump features
    # Distance squared from each point to each center
    # x, y, z are shape (N,)
    # centers is shape (7, 3)
    
    # Expand x, y, z to (N, 1) for broadcasting
    xx = x[:, np.newaxis]
    yy = y[:, np.newaxis]
    zz = z[:, np.newaxis]
    
    cx = centers[:, 0]
    cy = centers[:, 1]
    cz = centers[:, 2]
    
    # Distance squared: (N, 7)
    dist_sq = (xx - cx[np.newaxis, :])**2 + \
              (yy - cy[np.newaxis, :])**2 + \
              (zz - cz[np.newaxis, :])**2
    
    # Bump functions: exp(-dist_sq / (2 * sigma^2))
    # This creates a smooth, localized response around each center.
    bumps = np.exp(-dist_sq / (2 * sigma**2))
    
    # Linear terms: x, y, z
    # We use the raw inputs or normalized? 
    # Normalized might help conditioning, but BFGS is fairly robust.
    # Let's use normalized to keep magnitudes similar to the bumps (which are ~0..1).
    # u, v, w in [-1, 1]
    u = (x - 1.25) / 0.75
    v = (y - 1.25) / 0.75
    w = (z - 1.25) / 0.75
    
    # Construct the final prediction
    # p[0] * bump_0 + ... + p[6] * bump_6 + p[7] * u + p[8] * v + p[9] * w
    
    # Sum of weighted bumps
    # bumps is (N, 7), p[:7] is (7,)
    pred_bumps = np.sum(bumps * p[:7][np.newaxis, :], axis=1)
    
    # Linear part
    pred_linear = p[7] * u + p[8] * v + p[9] * w
    
    return pred_bumps + pred_linear
