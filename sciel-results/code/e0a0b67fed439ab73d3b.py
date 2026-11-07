import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a Gaussian RBF basis with fixed centers.
    
    This structural change moves from a physics-inspired polynomial/trig basis to a
    Radial Basis Function network. The 10 parameters are the weights of 10 Gaussian
    kernels centered at fixed points covering the input domain. This creates a
    smooth, flexible surface that is linear in the parameters, making it highly
    amenable to BFGS optimization (convex problem for fixed centers).
    """
    # Fixed centers covering the input range x in [-0.6, 0.6], v in [-0.2, 0.3]
    # 5 x 2 grid
    centers = np.array([
        [-0.50, -0.15],
        [-0.50,  0.15],
        [-0.25, -0.15],
        [-0.25,  0.15],
        [ 0.00, -0.15],
        [ 0.00,  0.15],
        [ 0.25, -0.15],
        [ 0.25,  0.15],
        [ 0.50, -0.15],
        [ 0.50,  0.15]
    ])
    
    sigma = 0.25  # Width of the Gaussians
    
    # Compute distances from each input point to each center
    # x and v are 1D arrays of length N
    # centers is 10x2
    # We need to compute exp(-((x-cx)^2 + (v-cv)^2) / (2*sigma^2)) for each center
    
    # Vectorized calculation
    # x[:, np.newaxis] -> N x 1
    # v[:, np.newaxis] -> N x 1
    # centers[0, :] -> 1 x 2
    
    # Efficiently compute the RBF matrix
    # dist_sq = (x - cx)^2 + (v - cv)^2
    # rbf = exp(-dist_sq / (2 * sigma^2))
    
    # Using broadcasting:
    # x_diff = x[:, np.newaxis] - centers[:, 0][np.newaxis, :]  # N x 10
    # v_diff = v[:, np.newaxis] - centers[:, 1][np.newaxis, :]  # N x 10
    # dist_sq = x_diff**2 + v_diff**2
    # rbf_matrix = np.exp(-dist_sq / (2 * sigma**2))
    
    # Then prediction = rbf_matrix @ params
    
    x_diff = x[:, np.newaxis] - centers[:, 0][np.newaxis, :]
    v_diff = v[:, np.newaxis] - centers[:, 1][np.newaxis, :]
    dist_sq = x_diff**2 + v_diff**2
    
    rbf_matrix = np.exp(-dist_sq / (2 * sigma**2))
    
    return rbf_matrix @ params
