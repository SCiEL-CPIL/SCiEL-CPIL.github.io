"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Radial Basis Function (RBF) model for nonlinear oscillator acceleration.
    
    Strategy:
    Instead of a global polynomial expansion which may struggle with complex 
    nonlinear interactions or localized behaviors, we use a linear combination 
    of radial basis functions centered at fixed points in the input space.
    
    The input space is 3D: (t, x, v).
    We select a fixed set of 9 centers distributed uniformly across the 
    training input ranges to provide local coverage.
    
    Input Ranges:
    t: [30, 50] -> center 40, range 20
    x: [-0.2056, 0.1749] -> center -0.01535, range 0.3805
    v: [-0.3391, 0.3624] -> center 0.01165, range 0.7015
    
    Centers:
    We create a 3x3x3 grid of centers (total 27 is too many for 10 params, 
    so we use 9 centers: a 3x3 grid in t-x plane? No, we need to cover 3D.
    Actually, 10 params limit us. 
    Let's use 9 centers + 1 bias.
    To cover 3D with 9 centers, we can use a sparse grid or just pick 9 strategic points.
    Alternatively, we can use fewer centers with wider variance.
    
    Let's define 9 centers in the 3D space.
    t_centers: [35, 40, 45]
    x_centers: [-0.1, 0, 0.1]
    v_centers: [-0.2, 0, 0.2]
    
    If we take the Cartesian product, we get 27 points. We only have 9 weights (params[0:9]).
    We can't fit 27 centers with 9 params unless we constrain them.
    
    Alternative: Use 1D RBFs? No, inputs are coupled.
    
    Let's try a different approach: Use the 10 params to fit a small number of 
    "active" basis functions or use a different basis entirely.
    
    Wait, the prompt says "up to 10 fitted constants".
    
    Let's stick to a simpler, robust local approximation:
    Use 5 centers in 3D space? 
    Center 1: Low t, Low x, Low v
    Center 2: Low t, High x, High v
    ...
    
    Actually, a better "materially different" strategy that fits 10 params easily 
    is a **Gaussian Process-like kernel regression with fixed kernel centers**, 
    but we must ensure the optimizer can handle it.
    
    Let's use 9 centers defined by a 3x3 grid in the (x, v) plane, assuming t 
    dependence is weak or captured by the x/v dynamics? No, t is an input.
    
    Let's try: 5 centers in 3D + 5 polynomial terms? No, keep it pure RBF.
    
    Let's use 9 centers. 
    We will define 9 specific (t, x, v) tuples.
    t_vals = [30, 40, 50]
    x_vals = [-0.2, 0.0, 0.2]
    v_vals = [-0.3, 0.0, 0.3]
    
    If we just pick 9 points from the full 27 grid, we might miss regions.
    
    Better: Use a **Mixture of Radial Basis Functions** where the centers are 
    fixed, but we only have 9 weights. This is under-specified for 3D coverage 
    with 9 centers if the function is complex.
    
    However, RBFs with large variance act like low-order polynomials. 
    With small variance, they act like local interpolants.
    
    Let's use **10 centers** and **0 bias** (or 1 center and 9 others?).
    The interface allows `params` of length 10.
    
    Let's define 10 fixed centers in 3D space.
    We can sample 10 points from the training box.
    
    Centers:
    1. (30, -0.2, -0.3)
    2. (30, 0.2, 0.3)
    3. (40, -0.2, 0.3)
    4. (40, 0.2, -0.3)
    5. (50, -0.2, -0.3)
    6. (50, 0.2, 0.3)
    7. (35, 0.0, 0.0)
    8. (45, 0.0, 0.0)
    9. (30, 0.0, 0.0)
    10. (50, 0.0, 0.0)
    
    This distributes centers across the domain.
    
    The RBF is exp(-sigma^2 * ||p - c_i||^2).
    We need to choose sigma (width). A fixed sigma is risky.
    If sigma is too small, the matrix is ill-conditioned.
    If sigma is too large, it becomes a constant.
    
    Normalizing inputs helps.
    t_norm = (t - 40) / 10
    x_norm = (x - (-0.015)) / 0.2 -> approx [-1, 1]
    v_norm = (v - 0.01) / 0.35 -> approx [-1, 1]
    
    Let's normalize inputs to roughly [-1, 1] range.
    t_c = 40, t_s = 10
    x_c = -0.015, x_s = 0.2
    v_c = 0.01, v_s = 0.35
    
    Centers in normalized space:
    Let's pick 10 centers in the normalized cube [-1, 1]^3.
    
    c1 = (-1, -1, -1)
    c2 = (-1, 1, 1)
    c3 = (0, -1, 1)
    c4 = (0, 1, -1)
    c5 = (1, -1, -1)
    c6 = (1, 1, 1)
    c7 = (-0.5, 0, 0)
    c8 = (0.5, 0, 0)
    c9 = (0, -0.5, 0.5)
    c10 = (0, 0.5, -0.5)
    
    Sigma (width): 
    If we use sigma = 1.0 in normalized space, the RBFs have significant overlap.
    This ensures the design matrix is well-conditioned enough for BFGS.
    
    Formula:
    y = sum( params[i] * exp( -1.0 * ||p_norm - c_i||^2 ) )
    
    This is a linear model in params. BFGS will converge quickly.
    """
    
    # Normalization constants based on training ranges
    # t: [30, 50] -> mean 40, std 10 (approx)
    # x: [-0.2056, 0.1749] -> mean -0.01535, std 0.19 (approx)
    # v: [-0.3391, 0.3624] -> mean 0.01165, std 0.35 (approx)
    
    t_norm = (t - 40.0) / 10.0
    x_norm = (x - (-0.01535)) / 0.19
    v_norm = (v - 0.01165) / 0.35
    
    # Define 10 fixed centers in the normalized 3D space
    # These are hardcoded to cover the domain
    centers = np.array([
        [-1.0, -1.0, -1.0],
        [-1.0,  1.0,  1.0],
        [ 0.0, -1.0,  1.0],
        [ 0.0,  1.0, -1.0],
        [ 1.0, -1.0, -1.0],
        [ 1.0,  1.0,  1.0],
        [-0.5,  0.0,  0.0],
        [ 0.5,  0.0,  0.0],
        [ 0.0, -0.5,  0.5],
        [ 0.0,  0.5, -0.5]
    ])
    
    # Compute squared distances from each input point to each center
    # t_norm, x_norm, v_norm are arrays of shape (N,)
    # centers is (10, 3)
    
    # We want to compute sum_i params[i] * exp(-sigma^2 * dist^2)
    # dist^2 = (t_norm - c_t)^2 + (x_norm - c_x)^2 + (v_norm - c_v)^2
    
    # Efficient computation:
    # dist_sq = (t_norm[:, None] - centers[:, 0][None, :])**2 + 
    #           (x_norm[:, None] - centers[:, 1][None, :])**2 + 
    #           (v_norm[:, None] - centers[:, 2][None, :])**2
    
    # This creates an (N, 10) array.
    
    dist_t = (t_norm[:, None] - centers[:, 0][None, :])
    dist_x = (x_norm[:, None] - centers[:, 1][None, :])
    dist_v = (v_norm[:, None] - centers[:, 2][None, :])
    
    dist_sq = dist_t * dist_t + dist_x * dist_x + dist_v * dist_v
    
    # Choose a width parameter (sigma_inv). 
    # If sigma_inv = 1.0, the RBFs are fairly broad.
    # If sigma_inv = 2.0, they are narrower.
    # Let's use 1.5 for a balance.
    sigma_inv_sq = 2.25  # 1.5^2
    
    # Compute RBF values
    rbf_vals = np.exp(-sigma_inv_sq * dist_sq)
    
    # Linear combination with params
    # rbf_vals is (N, 10), params is (10,)
    # Result is (N,)
    res = rbf_vals @ params
    
    return res
# EVOLVE-BLOCK-END
