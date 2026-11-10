# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize

def compute_volume_mc(x_pos, y_pos, n_samples=10000, seed=42):
    """
    Compute the Monte Carlo estimate of the union volume.
    x_pos, y_pos: shape (64,)
    """
    N = 8
    # Reshape to (8, 8)
    x_grid = x_pos.reshape(8, 8)
    y_grid = y_pos.reshape(8, 8)
    
    # Generate random points
    # We need to define the bounding box first
    # X0 = min x_k, X1 = max (x_k + (i+1)/8)
    # But i depends on k. k = i*8 + j.
    # So for k, i = k//8.
    # x_k is the base.
    # The tube k at height c is [x_k + c*i/8, x_k + c*i/8 + 1/8].
    
    k = np.arange(64)
    i_idx = k // 8
    j_idx = k % 8
    
    # Calculate bounds
    # X_min = min_k x_k
    # X_max = max_k (x_k + (i_k+1)/8)
    # Note: The term is c*i/8. At c=1, shift is i/8. Width is 1/8.
    # So max x extent is x_k + i/8 + 1/8.
    # Wait, the definition says: [x_k + c*i/8, x_k + c*i/8 + 1/8].
    # So at c=0, it's [x_k, x_k+1/8].
    # At c=1, it's [x_k + i/8, x_k + i/8 + 1/8].
    
    x_starts = x_pos
    x_ends = x_pos + i_idx / 8.0 + 1.0/8.0
    
    y_starts = y_pos
    y_ends = y_pos + j_idx / 8.0 + 1.0/8.0
    
    X0 = np.min(x_starts)
    X1 = np.max(x_ends)
    Y0 = np.min(y_starts)
    Y1 = np.max(y_ends)
    
    if X1 - X0 < 1e-9 or Y1 - Y0 < 1e-9:
        return 1e9 # Penalty for degenerate box
        
    rng = np.random.default_rng(seed)
    a = rng.uniform(X0, X1, n_samples)
    b = rng.uniform(Y0, Y1, n_samples)
    c = rng.uniform(0, 1, n_samples)
    
    # Check if point (a, b, c) is inside any tube
    # Tube k is inside if:
    # x_k + c*i_k/8 <= a <= x_k + c*i_k/8 + 1/8
    # y_k + c*j_k/8 <= b <= y_k + c*j_k/8 + 1/8
    
    # Vectorize over k and samples
    # This might be memory heavy: 64 * 10000 = 640,000 booleans. That's fine.
    
    # a: (n_samples,)
    # x_k: (64,)
    # c: (n_samples,)
    # i_k: (64,)
    
    # We need to check for each sample s and each tube k.
    # left_x_k_s = x_k + c_s * i_k / 8
    # right_x_k_s = left_x_k_s + 1/8
    
    # Let's broadcast
    # a is (n_samples, 1)
    # left_x is (1, 64)
    
    a_col = a[:, np.newaxis]
    b_col = b[:, np.newaxis]
    c_col = c[:, np.newaxis]
    
    x_row = x_pos[np.newaxis, :]
    y_row = y_pos[np.newaxis, :]
    i_row = i_idx[np.newaxis, :]
    j_row = j_idx[np.newaxis, :]
    
    left_x = x_row + c_col * i_row / 8.0
    right_x = left_x + 1.0/8.0
    
    left_y = y_row + c_col * j_row / 8.0
    right_y = left_y + 1.0/8.0
    
    in_x = (a_col >= left_x) & (a_col <= right_x)
    in_y = (b_col >= left_y) & (b_col <= right_y)
    
    in_tube = in_x & in_y
    
    # A point is in the union if it is in at least one tube
    # in_tube is (n_samples, 64)
    in_union = np.any(in_tube, axis=1)
    
    fraction = np.mean(in_union)
    volume = fraction * (X1 - X0) * (Y1 - Y0)
    
    return volume

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    
    # Initial guess: Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        
    # The current program uses this for x and y.
    # We will optimize 128 variables: 64 x, 64 y.
    # Bounds: Let's say [-2, 2]
    bounds = [(-2.0, 2.0)] * 128
    
    # Objective function wrapper
    def obj(params):
        x_pos = params[:64]
        y_pos = params[64:]
        # Use a small sample size for speed during optimization
        # The final score uses 500k samples, but we just need the relative minimum.
        v = compute_volume_mc(x_pos, y_pos, n_samples=2000, seed=123)
        return v
        
    # Run Differential Evolution
    # This can be slow. Let's limit maxiter.
    res = differential_evolution(obj, bounds, seed=42, maxiter=50, tol=1e-6, popsize=15)
    
    best_params = res.x
    x_opt = best_params[:64]
    y_opt = best_params[64:]
    
    # Refine with local optimizer
    # L-BFGS-B might work if we provide gradients? No, numerical gradients.
    # Let's try a few steps of Nelder-Mead or L-BFGS-B on the refined result.
    
    def obj_refine(params):
        x_pos = params[:64]
        y_pos = params[64:]
        return compute_volume_mc(x_pos, y_pos, n_samples=5000, seed=123)
        
    res_refine = minimize(obj_refine, best_params, method='L-BFGS-B', 
                          bounds=bounds, options={'maxiter': 200, 'ftol': 1e-10})
    
    final_params = res_refine.x
    x_final = final_params[:64]
    y_final = final_params[64:]
    
    # Construct the output array
    # Row k is tube (i, j) with i = k // 8 and j = k % 8
    # positions[k] = [x_k, y_k]
    # x_k corresponds to x_pos[k]
    # y_k corresponds to y_pos[k]
    
    positions = np.stack([x_final, y_final], axis=1)
    
    return positions
# EVOLVE-BLOCK-END
