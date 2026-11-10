# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time

def kakeya_needle_3d8() -> np.ndarray:
    """
    Optimizes the base positions of 64 sheared tubes to minimize the union volume.
    Uses a smooth surrogate for the volume and L-BFGS-B optimization.
    """
    N = 8
    start_time = time.time()
    timeout = 600.0
    safety_margin = 10.0
    
    # Initial positions: Keich binary construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)

    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    base_x = keich[i_flat]
    base_y = keich[j_flat]
    init_positions = np.stack([base_x, base_y], axis=1)
    
    # Flatten for optimizer
    p0 = init_positions.flatten()
    
    # Precompute indices for tubes
    i_vals = i_flat
    j_vals = j_flat
    
    # Grid for smooth volume integration
    # 10x10x10 = 1000 points
    grid_size = 10
    u_grid = np.linspace(0, 1, grid_size)
    v_grid = np.linspace(0, 1, grid_size)
    c_grid = np.linspace(0, 1, grid_size)
    u, v, c = np.meshgrid(u_grid, v_grid, c_grid, indexing='ij')
    u_flat = u.flatten()
    v_flat = v.flatten()
    c_flat = c.flatten()
    n_points = len(u_flat)
    
    beta = 50.0  # Steepness of sigmoid
    alpha = 20.0 # Steepness of soft min/max
    
    def soft_min(x):
        return -np.logaddexp.reduce(-alpha * x) / alpha
    
    def soft_max(x):
        return np.logaddexp.reduce(alpha * x) / alpha

    def smooth_volume(p_flat):
        p = p_flat.reshape(64, 2)
        x = p[:, 0]
        y = p[:, 1]
        
        # Compute soft bounds
        x1 = x + (i_vals + 1) / 8.0
        y1 = y + (j_vals + 1) / 8.0
        
        X0 = soft_min(x)
        X1 = soft_max(x1)
        Y0 = soft_min(y)
        Y1 = soft_max(y1)
        
        # Avoid division by zero or negative volume
        if X1 <= X0 or Y1 <= Y0:
            return 1e9
            
        box_vol = (X1 - X0) * (Y1 - Y0)
        
        # Map grid points to physical coordinates
        a = X0 + u_flat * (X1 - X0)
        b = Y0 + v_flat * (Y1 - Y0)
        
        # Compute smooth indicators for all tubes and points
        # I_k(a,b,c) = S_x * S_y
        # S_x = sigmoid(beta * (x_k + c*i_k/8 - a)) * sigmoid(beta * (x_k + c*i_k/8 + 1/8 - a))
        # We can vectorize over tubes (64) and points (1000)
        
        # Shapes:
        # x: (64,)
        # i_vals: (64,)
        # c_flat: (1000,)
        
        # x_center_x = x[:, None] + c_flat[None, :] * i_vals[:, None] / 8.0  # (64, 1000)
        # x_left = x_center_x
        # x_right = x_center_x + 1/8.0
        
        # sig_x_left = 1.0 / (1.0 + np.exp(-beta * (x_left - a[None, :])))
        # sig_x_right = 1.0 / (1.0 + np.exp(-beta * (x_right - a[None, :])))
        # sig_x = sig_x_left * sig_x_right
        
        # Similarly for y
        
        # To save memory and time, we can do this in chunks or just let NumPy handle it.
        # 64 * 1000 = 64000 floats. Very small.
        
        x_off = x[:, None] + c_flat[None, :] * i_vals[:, None] / 8.0
        y_off = y[:, None] + c_flat[None, :] * j_vals[:, None] / 8.0
        
        # a[None, :] is (1, 1000)
        # x_off is (64, 1000)
        
        # Sigmoid
        # Use stable sigmoid: 1 / (1 + exp(-z))
        # For large positive z, exp(-z) is 0, sig is 1.
        # For large negative z, exp(-z) is inf, sig is 0.
        
        sig_x_left = 1.0 / (1.0 + np.exp(-beta * (x_off - a[None, :])))
        sig_x_right = 1.0 / (1.0 + np.exp(-beta * (x_off + 1/8.0 - a[None, :])))
        sig_x = sig_x_left * sig_x_right
        
        sig_y_left = 1.0 / (1.0 + np.exp(-beta * (y_off - b[None, :])))
        sig_y_right = 1.0 / (1.0 + np.exp(-beta * (y_off + 1/8.0 - b[None, :])))
        sig_y = sig_y_left * sig_y_right
        
        I = sig_x * sig_y  # (64, 1000)
        
        # Union indicator: 1 - prod(1 - I_k)
        # prod over axis 0
        # If I is close to 1, 1-I is close to 0, prod is 0, union is 1.
        # If I is 0, 1-I is 1, prod is 1, union is 0.
        
        # Numerical stability: if any I is 1, prod is 0.
        # log(1-I) can be -inf.
        # Let's use: 1 - exp(sum(log(1-I)))
        # But 1-I can be 0.
        # Clip 1-I to [1e-15, 1]
        
        one_minus_I = 1.0 - I
        one_minus_I = np.clip(one_minus_I, 1e-15, 1.0)
        
        log_prod = np.sum(np.log(one_minus_I), axis=0)
        prod = np.exp(log_prod)
        
        union_ind = 1.0 - prod  # (1000,)
        
        # Integral approximation
        # Sum over grid points * cell volume
        # Cell volume in u,v,c space is (1/grid_size)^3
        # But we scaled a,b to [X0, X1] etc.
        # The integral is \int U da db dc.
        # da = (X1-X0) du, db = (Y1-Y0) dv.
        # So V = (X1-X0)(Y1-Y0) \int U du dv dc.
        # \int U du dv dc approx sum(U) * (1/grid_size)^3
        
        du = 1.0 / grid_size
        dv = 1.0 / grid_size
        dc = 1.0 / grid_size
        cell_vol = du * dv * dc
        
        V = box_vol * np.sum(union_ind) * cell_vol
        
        return V

    # Optimization
    # Use L-BFGS-B
    # We don't provide jac, so it uses numerical gradients.
    # This might be slow, but 128 dimensions is manageable if eval is fast.
    
    best_p = p0.copy()
    best_vol = smooth_volume(p0)
    
    try:
        # Options for L-BFGS-B
        # maxiter: limit iterations
        # ftol: tolerance for function value
        # gtol: tolerance for gradient
        # maxfun: limit function evaluations
        
        res = scipy.optimize.minimize(
            smooth_volume, 
            p0, 
            method='L-BFGS-B', 
            options={
                'maxiter': 200, 
                'ftol': 1e-10, 
                'gtol': 1e-8, 
                'maxfun': 500,
                'disp': False
            }
        )
        
        if res.success or res.fun < best_vol:
            best_p = res.x
            best_vol = res.fun
            
    except Exception as e:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > timeout - safety_margin:
        # Return best found so far
        return best_p.reshape(64, 2)
    
    # Optional: Local refinement with more precise volume?
    # For now, return the optimized positions.
    
    return best_p.reshape(64, 2)

# EVOLVE-BLOCK-END
