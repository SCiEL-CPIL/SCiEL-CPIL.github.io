import numpy as np
import time

def _calc_R_vec(thicknesses, labels, materials, n_sub, n_amb, wavelengths):
    """
    Vectorized calculation of reflectance for a stack of layers.
    """
    n_layers = len(thicknesses)
    n_w = len(wavelengths)
    
    M_total = np.zeros((2, 2, n_w), dtype=np.complex128)
    M_total[0, 0, :] = 1.0
    M_total[1, 1, :] = 1.0
    
    for k in range(n_layers):
        n_k = materials[labels[k]]
        d_k = thicknesses[k]
        
        delta = 2.0 * np.pi * n_k * d_k / wavelengths
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        M_k = np.zeros((2, 2, n_w), dtype=np.complex128)
        M_k[0, 0, :] = cos_d
        M_k[0, 1, :] = (1.0j / n_k) * sin_d
        M_k[1, 0, :] = (1.0j * n_k) * sin_d
        M_k[1, 1, :] = cos_d
        
        M_total = np.einsum('ilw,ljw->ijw', M_k, M_total)
        
    M11 = M_total[0, 0, :]
    M12 = M_total[0, 1, :]
    M21 = M_total[1, 0, :]
    M22 = M_total[1, 1, :]
    
    Y_in = (M21 + M22 * n_sub) / (M11 + M12 * n_sub)
    
    r = (Y_in - n_amb) / (Y_in + n_amb)
    R = np.abs(r)**2
    
    return R

def _objective(thicknesses, labels, materials, n_sub, n_amb, wavelengths):
    """
    Objective function: Mean Reflectance.
    """
    R = _calc_R_vec(thicknesses, labels, materials, n_sub, n_amb, wavelengths)
    return np.mean(R)

def _objective_with_penalty(thicknesses, labels, materials, n_sub, n_amb, wavelengths, max_total_nm):
    """
    Objective function with smooth penalty for total thickness.
    """
    R = _calc_R_vec(thicknesses, labels, materials, n_sub, n_amb, wavelengths)
    mean_R = np.mean(R)
    
    total_t = np.sum(thicknesses)
    excess = max(0.0, total_t - max_total_nm)
    penalty = 0.01 * excess**2
    
    return mean_R + penalty

def _optimize_coordinate_descent(objective, x0, bounds, max_iter=100, tol=1e-8, time_limit=60.0):
    """
    Simple coordinate descent optimization with local line search.
    """
    x = np.array(x0, dtype=float)
    n_vars = len(x)
    start_time = time.time()
    
    # Initial evaluation
    current_val = objective(x)
    
    # Determine step sizes for line search
    # We will use a simple grid search within bounds for each coordinate
    # To speed up, we can use a coarse-to-fine approach or just a fixed number of points
    n_points = 20 # Points to sample in 1D line search
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        improved = False
        
        for i in range(n_vars):
            if time.time() - start_time > time_limit:
                break
                
            low, high = bounds[i]
            
            # Sample points in the interval [low, high]
            # Include the current value to ensure we don't miss it
            candidates = np.linspace(low, high, n_points)
            
            best_val = current_val
            best_x_i = x[i]
            
            for x_i in candidates:
                x_temp = x.copy()
                x_temp[i] = x_i
                val = objective(x_temp)
                if val < best_val:
                    best_val = val
                    best_x_i = x_i
            
            if best_val < current_val - tol:
                x[i] = best_x_i
                current_val = best_val
                improved = True
        
        if not improved:
            break
            
    return x, current_val

def design(problem):
    """
    Design a broadband antireflection coating.
    """
    start_time = time.time()
    deadline = start_time + 110.0 # 110s limit, leave margin
    
    n_amb = problem['ambient_index']
    n_sub = problem['substrate_index']
    materials = problem['materials']
    band = problem['band_nm']
    max_layers = problem['max_layers']
    max_total_nm = problem['max_total_nm']
    max_layer_nm = problem['max_layer_nm']
    
    opt_wavelengths = np.arange(band[0], band[1] + 1, 10.0)
    
    sequences = [
        ['L', 'H', 'H', 'L'],
        ['L', 'H', 'H', 'H', 'H', 'L'],
        ['L', 'H', 'H', 'L', 'L', 'H', 'H', 'L'],
        ['L', 'H', 'H', 'H', 'H', 'L', 'L', 'H', 'H', 'H', 'H', 'L'],
        ['L', 'H', 'H', 'L', 'L', 'H', 'H', 'H', 'H', 'L', 'L', 'H', 'H', 'L'],
        ['L', 'H', 'H', 'H', 'H', 'L', 'L', 'H', 'H', 'H', 'H', 'L', 'L', 'H', 'H', 'H', 'H', 'L']
    ]
    
    best_score = float('inf')
    best_materials = None
    best_thicknesses = None
    
    center_wl = 0.5 * (band[0] + band[1])
    
    for seq in sequences:
        if time.time() > deadline:
            break
            
        n_layers = len(seq)
        if n_layers > max_layers:
            continue
            
        init_thicknesses = []
        for label in seq:
            n = materials[label]
            d = center_wl / (4.0 * n)
            d = np.clip(d, 0.0, max_layer_nm)
            init_thicknesses.append(d)
            
        bounds = [(0.0, max_layer_nm) for _ in range(n_layers)]
        
        try:
            # Use Coordinate Descent optimizer
            obj_func = lambda x: _objective_with_penalty(x, seq, materials, n_sub, n_amb, opt_wavelengths, max_total_nm)
            
            # Allocate time based on remaining budget
            remaining_time = deadline - time.time()
            # Reserve 10s for final checks and other sequences
            alloc_time = max(1.0, remaining_time - 10.0)
            
            t_opt, val_opt = _optimize_coordinate_descent(
                obj_func,
                init_thicknesses,
                bounds,
                max_iter=50, # Coordinate descent iterations
                tol=1e-8,
                time_limit=alloc_time
            )
            
            # Verify constraints
            t_opt = np.clip(t_opt, 0.0, max_layer_nm)
            if np.sum(t_opt) <= max_total_nm + 1e-6 and all(0 <= t <= max_layer_nm for t in t_opt):
                eval_wavelengths = np.linspace(band[0], band[1], 401)
                R_final = _calc_R_vec(t_opt, seq, materials, n_sub, n_amb, eval_wavelengths)
                final_score = np.mean(R_final)
                
                if final_score < best_score:
                    best_score = final_score
                    best_materials = seq
                    best_thicknesses = t_opt
                    
        except Exception:
            pass

    if best_materials is None:
        best_materials = ['L']
        best_thicknesses = [center_wl / (4.0 * materials['L'])]
        
    return best_materials, [float(t) for t in best_thicknesses]
