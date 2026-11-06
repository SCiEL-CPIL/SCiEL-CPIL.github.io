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
    Objective function: Mean Reflectance + Penalty for total thickness.
    """
    R = _calc_R_vec(thicknesses, labels, materials, n_sub, n_amb, wavelengths)
    mean_R = np.mean(R)
    
    total_t = np.sum(thicknesses)
    if total_t > 1000.0:
        penalty = (total_t - 1000.0) * 0.1
    else:
        penalty = 0.0
        
    return mean_R + penalty

def _optimize_gd(objective, x0, bounds, max_iter=500, lr=10.0, momentum=0.9):
    """
    Simple Gradient Descent with Momentum and Adaptive Learning Rate.
    """
    x = np.array(x0, dtype=float)
    lower, upper = zip(*bounds)
    lower = np.array(lower)
    upper = np.array(upper)
    
    v = np.zeros_like(x)
    best_x = x.copy()
    best_val = objective(x)
    current_lr = lr
    
    for i in range(max_iter):
        # Compute gradient numerically (central difference)
        grad = np.zeros_like(x)
        h = 1e-4
        for j in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[j] += h
            x_minus[j] -= h
            # Ensure bounds are respected for gradient calculation? 
            # It's better to evaluate the objective at the perturbed points directly.
            # If perturbed points go out of bounds, the objective penalty will handle it, 
            # or we can clip. Clipping in the objective is safer.
            # However, for gradient, we assume smoothness within bounds.
            # Let's just evaluate.
            f_plus = objective(x_plus)
            f_minus = objective(x_minus)
            grad[j] = (f_plus - f_minus) / (2 * h)
            
        # Update velocity
        v = momentum * v - current_lr * grad
        
        # Update position
        x_new = x + v
        
        # Project to bounds
        x_new = np.clip(x_new, lower, upper)
        
        # Evaluate new objective
        val_new = objective(x_new)
        
        # Adaptive learning rate
        if val_new > best_val:
            # If it got worse, reduce learning rate and revert
            current_lr *= 0.5
            x = x_new # Still move, but slower next time? Or revert?
            # Reverting is safer for stability, but might stall.
            # Let's just keep x_new but reduce lr.
            if current_lr < 0.01:
                break
        else:
            # If it improved, we can slightly increase lr or keep it
            if val_new < best_val:
                best_val = val_new
                best_x = x_new.copy()
            # Optional: increase lr slightly if improvement is significant
            # current_lr *= 1.05
            
        x = x_new
        
    return best_x, best_val

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
            # Use the custom GD optimizer
            t_opt, val_opt = _optimize_gd(
                _objective,
                init_thicknesses,
                bounds,
                max_iter=300,
                lr=5.0,
                momentum=0.9
            )
            
            # Verify constraints
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
