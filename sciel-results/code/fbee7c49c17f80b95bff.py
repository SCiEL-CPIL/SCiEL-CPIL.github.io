import numpy as np
import time
from scipy.optimize import minimize

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

def _objective_fast(thicknesses, labels, materials, n_sub, n_amb, wavelengths):
    """
    Fast objective with penalty for total thickness.
    """
    R = _calc_R_vec(thicknesses, labels, materials, n_sub, n_amb, wavelengths)
    mean_R = np.mean(R)
    
    total_t = np.sum(thicknesses)
    if total_t > 1000.0:
        penalty = (total_t - 1000.0) * 0.5
    else:
        penalty = 0.0
        
    return mean_R + penalty

def _optimize_sequence(labels, materials, n_sub, n_amb, wavelengths, 
                        max_layer_nm, max_total_nm, deadline, initial_guesses=None):
    """
    Optimize a single layer sequence using Nelder-Mead.
    """
    n_layers = len(labels)
    center_wl = 0.5 * (wavelengths[0] + wavelengths[-1])
    
    # Generate multiple initial guesses
    if initial_guesses is None:
        initial_guesses = []
        
        # Quarter-wave at center wavelength
        q_init = []
        for label in labels:
            n = materials[label]
            d = center_wl / (4.0 * n)
            q_init.append(np.clip(d, 0.0, max_layer_nm))
        initial_guesses.append(q_init)
        
        # Quarter-wave at different wavelengths
        for wl_frac in [0.4, 0.5, 0.6]:
            wl = wavelengths[0] + wl_frac * (wavelengths[-1] - wavelengths[0])
            q_init = []
            for label in labels:
                n = materials[label]
                d = wl / (4.0 * n)
                q_init.append(np.clip(d, 0.0, max_layer_nm))
            initial_guesses.append(q_init)
        
        # Random initializations
        rng = np.random.default_rng(42)
        for _ in range(5):
            init = rng.uniform(0, max_layer_nm * 0.5, size=n_layers)
            # Scale to stay under total budget
            total = np.sum(init)
            if total > max_total_nm * 0.8:
                init = init * (max_total_nm * 0.8 / total)
            initial_guesses.append(init)
    
    best_x = None
    best_val = float('inf')
    
    for x0 in initial_guesses:
        if time.time() > deadline:
            break
            
        x0 = np.array(x0, dtype=float)
        
        # Ensure initial point is feasible
        if np.sum(x0) > max_total_nm:
            x0 = x0 * (max_total_nm / np.sum(x0))
        x0 = np.clip(x0, 0.0, max_layer_nm)
        
        try:
            # Nelder-Mead optimization
            bounds = [(0.0, max_layer_nm)] * n_layers
            
            result = minimize(
                _objective_fast,
                x0,
                args=(labels, materials, n_sub, n_amb, wavelengths),
                method='Nelder-Mead',
                bounds=bounds,
                options={
                    'maxiter': 500,
                    'xatol': 1e-4,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            x_opt = np.clip(result.x, 0.0, max_layer_nm)
            
            # Check total thickness constraint
            if np.sum(x_opt) > max_total_nm + 1e-6:
                # Scale down proportionally
                x_opt = x_opt * (max_total_nm / np.sum(x_opt))
            
            val = _objective_fast(x_opt, labels, materials, n_sub, n_amb, wavelengths)
            
            if val < best_val:
                best_val = val
                best_x = x_opt.copy()
                
        except Exception:
            continue
    
    return best_x, best_val

def design(problem):
    """
    Design a broadband antireflection coating.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # 130s limit, leave 20s margin
    
    n_amb = problem['ambient_index']
    n_sub = problem['substrate_index']
    materials = problem['materials']
    band = problem['band_nm']
    max_layers = problem['max_layers']
    max_total_nm = problem['max_total_nm']
    max_layer_nm = problem['max_layer_nm']
    
    # Use fewer wavelengths for optimization, full set for final evaluation
    opt_wavelengths = np.arange(band[0], band[1] + 1, 20.0)
    eval_wavelengths = np.linspace(band[0], band[1], 401)
    
    # Layer sequences to try - focus on promising configurations
    sequences = [
        # 4-layer: classic broadband AR
        ['L', 'H', 'H', 'L'],
        # 6-layer
        ['L', 'H', 'H', 'H', 'H', 'L'],
        ['L', 'H', 'H', 'L', 'H', 'L'],
        # 8-layer
        ['L', 'H', 'H', 'L', 'L', 'H', 'H', 'L'],
        ['L', 'H', 'H', 'H', 'H', 'L', 'H', 'L'],
        # 10-layer
        ['L', 'H', 'H', 'H', 'H', 'L', 'L', 'H', 'H', 'L'],
        ['L', 'H', 'H', 'L', 'L', 'H', 'H', 'L', 'H', 'L'],
        # 12-layer
        ['L', 'H', 'H', 'H', 'H', 'L', 'L', 'H', 'H', 'H', 'H', 'L'],
        ['L', 'H', 'H', 'L', 'L', 'H', 'H', 'H', 'H', 'L', 'H', 'L'],
        # 14-layer
        ['L', 'H', 'H', 'L', 'L', 'H', 'H', 'H', 'H', 'L', 'L', 'H', 'H', 'L'],
        # 16-layer
        ['L', 'H', 'H', 'H', 'H', 'L', 'L', 'H', 'H', 'H', 'H', 'L', 'L', 'H', 'H', 'L'],
        # 18-layer
        ['L', 'H', 'H', 'H', 'H', 'L', 'L', 'H', 'H', 'H', 'H', 'L', 'L', 'H', 'H', 'H', 'H', 'L'],
    ]
    
    best_score = float('inf')
    best_materials = None
    best_thicknesses = None
    
    # Try each sequence
    for seq in sequences:
        if time.time() > deadline:
            break
            
        n_layers = len(seq)
        if n_layers > max_layers:
            continue
        
        try:
            t_opt, val_opt = _optimize_sequence(
                seq, materials, n_sub, n_amb, opt_wavelengths,
                max_layer_nm, max_total_nm, deadline
            )
            
            if t_opt is not None:
                # Verify constraints
                if np.sum(t_opt) <= max_total_nm + 1e-6 and all(0 <= t <= max_layer_nm for t in t_opt):
                    # Evaluate with full wavelength set
                    R_final = _calc_R_vec(t_opt, seq, materials, n_sub, n_amb, eval_wavelengths)
                    final_score = np.mean(R_final)
                    
                    if final_score < best_score:
                        best_score = final_score
                        best_materials = seq
                        best_thicknesses = t_opt.copy()
                        
        except Exception:
            continue
    
    # Final polish: refine the best solution with more precise optimization
    if best_materials is not None and time.time() < deadline - 10:
        try:
            # Use full wavelength set for final polish
            x0 = best_thicknesses.copy()
            
            result = minimize(
                _objective_fast,
                x0,
                args=(best_materials, materials, n_sub, n_amb, eval_wavelengths),
                method='Nelder-Mead',
                bounds=[(0.0, max_layer_nm)] * len(best_materials),
                options={
                    'maxiter': 200,
                    'xatol': 1e-5,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            t_polished = np.clip(result.x, 0.0, max_layer_nm)
            
            if np.sum(t_polished) > max_total_nm + 1e-6:
                t_polished = t_polished * (max_total_nm / np.sum(t_polished))
            
            R_polished = _calc_R_vec(t_polished, best_materials, materials, n_sub, n_amb, eval_wavelengths)
            score_polished = np.mean(R_polished)
            
            if score_polished < best_score:
                best_score = score_polished
                best_thicknesses = t_polished.copy()
                
        except Exception:
            pass
    
    # Fallback
    if best_materials is None:
        best_materials = ['L']
        center_wl = 0.5 * (band[0] + band[1])
        best_thicknesses = [center_wl / (4.0 * materials['L'])]
        
    return best_materials, [float(t) for t in best_thicknesses]
