"""
Broadband antireflection coating design via multi-start L-BFGS-B optimization.

Strategy:
- Use scipy.optimize.minimize with L-BFGS-B for fast gradient-based local optimization
- Multiple restarts with different initial thicknesses and layer counts
- Try various layer patterns (alternating L/H starting with L or H)
- Handle the total thickness constraint via penalty or projection
- Explicit time deadline guard to return best result before timeout
"""
import numpy as np
import time
from scipy.optimize import minimize

def _compute_reflectance(wavelengths_nm, indices, thicknesses_nm, ambient_idx, substrate_idx):
    """
    Compute reflectance (fraction, 0-1) at each wavelength using the characteristic matrix method.
    """
    n_layers = len(indices)
    if n_layers == 0:
        r0 = (ambient_idx - substrate_idx) / (ambient_idx + substrate_idx)
        return np.full(len(wavelengths_nm), r0 * r0)
    
    k = 2.0 * np.pi / wavelengths_nm
    
    M00 = np.ones(len(wavelengths_nm), dtype=complex)
    M01 = np.zeros(len(wavelengths_nm), dtype=complex)
    M10 = np.zeros(len(wavelengths_nm), dtype=complex)
    M11 = np.ones(len(wavelengths_nm), dtype=complex)
    
    for i in range(n_layers):
        n_i = indices[i]
        d_i = thicknesses_nm[i]
        delta = k * n_i * d_i
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        B00 = cos_d
        B01 = 1j * sin_d / n_i
        B10 = 1j * n_i * sin_d
        B11 = cos_d
        
        new_M00 = M00 * B00 + M01 * B10
        new_M01 = M00 * B01 + M01 * B11
        new_M10 = M10 * B00 + M11 * B10
        new_M11 = M10 * B01 + M11 * B11
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    Y_out = substrate_idx
    Y_in = (M10 + M11 * Y_out) / (M00 + M01 * Y_out)
    
    r = (ambient_idx - Y_in) / (ambient_idx + Y_in)
    R = np.abs(r) ** 2
    
    return R

def _objective_and_grad(x, indices, wavelengths_nm, ambient_idx, substrate_idx,
                        max_layer_nm, max_total_nm):
    """
    Compute objective (mean reflectance in percent) and its gradient.
    Uses numerical differentiation for the gradient.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    
    # Clip to valid range
    x_clipped = np.clip(x, 0.0, max_layer_nm)
    
    # Check total thickness constraint
    if np.sum(x_clipped) > max_total_nm:
        # Scale down to satisfy constraint
        x_clipped = x_clipped * (max_total_nm / np.sum(x_clipped))
    
    R = _compute_reflectance(wavelengths_nm, indices, x_clipped, ambient_idx, substrate_idx)
    f_val = np.mean(R) * 100.0
    
    # Numerical gradient
    grad = np.zeros(n)
    eps = 1e-6
    for i in range(n):
        x_plus = x_clipped.copy()
        x_plus[i] = min(x_plus[i] + eps, max_layer_nm)
        if np.sum(x_plus) > max_total_nm:
            x_plus = x_plus * (max_total_nm / np.sum(x_plus))
        R_plus = _compute_reflectance(wavelengths_nm, indices, x_plus, ambient_idx, substrate_idx)
        f_plus = np.mean(R_plus) * 100.0
        
        x_minus = x_clipped.copy()
        x_minus[i] = max(x_minus[i] - eps, 0.0)
        if np.sum(x_minus) > max_total_nm:
            x_minus = x_minus * (max_total_nm / np.sum(x_minus))
        R_minus = _compute_reflectance(wavelengths_nm, indices, x_minus, ambient_idx, substrate_idx)
        f_minus = np.mean(R_minus) * 100.0
        
        grad[i] = (f_plus - f_minus) / (2.0 * eps)
    
    return f_val, grad

def _objective_only(x, indices, wavelengths_nm, ambient_idx, substrate_idx,
                    max_layer_nm, max_total_nm):
    """Objective function for scipy.optimize.minimize."""
    x = np.asarray(x, dtype=float)
    x_clipped = np.clip(x, 0.0, max_layer_nm)
    if np.sum(x_clipped) > max_total_nm:
        x_clipped = x_clipped * (max_total_nm / np.sum(x_clipped))
    R = _compute_reflectance(wavelengths_nm, indices, x_clipped, ambient_idx, substrate_idx)
    return np.mean(R) * 100.0

def _project_to_feasible(x, max_layer_nm, max_total_nm):
    """Project x onto the feasible region: each in [0, max_layer_nm], sum <= max_total_nm."""
    x = np.clip(x, 0.0, max_layer_nm)
    if np.sum(x) > max_total_nm:
        x = x * (max_total_nm / np.sum(x))
    return x

def _optimize_single_config(labels, materials, wavelengths_nm, ambient_idx, substrate_idx,
                            max_layer_nm, max_total_nm, center_wl, time_deadline, rng):
    """
    Optimize a single layer configuration using L-BFGS-B with multiple restarts.
    Returns (best_x, best_f, best_labels).
    """
    indices = [materials[label] for label in labels]
    n_layers = len(indices)
    bounds = [(0.0, float(max_layer_nm))] * n_layers
    
    best_x = None
    best_f = np.inf
    
    # Generate multiple initial points
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > time_deadline:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Quarter-wave at center wavelength
            x0 = np.array([center_wl / (4.0 * n_i) for n_i in indices])
        elif restart == 1:
            # Quarter-wave with slight perturbation
            x0 = np.array([center_wl / (4.0 * n_i) * (1.0 + 0.1 * rng.standard_normal()) for n_i in indices])
        elif restart == 2:
            # Random feasible point
            x0 = rng.uniform(0, max_layer_nm, size=n_layers)
            x0 = _project_to_feasible(x0, max_layer_nm, max_total_nm)
        elif restart == 3:
            # Geometric progression inspired
            ratio = (1.52 / 1.0) ** (1.0 / max(n_layers - 1, 1))
            x0 = np.array([center_wl / (4.0 * n_i) * ratio ** (i - n_layers // 2) 
                          for i, n_i in enumerate(indices)])
        elif restart == 4:
            # All same thickness
            d = max_total_nm / n_layers
            x0 = np.full(n_layers, min(d, max_layer_nm))
        elif restart == 5:
            # Linear progression
            d_min = max_total_nm / (n_layers * 2)
            d_max = max_total_nm / (n_layers * 2) * 2
            x0 = np.linspace(d_min, d_max, n_layers)
            x0 = np.clip(x0, 0, max_layer_nm)
            if np.sum(x0) > max_total_nm:
                x0 = x0 * (max_total_nm / np.sum(x0))
        elif restart == 6:
            # Random with bias toward quarter-wave
            x0 = np.array([center_wl / (4.0 * n_i) * (0.5 + 0.8 * rng.random()) for n_i in indices])
            x0 = _project_to_feasible(x0, max_layer_nm, max_total_nm)
        else:
            # Pure random
            x0 = rng.uniform(0, max_layer_nm * 0.8, size=n_layers)
            x0 = _project_to_feasible(x0, max_layer_nm, max_total_nm)
        
        x0 = _project_to_feasible(x0, max_layer_nm, max_total_nm)
        
        # Evaluate initial point
        f0 = _objective_only(x0, indices, wavelengths_nm, ambient_idx, substrate_idx,
                            max_layer_nm, max_total_nm)
        
        if f0 < best_f:
            best_f = f0
            best_x = x0.copy()
        
        # Run L-BFGS-B optimization
        try:
            result = minimize(
                _objective_only,
                x0,
                args=(indices, wavelengths_nm, ambient_idx, substrate_idx,
                      max_layer_nm, max_total_nm),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if result.success and result.fun < best_f:
                best_f = result.fun
                best_x = result.x.copy()
            elif result.fun < best_f:
                best_f = result.fun
                best_x = result.x.copy()
                
        except Exception:
            pass
    
    if best_x is None:
        return None, np.inf, labels
    
    # Final projection
    best_x = _project_to_feasible(best_x, max_layer_nm, max_total_nm)
    return best_x, best_f, labels

def design(problem):
    """
    Design a broadband antireflection coating.
    Returns: (labels, thicknesses)
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_idx"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wavelengths_nm = np.linspace(band_nm[0], band_nm[1], 401)
    center_wl = 0.5 * (band_nm[0] + band_nm[1])
    
    rng = np.random.default_rng(42)
    
    best_overall_x = None
    best_overall_f = np.inf
    best_overall_labels = None
    
    # Try various layer configurations
    # Alternating patterns with different starting layers and counts
    layer_configs = []
    
    # Odd number of layers (start and end with same type)
    for n in [5, 7, 9, 11, 13, 15, 17, 19]:
        if n > max_layers:
            continue
        # Start with L
        labels_L = ['L' if i % 2 == 0 else 'H' for i in range(n)]
        layer_configs.append(labels_L)
        # Start with H
        labels_H = ['H' if i % 2 == 0 else 'L' for i in range(n)]
        layer_configs.append(labels_H)
    
    # Also try some even counts
    for n in [6, 8, 10, 12, 14, 16, 18, 20]:
        if n > max_layers:
            continue
        labels_L = ['L' if i % 2 == 0 else 'H' for i in range(n)]
        layer_configs.append(labels_L)
        labels_H = ['H' if i % 2 == 0 else 'L' for i in range(n)]
        layer_configs.append(labels_H)
    
    # Sort by complexity (fewer layers first) to get quick wins
    layer_configs.sort(key=len)
    
    # Optimize each configuration
    for labels in layer_configs:
        if time.time() > deadline - 5.0:
            break
        
        try:
            x_opt, f_opt, lbl_opt = _optimize_single_config(
                labels, materials, wavelengths_nm, ambient_idx, substrate_idx,
                max_layer_nm, max_total_nm, center_wl, deadline - 5.0, rng
            )
            
            if x_opt is not None and f_opt < best_overall_f:
                best_overall_f = f_opt
                best_overall_x = x_opt.copy()
                best_overall_labels = lbl_opt.copy()
        except Exception:
            continue
    
    # If no optimization succeeded, use a simple quarter-wave design
    if best_overall_x is None:
        n = 7
        labels = ['L' if i % 2 == 0 else 'H' for i in range(n)]
        indices = [materials[label] for label in labels]
        thicknesses = np.array([center_wl / (4.0 * n_i) for n_i in indices])
        thicknesses = _project_to_feasible(thicknesses, max_layer_nm, max_total_nm)
        best_overall_labels = labels
        best_overall_x = thicknesses
        best_overall_f = _objective_only(thicknesses, indices, wavelengths_nm, 
                                         ambient_idx, substrate_idx, max_layer_nm, max_total_nm)
    
    # Final constraint check
    thicknesses = np.asarray(best_overall_x, dtype=float)
    thicknesses = _project_to_feasible(thicknesses, max_layer_nm, max_total_nm)
    
    return best_overall_labels, thicknesses.tolist()
