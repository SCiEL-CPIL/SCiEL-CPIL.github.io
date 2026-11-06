"""Broadband antireflection coating design via vectorized characteristic-matrix evaluation and scipy optimization."""
import numpy as np
import time

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

def _objective(params, indices, wavelengths_nm, ambient_idx, substrate_idx, max_layer_nm, max_total_nm):
    """
    Objective function: mean reflectance in percent with penalty for constraint violations.
    """
    thicknesses = np.clip(params, 0, max_layer_nm)
    
    total = np.sum(thicknesses)
    if total > max_total_nm:
        penalty = (total - max_total_nm) * 0.1
        return 100.0 + penalty
    
    R = _compute_reflectance(wavelengths_nm, indices, thicknesses, ambient_idx, substrate_idx)
    return np.mean(R) * 100.0

def _optimize_thicknesses(labels, problem, wavelengths_nm, start_time, time_budget):
    """
    Optimize thicknesses for a given label sequence using scipy.optimize.
    Returns (best_thicknesses, best_score) or (None, inf) on failure.
    """
    from scipy.optimize import minimize
    
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    materials = problem["materials"]
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    indices = [materials[label] for label in labels]
    n_layers = len(labels)
    
    center_wl = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    
    # Initial guess: quarter-wave
    x0 = np.array([center_wl / (4.0 * n) for n in indices])
    
    # Bounds
    bounds = [(0, max_layer_nm)] * n_layers
    
    def obj(x):
        return _objective(x, indices, wavelengths_nm, ambient_idx, substrate_idx, max_layer_nm, max_total_nm)
    
    best_x = x0.copy()
    best_f = obj(x0)
    
    # Try Nelder-Mead
    try:
        result = minimize(obj, x0, method='Nelder-Mead', bounds=bounds,
                         options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
        if result.success or result.fun < best_f:
            best_x = result.x
            best_f = result.fun
    except Exception:
        pass
    
    # Clip and check constraints
    best_x = np.clip(best_x, 0, max_layer_nm)
    if np.sum(best_x) > max_total_nm:
        best_x = best_x * (max_total_nm / np.sum(best_x))
    
    # Final evaluation
    final_f = obj(best_x)
    
    return best_x, final_f

def design(problem):
    """
    Design a broadband antireflection coating.
    Returns: (labels, thicknesses)
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wavelengths_nm = np.linspace(band_nm[0], band_nm[1], 401)
    
    best_result = None
    best_score = np.inf
    
    # Define candidate label patterns to try
    # Start with promising configurations
    candidates = []
    
    # Single layer
    candidates.append(['L'])
    candidates.append(['H'])
    
    # 3 layers
    candidates.append(['L', 'H', 'L'])
    candidates.append(['H', 'L', 'H'])
    
    # 5 layers
    candidates.append(['L', 'H', 'L', 'H', 'L'])
    candidates.append(['H', 'L', 'H', 'L', 'H'])
    
    # 7 layers
    candidates.append(['L', 'H', 'L', 'H', 'L', 'H', 'L'])
    candidates.append(['H', 'L', 'H', 'L', 'H', 'L', 'H'])
    
    # 9 layers
    candidates.append(['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'])
    candidates.append(['H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'])
    
    # 11 layers
    candidates.append(['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'])
    
    # 13 layers
    candidates.append(['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'])
    
    # 15 layers
    candidates.append(['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'])
    
    # 17 layers
    candidates.append(['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'])
    
    # 19 layers
    candidates.append(['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'])
    
    # Filter candidates that fit within max_layers
    candidates = [c for c in candidates if len(c) <= max_layers]
    
    # Try each candidate
    for labels in candidates:
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
        
        # Allocate time budget based on remaining time and number of remaining candidates
        remaining_candidates = len(candidates) - candidates.index(labels)
        remaining_time = deadline - 10 - elapsed
        time_budget = max(5.0, remaining_time / remaining_candidates)
        
        try:
            thicknesses, score = _optimize_thicknesses(labels, problem, wavelengths_nm, start_time, time_budget)
        except Exception:
            continue
        
        if thicknesses is not None and score < best_score:
            best_score = score
            best_result = (labels, thicknesses.tolist())
        
        # Early termination if we're doing very well
        if best_score < 0.5:
            break
    
    # If no result found, fall back to single quarter-wave layer
    if best_result is None:
        center_wl = 0.5 * (band_nm[0] + band_nm[1])
        return ["L"], [center_wl / (4.0 * materials["L"])]
    
    labels, thicknesses = best_result
    
    # Final constraint enforcement
    thicknesses = [max(0, min(t, max_layer_nm)) for t in thicknesses]
    total = sum(thicknesses)
    if total > max_total_nm:
        scale = max_total_nm / total
        thicknesses = [t * scale for t in thicknesses]
    
    return labels, thicknesses
