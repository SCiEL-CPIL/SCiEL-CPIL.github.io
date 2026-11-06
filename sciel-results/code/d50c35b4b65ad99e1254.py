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

def _objective(params, problem, wavelengths_nm):
    """
    Objective function: mean reflectance in percent + penalty for constraint violation.
    """
    max_layer_nm = problem.get("max_layer_nm", 400)
    max_total_nm = problem.get("max_total_nm", 1000)
    
    thicknesses = np.clip(params, 0, max_layer_nm)
    
    total_thick = np.sum(thicknesses)
    penalty = 0.0
    if total_thick > max_total_nm:
        penalty = 100.0 * (total_thick - max_total_nm) ** 2
    
    indices = [problem["materials"][label] for label in problem["labels"]]
    
    R = _compute_reflectance(wavelengths_nm, indices, thicknesses, 
                             problem["ambient_index"], problem["substrate_index"])
    
    mean_R_percent = np.mean(R) * 100.0
    return mean_R_percent + penalty

def design(problem):
    start_time = time.time()
    time_limit = 145.0
    
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wavelengths_nm = np.linspace(band_nm[0], band_nm[1], 401)
    
    best_score = np.inf
    best_labels = None
    best_thicknesses = None
    
    layer_counts = [20, 18, 16, 14, 12]
    
    for n_layers in layer_counts:
        if time.time() - start_time > time_limit - 5.0:
            break
        if n_layers > max_layers:
            continue
            
        patterns = [
            ['L' if i % 2 == 0 else 'H' for i in range(n_layers)],
            ['H' if i % 2 == 0 else 'L' for i in range(n_layers)]
        ]
        
        for labels in patterns:
            if time.time() - start_time > time_limit - 2.0:
                break
                
            problem_local = dict(problem)
            problem_local["labels"] = labels
            
            indices = [materials[label] for label in labels]
            center_wl = 0.5 * (band_nm[0] + band_nm[1])
            
            x0_qw = np.array([center_wl / (4.0 * n) for n in indices])
            x0_hw = np.array([center_wl / (2.0 * n) for n in indices])
            
            initial_guesses = [x0_qw, x0_hw]
            
            bounds = [(0, max_layer_nm)] * n_layers
            
            for x0 in initial_guesses:
                if time.time() - start_time > time_limit - 1.0:
                    break
                
                try:
                    res = minimize(_objective, x0, args=(problem_local, wavelengths_nm),
                                   method='L-BFGS-B', bounds=bounds,
                                   options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6})
                    
                    if res.fun < best_score:
                        x_opt = res.x
                        total_thick = np.sum(x_opt)
                        if total_thick <= max_total_nm + 1e-5:
                            best_score = res.fun
                            best_labels = labels.copy()
                            best_thicknesses = x_opt.copy()
                except Exception:
                    pass
    
    if best_labels is None:
        center_wl = 0.5 * (band_nm[0] + band_nm[1])
        return ["L"], [center_wl / (4.0 * materials["L"])]
        
    thicknesses = np.clip(best_thicknesses, 0, max_layer_nm)
    if np.sum(thicknesses) > max_total_nm:
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
        
    return best_labels, thicknesses.tolist()
