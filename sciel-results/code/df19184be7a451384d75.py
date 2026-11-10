"""Broadband AR coating design using characteristic-matrix method and L-BFGS-B optimization."""
import numpy as np
import time
import sys
from scipy.optimize import minimize

def _characteristic_matrix(n0, ns, materials, thicknesses, lambdas):
    """
    Compute reflectance using the characteristic matrix method.
    n0: ambient index, ns: substrate index
    materials: list of layer indices (same order as thicknesses)
    thicknesses: list of physical thicknesses in nm
    lambdas: array of wavelengths in nm
    Returns: array of reflectance values (0 to 1)
    """
    N = len(thicknesses)
    if N == 0:
        r = (n0 - ns) / (n0 + ns)
        return np.full_like(lambdas, r * r, dtype=float)
    
    # Build total characteristic matrix M = M1 * M2 * ... * MN
    # Mj = [[cos(delta_j), i*sin(delta_j)/nj], [i*nj*sin(delta_j), cos(delta_j)]]
    # delta_j = 2*pi*nj*d_j/lambda
    
    M11 = np.ones_like(lambdas, dtype=complex)
    M12 = np.zeros_like(lambdas, dtype=complex)
    M21 = np.zeros_like(lambdas, dtype=complex)
    M22 = np.ones_like(lambdas, dtype=complex)
    
    for idx in range(N - 1, -1, -1):
        nj = materials[idx]
        dj = thicknesses[idx]
        if dj == 0:
            continue
        delta = 2.0 * np.pi * nj * dj / lambdas
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        # New M = M_old * Mj
        new_M11 = M11 * cos_d + M12 * 1j * nj * sin_d
        new_M12 = M11 * 1j * sin_d / nj + M12 * cos_d
        new_M21 = M21 * cos_d + M22 * 1j * nj * sin_d
        new_M22 = M21 * 1j * sin_d / nj + M22 * cos_d
        
        M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
    
    # Reflectance: R = |Y - n0|^2 / |Y + n0|^2 where Y = M21/M11
    # Avoid division by zero if M11 is zero (unlikely for passive layers)
    Y = M21 / M11
    r = (Y - n0) / (Y + n0)
    R = np.abs(r) ** 2
    return R

def _mean_reflectance_percent(n0, ns, materials_list, thicknesses, lambdas):
    """Compute mean reflectance in percent."""
    R = _characteristic_matrix(n0, ns, materials_list, thicknesses, lambdas)
    return 100.0 * np.mean(R)

def _is_valid(thicknesses, max_layer_nm, max_total_nm, max_layers):
    """Check if thicknesses satisfy constraints."""
    if len(thicknesses) > max_layers:
        return False
    if any(t < -1e-10 or t > max_layer_nm + 1e-10 for t in thicknesses):
        return False
    if sum(thicknesses) > max_total_nm + 1e-10:
        return False
    return True

def design(problem):
    """Design a broadband antireflection coating."""
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin before 150s kill
    
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    band = problem["band_nm"]
    wl_min, wl_max = band[0], band[1]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Wavelengths for evaluation
    lambdas = np.linspace(wl_min, wl_max, 401)
    centre_wl = 0.5 * (wl_min + wl_max)
    
    # Define candidate structures
    # We try alternating L and H layers, starting with L.
    # We vary the number of layers.
    structures = []
    for n_layers in [4, 6, 8, 10, 12, 14, 16, 18, 20]:
        if n_layers > max_layers:
            continue
        materials_list = []
        for i in range(n_layers):
            if i % 2 == 0:
                materials_list.append(nL)
            else:
                materials_list.append(nH)
        structures.append(materials_list)
        
        # Also try starting with H
        materials_list_h = []
        for i in range(n_layers):
            if i % 2 == 0:
                materials_list_h.append(nH)
            else:
                materials_list_h.append(nL)
        structures.append(materials_list_h)
        
        if time.time() > deadline:
            break

    best_score = float('inf')
    best_labels = None
    best_thicknesses = None
    
    # Initial guess: Quarter-wave stack at center wavelength
    # t_L = lambda / (4 * n_L), t_H = lambda / (4 * n_H)
    
    for materials_list in structures:
        if time.time() > deadline:
            break
            
        n_layers = len(materials_list)
        
        # Initial thicknesses
        x0 = []
        for n in materials_list:
            t = centre_wl / (4.0 * n)
            x0.append(t)
        
        # Define objective function
        def objective(x):
            # Clip to bounds just in case, though L-BFGS-B respects bounds
            x_clipped = np.clip(x, 0, max_layer_nm)
            
            # Check total thickness
            total_t = np.sum(x_clipped)
            penalty = 0.0
            if total_t > max_total_nm:
                penalty = 100.0 * (total_t - max_total_nm)
            
            R = _characteristic_matrix(n0, ns, materials_list, x_clipped, lambdas)
            mean_R = 100.0 * np.mean(R)
            
            return mean_R + penalty
        
        # Bounds for each layer
        bounds = [(0, max_layer_nm) for _ in range(n_layers)]
        
        # Run optimization
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8})
            
            if res.success:
                x_opt = res.x
                # Validate
                if _is_valid(x_opt, max_layer_nm, max_total_nm, max_layers):
                    score = objective(x_opt)
                    if score < best_score:
                        best_score = score
                        best_thicknesses = x_opt.copy()
                        # Store material labels
                        labels = []
                        for n in materials_list:
                            if abs(n - nL) < 1e-10:
                                labels.append("L")
                            else:
                                labels.append("H")
                        best_labels = labels
        except Exception:
            pass

    # If no valid result found, try a simple fallback
    if best_labels is None:
        # Try a single layer L
        t_L = centre_wl / (4.0 * nL)
        if t_L <= max_layer_nm and t_L <= max_total_nm:
            best_labels = ["L"]
            best_thicknesses = [t_L]
            best_score = _mean_reflectance_percent(n0, ns, [nL], best_thicknesses, lambdas)
        else:
            # Bare glass? No, must return valid coating.
            # Just return a very thin layer to be valid
            best_labels = ["L"]
            best_thicknesses = [1.0]
            best_score = _mean_reflectance_percent(n0, ns, [nL], best_thicknesses, lambdas)

    # Final check and cleanup
    # Remove zero thickness layers if any (though L-BFGS-B might keep them small)
    final_labels = []
    final_thicknesses = []
    for i in range(len(best_labels)):
        if best_thicknesses[i] > 1e-6:
            final_labels.append(best_labels[i])
            final_thicknesses.append(best_thicknesses[i])
            
    if not final_labels:
        # If all zero, return a minimal valid layer
        final_labels = ["L"]
        final_thicknesses = [1.0]

    # Ensure we don't exceed max_layers
    if len(final_labels) > max_layers:
        final_labels = final_labels[:max_layers]
        final_thicknesses = final_thicknesses[:max_layers]

    # Final validity check
    if not _is_valid(final_thicknesses, max_layer_nm, max_total_nm, max_layers):
        # Fallback to simple quarter-wave
        t_L = centre_wl / (4.0 * nL)
        if t_L > max_layer_nm:
            t_L = max_layer_nm
        if t_L > max_total_nm:
            t_L = max_total_nm
        return ["L"], [t_L]

    return final_labels, final_thicknesses
