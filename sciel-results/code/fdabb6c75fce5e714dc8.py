"""Broadband antireflection coating design via multi-start Nelder-Mead optimization."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize

def _compute_reflectance_vectorized(materials, thicknesses, wavelengths, n_ambient, n_substrate, n_L, n_H):
    """
    Compute reflectance for all wavelengths simultaneously.
    materials: list of 'L' or 'H'
    thicknesses: list of floats (nm)
    wavelengths: np.array of floats (nm)
    Returns: np.array of reflectance values (0 to 1)
    """
    n_layers = len(materials)
    if n_layers == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return np.full_like(wavelengths, r * r)
    
    # Pre-compute indices and thicknesses as arrays
    indices = np.array([n_L if m == 'L' else n_H for m in materials])
    d = np.array(thicknesses)
    
    # M matrix accumulation: M = M1 * M2 * ... * MN
    # M is 2x2 complex matrix
    # We can vectorize the matrix multiplication across wavelengths
    # M_total = Identity
    M11 = np.ones_like(wavelengths, dtype=complex)
    M12 = np.zeros_like(wavelengths, dtype=complex)
    M21 = np.zeros_like(wavelengths, dtype=complex)
    M22 = np.ones_like(wavelengths, dtype=complex)
    
    for i in range(n_layers):
        n_i = indices[i]
        d_i = d[i]
        
        # delta = 2 * pi * n * d / lambda
        delta = 2.0 * np.pi * n_i * d_i / wavelengths
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        # Layer matrix elements
        # L11 = cos_d
        # L12 = 1j * sin_d / n_i
        # L21 = 1j * n_i * sin_d
        # L22 = cos_d
        
        # M_new = M_old * L
        # M11_new = M11*L11 + M12*L21
        # M12_new = M11*L12 + M12*L22
        # M21_new = M21*L11 + M22*L21
        # M22_new = M21*L12 + M22*L22
        
        L12 = 1j * sin_d / n_i
        L21 = 1j * n_i * sin_d
        
        new_M11 = M11 * cos_d + M12 * L21
        new_M12 = M11 * L12 + M12 * cos_d
        new_M21 = M21 * cos_d + M22 * L21
        new_M22 = M21 * L12 + M22 * cos_d
        
        M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
        
    # Calculate Admittance Y
    # B = M11 + M12 * n_substrate
    # C = M21 + M22 * n_substrate
    # Y = C / B
    B = M11 + M12 * n_substrate
    C = M21 + M22 * n_substrate
    Y = C / B
    
    # Reflection coefficient r
    r = (n_ambient - Y) / (n_ambient + Y)
    
    return np.abs(r) ** 2

def _objective_function(thicknesses, materials, problem, wavelengths, n_ambient, n_substrate, n_L, n_H):
    """
    Objective function for scipy.optimize.minimize.
    thicknesses: np.array
    Returns: mean reflectance in percent
    """
    # Check constraints
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    # If any thickness is out of bounds, return a large penalty
    # We assume the optimizer stays within bounds mostly, but we clamp just in case for evaluation
    # However, for optimization, it's better to let it explore and penalize heavily if invalid
    # Actually, Nelder-Mead can step outside. We should return a high value if invalid.
    if np.any(thicknesses < 0) or np.any(thicknesses > max_layer_nm):
        return 1000.0
    if np.sum(thicknesses) > max_total_nm:
        return 1000.0
    
    R = _compute_reflectance_vectorized(materials, thicknesses, wavelengths, n_ambient, n_substrate, n_L, n_H)
    return 100.0 * np.mean(R)

def design(problem):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wavelengths = np.linspace(band[0], band[1], 401)
    
    # Candidate structures
    # We will try different numbers of layers and alternating patterns
    # Common good starting points: LHLH, HHLH, etc.
    # Let's try N = 4, 6, 8, 10, 12, 14 layers
    # For each N, we try a few material patterns
    
    best_score = float('inf')
    best_materials = []
    best_thicknesses = []
    
    # Helper to check time
    def time_left():
        return deadline - time.time()
    
    # Generate candidate patterns
    candidates = []
    for N in [4, 6, 8, 10, 12, 14, 16]:
        if N > max_layers:
            continue
        
        # Pattern 1: LHLH...
        mats1 = ['L' if i % 2 == 0 else 'H' for i in range(N)]
        # Pattern 2: HHLH... (Start with H, double H)
        # A common AR stack is L H L H ... but sometimes H L H L ...
        # Let's try alternating starting with L and starting with H
        mats2 = ['H' if i % 2 == 0 else 'L' for i in range(N)]
        
        candidates.append(mats1)
        candidates.append(mats2)
        
        # If N is even, also try a "quasi-quarter wave" mix
        # But let's keep it simple first: just alternating.
        
    # To avoid redundancy, we can unique them, but they are distinct lists.
    
    for mats in candidates:
        if time_left() < 5.0:
            break
            
        N = len(mats)
        
        # Initial guess for thicknesses
        # Quarter wave at center wavelength
        center_wl = 0.5 * (band[0] + band[1])
        init_thick = []
        for m in mats:
            n = n_L if m == 'L' else n_H
            d = center_wl / (4.0 * n)
            # Ensure it fits within max_layer_nm
            d = min(d, max_layer_nm)
            init_thick.append(d)
        
        # Check if initial guess is valid
        if sum(init_thick) > max_total_nm:
            # Scale down
            scale = max_total_nm / sum(init_thick)
            init_thick = [t * scale for t in init_thick]
            
        # Run Nelder-Mead
        # We need to be careful with the number of iterations to not timeout
        # maxiter = 200 is usually enough for N-M on this scale
        # We'll run multiple restarts with slightly different initializations
        
        best_local_score = float('inf')
        best_local_thick = None
        
        # Restart 1: Standard quarter-wave
        x0 = np.array(init_thick)
        
        try:
            res = minimize(
                _objective_function,
                x0,
                args=(mats, problem, wavelengths, n_ambient, n_substrate, n_L, n_H),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True}
            )
            if res.fun < best_local_score:
                best_local_score = res.fun
                best_local_thick = res.x
        except Exception:
            pass
        
        # Restart 2: Random perturbation
        if time_left() > 10.0:
            np.random.seed(42) # Fixed seed for reproducibility in this run
            x0_perturbed = x0 * (1.0 + 0.1 * np.random.randn(len(x0)))
            # Clamp to valid range
            x0_perturbed = np.clip(x0_perturbed, 0, max_layer_nm)
            
            # Check total thickness
            if np.sum(x0_perturbed) > max_total_nm:
                x0_perturbed *= (max_total_nm / np.sum(x0_perturbed))
                
            try:
                res = minimize(
                    _objective_function,
                    x0_perturbed,
                    args=(mats, problem, wavelengths, n_ambient, n_substrate, n_L, n_H),
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True}
                )
                if res.fun < best_local_score:
                    best_local_score = res.fun
                    best_local_thick = res.x
            except Exception:
                pass
        
        # Restart 3: Different center wavelength bias
        if time_left() > 10.0:
            # Try optimizing for slightly shorter/longer center
            for shift in [-0.1, 0.1]:
                center_wl_shifted = center_wl * (1.0 + shift)
                init_thick_shifted = []
                for m in mats:
                    n = n_L if m == 'L' else n_H
                    d = center_wl_shifted / (4.0 * n)
                    d = min(d, max_layer_nm)
                    init_thick_shifted.append(d)
                
                if sum(init_thick_shifted) > max_total_nm:
                    scale = max_total_nm / sum(init_thick_shifted)
                    init_thick_shifted = [t * scale for t in init_thick_shifted]
                    
                x0_shifted = np.array(init_thick_shifted)
                try:
                    res = minimize(
                        _objective_function,
                        x0_shifted,
                        args=(mats, problem, wavelengths, n_ambient, n_substrate, n_L, n_H),
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True}
                    )
                    if res.fun < best_local_score:
                        best_local_score = res.fun
                        best_local_thick = res.x
                except Exception:
                    pass
        
        if best_local_thick is not None and best_local_score < best_score:
            # Final validation
            final_score = _objective_function(best_local_thick, mats, problem, wavelengths, n_ambient, n_substrate, n_L, n_H)
            if final_score < best_score:
                best_score = final_score
                best_materials = mats
                best_thicknesses = list(best_local_thick)
                
    # If no valid result found, return a simple valid design
    if not best_materials:
        best_materials = ['L']
        center_wl = 0.5 * (band[0] + band[1])
        best_thicknesses = [min(center_wl / (4.0 * n_L), max_layer_nm)]
        
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
