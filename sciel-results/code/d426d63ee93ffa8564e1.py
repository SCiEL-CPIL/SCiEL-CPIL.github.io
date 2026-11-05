"""Broadband antireflection coating design via Multi-Start Nelder-Mead with Structured Initialization."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _vectorized_reflectance(thicknesses, materials, n_ambient, n_substrate, n_L, n_H, wls):
    """
    Compute mean reflectance fraction for a single design across all wavelengths.
    Uses vectorized operations over the wavelength axis.
    
    thicknesses: array of physical thicknesses in nm
    materials: array of 'L' or 'H' strings
    wls: array of wavelengths in nm
    """
    n_layers = len(thicknesses)
    if n_layers == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return float(r * r)
    
    # Pre-compute refractive indices for each layer
    n_layers_arr = np.array([n_H if m == 'H' else n_L for m in materials])
    
    # M = Identity
    M11 = np.ones_like(wls, dtype=complex)
    M12 = np.zeros_like(wls, dtype=complex)
    M21 = np.zeros_like(wls, dtype=complex)
    M22 = np.ones_like(wls, dtype=complex)
    
    for i in range(n_layers):
        n = n_layers_arr[i]
        d = thicknesses[i]
        beta = 2.0 * np.pi * n * d / wls
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        L11 = cos_b
        L12 = -1j * sin_b / n
        L21 = 1j * n * sin_b
        L22 = cos_b
        
        # M = M @ L
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
    
    # Admittance: Y = (M11 * n_substrate + M12) / (M21 * n_substrate + M22)
    Y = (M11 * n_substrate + M12) / (M21 * n_substrate + M22)
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.real(r * r.conjugate())
    
    return float(np.mean(R))

def design(problem):
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    wl_min, wl_max = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # 401 equally spaced wavelengths
    wls = np.linspace(wl_min, wl_max, 401)
    wl_center = 0.5 * (wl_min + wl_max)
    
    # Fallback solution: Single quarter-wave L layer
    qw_L = wl_center / (4.0 * n_L)
    qw_H = wl_center / (4.0 * n_H)
    
    best_score = float('inf')
    best_thicknesses = [qw_L]
    best_materials = ['L']
    
    # Helper to evaluate and update best
    def update_best(materials, thicknesses):
        nonlocal best_score, best_thicknesses, best_materials
        if len(thicknesses) == 0:
            return
        if any(t < -1e-6 or t > max_layer_nm + 1e-6 for t in thicknesses):
            return
        if sum(thicknesses) > max_total_nm + 1e-6:
            return
        
        t_calc = np.clip(thicknesses, 0, max_layer_nm)
        R_frac = _vectorized_reflectance(t_calc, materials, n_ambient, n_substrate, n_L, n_H, wls)
        score = R_frac * 100.0
        
        if score < best_score:
            best_score = score
            best_thicknesses = [max(0.0, t) for t in thicknesses]
            best_materials = materials[:]

    # Update with fallback
    update_best(best_materials, best_thicknesses)
    
    # Generate diverse initial structures (motifs)
    # 1. Single layer L
    # 2. Single layer H
    # 3. Two layers H-L (standard AR)
    # 4. Two layers L-H
    # 5. Three layers H-L-H
    # 6. Three layers L-H-L
    # 7. Four layers H-L-H-L
    # 8. Five layers H-L-H-L-H
    # 9. Random 3-5 layer structures
    
    initial_structures = []
    
    # 1
    initial_structures.append((['L'], [qw_L]))
    # 2
    initial_structures.append((['H'], [qw_H]))
    # 3
    initial_structures.append((['H', 'L'], [qw_H, qw_L]))
    # 4
    initial_structures.append((['L', 'H'], [qw_L, qw_H]))
    # 5
    initial_structures.append((['H', 'L', 'H'], [qw_H, qw_L, qw_H]))
    # 6
    initial_structures.append((['L', 'H', 'L'], [qw_L, qw_H, qw_L]))
    # 7
    initial_structures.append((['H', 'L', 'H', 'L'], [qw_H, qw_L, qw_H, qw_L]))
    # 8
    initial_structures.append((['H', 'L', 'H', 'L', 'H'], [qw_H, qw_L, qw_H, qw_L, qw_H]))
    
    # Add some random structures
    np.random.seed(42)
    for _ in range(10):
        n_layers = np.random.randint(3, 6)
        mats = []
        thicks = []
        for i in range(n_layers):
            if i % 2 == 0:
                mats.append('H')
                thicks.append(np.random.uniform(0.5 * qw_H, 1.5 * qw_H))
            else:
                mats.append('L')
                thicks.append(np.random.uniform(0.5 * qw_L, 1.5 * qw_L))
        initial_structures.append((mats, thicks))
    
    # Optimize each structure using Nelder-Mead
    for mats, thicks in initial_structures:
        if time.time() > deadline:
            break
            
        n_layers = len(mats)
        if n_layers == 0 or n_layers > max_layers:
            continue
            
        # Define objective function for Nelder-Mead
        def objective(x):
            # x is an array of thicknesses
            if np.any(x < 0) or np.any(x > max_layer_nm):
                return 100.0 # Large penalty
            if np.sum(x) > max_total_nm:
                return 100.0 # Large penalty
            
            R_frac = _vectorized_reflectance(x, mats, n_ambient, n_substrate, n_L, n_H, wls)
            return R_frac * 100.0
        
        # Initial guess
        x0 = np.array(thicks)
        
        # Bounds are handled by penalty in objective, but we can also use bounds in some optimizers.
        # Nelder-Mead doesn't support bounds directly in scipy, so we rely on penalty or clipping.
        # To be safe, we clip inside the objective if needed, but penalty is faster.
        
        try:
            # Run Nelder-Mead
            # maxiter is set to ensure we don't hang
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
            
            if res.success or res.fun < best_score:
                # Check if result is valid
                x_opt = res.x
                if np.all(x_opt >= 0) and np.all(x_opt <= max_layer_nm) and np.sum(x_opt) <= max_total_nm:
                    update_best(mats, x_opt.tolist())
                    
        except Exception:
            pass
            
        # Also try a local search with slightly different initial points around the best found so far
        # This helps if the first NM run got stuck.
        if best_score < 1.0: # If we have a good result, refine it further
             # Skip for now to save time, NM is usually sufficient for these small dimensions.
             pass

    # Final check
    if not best_thicknesses or any(t < 0 or t > max_layer_nm for t in best_thicknesses) or sum(best_thicknesses) > max_total_nm:
        return ["L"], [qw_L]
        
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
