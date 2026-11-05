"""Broadband antireflection coating design via coordinate descent + Nelder-Mead."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def _char_matrix(n, d, wl):
    """
    Characteristic matrix for a single lossless layer at normal incidence.
    n: refractive index (real)
    d: physical thickness (nm)
    wl: wavelength (nm)
    Returns 2x2 numpy complex array.
    """
    beta = 2.0 * np.pi * n * d / wl
    cos_b = np.cos(beta)
    sin_b = np.sin(beta)
    # M = [[cos(beta), (1/(i*n))*sin(beta)], [i*n*sin(beta), cos(beta)]]
    # 1/i = -i, so (1/(i*n))*sin(beta) = -1j*sin(beta)/n
    # i*n*sin(beta) = 1j*n*sin(beta)
    return np.array([
        [cos_b, -1j * sin_b / n],
        [1j * n * sin_b, cos_b]
    ], dtype=complex)

def _compute_reflectance(materials, thicknesses, n_ambient, n_substrate, wl):
    """
    Compute reflectance (0 to 1) at a single wavelength using the characteristic matrix method.
    materials: list of 'L' or 'H'
    thicknesses: list of thicknesses in nm
    n_ambient: ambient refractive index
    n_substrate: substrate refractive index
    wl: wavelength in nm
    """
    if len(materials) == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return r * r
    
    # Build total matrix
    M_total = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    
    for mat, d in zip(materials, thicknesses):
        n = problem_cache["materials"][mat]
        M_layer = _char_matrix(n, d, wl)
        M_total = M_total @ M_layer
    
    # Reflectance calculation
    # Y = (M11 * n_sub + M12 * n_sub^2) / (M21 * n_sub + M22 * n_sub^2) -- wait, standard formula:
    # The admittance Y = (M21 + M22 * n_s) / (M11 + M12 * n_s)
    # Actually, let's use the standard characteristic matrix approach:
    # [B C] = M_total
    # Y = (C + B * n_s) / (A + D * n_s) -- no, let's be precise.
    
    # Standard definition:
    # M_total = [[M11, M12], [M21, M22]]
    # The optical admittance at the input plane is:
    # Y = (M21 + M22 * n_sub) / (M11 + M12 * n_sub)
    
    M11 = M_total[0, 0]
    M12 = M_total[0, 1]
    M21 = M_total[1, 0]
    M22 = M_total[1, 1]
    
    Y = (M21 + M22 * n_substrate) / (M11 + M12 * n_substrate)
    
    # Reflectance
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.abs(r) ** 2
    return float(np.real(R))

def _mean_reflectance_percent(materials, thicknesses, problem):
    """
    Compute mean reflectance in percent over the band.
    """
    band = problem["band_nm"]
    n_wl = 401
    wls = np.linspace(band[0], band[1], n_wl)
    
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    
    total_R = 0.0
    for wl in wls:
        R = _compute_reflectance(materials, thicknesses, n_ambient, n_substrate, wl)
        total_R += R
    
    mean_R = total_R / n_wl
    return mean_R * 100.0

def _is_valid(materials, thicknesses, problem):
    """Check if the design is valid."""
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > problem["max_layers"]:
        return False
    if any(t < 0 or t > problem["max_layer_nm"] for t in thicknesses):
        return False
    if sum(thicknesses) > problem["max_total_nm"]:
        return False
    if any(m not in ["L", "H"] for m in materials):
        return False
    return True

def _nm_optimize(materials, thicknesses, problem, deadline):
    """
    Nelder-Mead optimization on the thicknesses, keeping materials fixed.
    """
    x0 = np.array(thicknesses, dtype=float)
    n_params = len(x0)
    
    # Initial simplex
    simplex = [x0.copy()]
    for i in range(n_params):
        x = x0.copy()
        x[i] += 10.0 if x[i] < 10.0 else 10.0
        simplex.append(x)
    
    # Evaluate initial simplex
    f_vals = []
    for x in simplex:
        if time.time() > deadline:
            break
        # Clamp to valid range
        x_clamped = np.clip(x, 0, problem["max_layer_nm"])
        # Check total thickness
        if np.sum(x_clamped) > problem["max_total_nm"]:
            f_vals.append(100.0)
        else:
            f_vals.append(_mean_reflectance_percent(materials, list(x_clamped), problem))
    
    # Nelder-Mead iterations
    max_iter = 200
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Sort simplex by function value
        order = np.argsort(f_vals)
        simplex = [simplex[i] for i in order]
        f_vals = [f_vals[i] for i in order]
        
        # Check convergence
        if abs(f_vals[0] - f_vals[-1]) < 1e-6:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        alpha = 1.0
        x_r = centroid + alpha * (centroid - simplex[-1])
        x_r = np.clip(x_r, 0, problem["max_layer_nm"])
        if np.sum(x_r) > problem["max_total_nm"]:
            f_r = 100.0
        else:
            f_r = _mean_reflectance_percent(materials, list(x_r), problem)
        
        if f_vals[0] <= f_r < f_vals[-2]:
            simplex[-1] = x_r
            f_vals[-1] = f_r
        elif f_r < f_vals[0]:
            # Expansion
            gamma = 2.0
            x_e = centroid + gamma * (x_r - centroid)
            x_e = np.clip(x_e, 0, problem["max_layer_nm"])
            if np.sum(x_e) > problem["max_total_nm"]:
                f_e = 100.0
            else:
                f_e = _mean_reflectance_percent(materials, list(x_e), problem)
            if f_e < f_r:
                simplex[-1] = x_e
                f_vals[-1] = f_e
            else:
                simplex[-1] = x_r
                f_vals[-1] = f_r
        else:
            # Contraction
            rho = 0.5
            x_c = centroid + rho * (simplex[-1] - centroid)
            x_c = np.clip(x_c, 0, problem["max_layer_nm"])
            if np.sum(x_c) > problem["max_total_nm"]:
                f_c = 100.0
            else:
                f_c = _mean_reflectance_percent(materials, list(x_c), problem)
            if f_c < f_vals[-1]:
                simplex[-1] = x_c
                f_vals[-1] = f_c
            else:
                # Shrink
                for i in range(1, n_params + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    x_clamped = np.clip(simplex[i], 0, problem["max_layer_nm"])
                    if np.sum(x_clamped) > problem["max_total_nm"]:
                        f_vals[i] = 100.0
                    else:
                        f_vals[i] = _mean_reflectance_percent(materials, list(x_clamped), problem)
    
    # Return best
    order = np.argsort(f_vals)
    best_idx = order[0]
    best_x = np.clip(simplex[best_idx], 0, problem["max_layer_nm"])
    if np.sum(best_x) > problem["max_total_nm"]:
        return list(thicknesses), _mean_reflectance_percent(materials, thicknesses, problem)
    return list(best_x), f_vals[best_idx]

def design(problem):
    """
    Design a broadband antireflection coating.
    """
    global problem_cache
    problem_cache = problem
    
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    
    # Initial design: 4-layer quarter-wave stack
    centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    
    # Try several initial configurations
    initial_configs = [
        # 4-layer: L H L H
        (["L", "H", "L", "H"], [centre/(4*n_L), centre/(4*n_H), centre/(4*n_L), centre/(4*n_H)]),
        # 6-layer: L H L H L H
        (["L", "H", "L", "H", "L", "H"], [centre/(4*n_L), centre/(4*n_H), centre/(4*n_L), centre/(4*n_H), centre/(4*n_L), centre/(4*n_H)]),
        # 8-layer: L H L H L H L H
        (["L", "H", "L", "H", "L", "H", "L", "H"], [centre/(4*n_L), centre/(4*n_H), centre/(4*n_L), centre/(4*n_H), centre/(4*n_L), centre/(4*n_H), centre/(4*n_L), centre/(4*n_H)]),
    ]
    
    best_materials = None
    best_thicknesses = None
    best_score = float('inf')
    
    for mats, thicks in initial_configs:
        if time.time() > deadline:
            break
        
        # Validate
        if not _is_valid(mats, thicks, problem):
            continue
        
        # Optimize
        opt_thicks, score = _nm_optimize(mats, thicks, problem, deadline)
        
        if score < best_score:
            best_score = score
            best_materials = mats
            best_thicknesses = opt_thicks
    
    if best_materials is None:
        # Fallback to seed
        return ["L"], [centre / (4.0 * n_L)]
    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
