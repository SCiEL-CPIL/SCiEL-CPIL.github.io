import time
import math
import numpy as np
import random
import itertools

def _reflectance_array(materials, thicknesses, n_ambient, n_substrate, wls, n_mat):
    """
    Compute reflectance array for given materials and thicknesses using
    the thin-film characteristic-matrix method.
    Returns array of reflectance values (0 to 1).
    """
    if len(materials) == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return np.full(len(wls), r * r)

    n_layers = len(materials)
    n_wls = len(wls)
    
    # Precompute beta = 2*pi*n*d/wl for all layers and wavelengths
    # Vectorized calculation
    n_arr = np.array([n_mat[m] for m in materials])
    d_arr = np.array(thicknesses)
    
    # beta shape: (n_layers, n_wls)
    beta = 2.0 * math.pi * np.outer(n_arr * d_arr, 1.0 / wls)

    cosb = np.cos(beta)
    sinb = np.sin(beta)

    # Build the total characteristic matrix
    # M = M_1 * M_2 * ... * M_N
    # M_i = [[cos(beta_i), i*sin(beta_i)/n_i], [i*n_i*sin(beta_i), cos(beta_i)]]
    
    # Initialize M as identity
    M11 = np.ones(n_wls, dtype=complex)
    M12 = np.zeros(n_wls, dtype=complex)
    M21 = np.zeros(n_wls, dtype=complex)
    M22 = np.ones(n_wls, dtype=complex)

    # Vectorized matrix multiplication
    # We can precompute A, B, C, D for all layers
    # A_i = cos(beta_i)
    # B_i = i*sin(beta_i)/n_i
    # C_i = i*n_i*sin(beta_i)
    # D_i = cos(beta_i)
    
    A = cosb
    B = 1j * sinb / n_arr[:, np.newaxis]
    C = 1j * n_arr[:, np.newaxis] * sinb
    D = cosb

    for i in range(n_layers):
        Ai = A[i]
        Bi = B[i]
        Ci = C[i]
        Di = D[i]
        
        # M_new = M_old * M_i
        nM11 = M11 * Ai + M12 * Ci
        nM12 = M11 * Bi + M12 * Di
        nM21 = M21 * Ai + M22 * Ci
        nM22 = M21 * Bi + M22 * Di
        
        M11, M12, M21, M22 = nM11, nM12, nM21, nM22

    Y = (M21 + n_substrate * M22) / (M11 + n_substrate * M12)
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.abs(r) ** 2
    return R

def _mean_reflectance_percent(materials, thicknesses, problem, wls, n_mat):
    """
    Compute mean reflectance in percent.
    Returns penalty if constraints are violated.
    """
    if len(materials) != len(thicknesses):
        return 1e6
    if len(materials) > problem["max_layers"]:
        return 1e6
        
    max_layer = problem["max_layer_nm"]
    max_total = problem["max_total_nm"]
    
    for t in thicknesses:
        if t < -1e-9 or t > max_layer + 1e-9:
            return 1e6
            
    total_th = sum(thicknesses)
    if total_th > max_total + 1e-6:
        return 1e6 + (total_th - max_total) * 10.0

    if len(materials) == 0:
        r = (problem["ambient_index"] - problem["substrate_index"]) / (
            problem["ambient_index"] + problem["substrate_index"])
        return r * r * 100.0

    R = _reflectance_array(materials, thicknesses, problem["ambient_index"],
                           problem["substrate_index"], wls, n_mat)
    return float(np.mean(R)) * 100.0

def _project_to_constraints(th, max_layer, max_total):
    """Project thicknesses onto the constraint box [0, max_layer] and sum <= max_total."""
    th = np.clip(th, 0.0, max_layer)
    s = np.sum(th)
    if s > max_total:
        th = th * (max_total / s)
        th = np.clip(th, 0.0, max_layer)
    return th

def _nelder_mead(objective, x0, bounds, deadline, max_iter=500, tol=1e-8):
    """
    Nelder-Mead optimization with bounds handling.
    """
    x0 = np.array(x0, dtype=float)
    n = len(x0)
    lo, hi = bounds
    
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        range_i = hi[i] - lo[i]
        step = 0.1 * range_i if range_i > 1e-6 else 1.0
        xi[i] += step
        simplex.append(np.clip(xi, lo, hi))
        
    f_vals = [objective(x) for x in simplex]
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    best_f = min(f_vals)
    stagnant_count = 0
    
    for iter_count in range(max_iter):
        if time.time() > deadline:
            break
            
        order = np.argsort(f_vals)
        simplex = [simplex[i] for i in order]
        f_vals = [f_vals[i] for i in order]
        
        spread = abs(f_vals[0] - f_vals[-1])
        if spread < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        x_r = np.clip(centroid + alpha * (centroid - simplex[-1]), lo, hi)
        f_r = objective(x_r)
        
        if f_vals[0] <= f_r < f_vals[-2]:
            simplex[-1] = x_r
            f_vals[-1] = f_r
            stagnant_count = 0
            continue
            
        if f_r < f_vals[0]:
            x_e = np.clip(centroid + gamma * (x_r - centroid), lo, hi)
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_vals[-1] = f_e
            else:
                simplex[-1] = x_r
                f_vals[-1] = f_r
            stagnant_count = 0
            continue
            
        x_c = np.clip(centroid + rho * (simplex[-1] - centroid), lo, hi)
        f_c = objective(x_c)
        if f_c < f_vals[-1]:
            simplex[-1] = x_c
            f_vals[-1] = f_c
            stagnant_count = 0
            continue
            
        for i in range(1, n + 1):
            simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), lo, hi)
            f_vals[i] = objective(simplex[i])
            
        if min(f_vals) > best_f - 1e-10:
            stagnant_count += 1
        else:
            stagnant_count = 0
            best_f = min(f_vals)
            
        if stagnant_count > 15:
            # Restart from best point with larger step
            x_best = simplex[0].copy()
            simplex = [x_best.copy()]
            for i in range(n):
                xi = x_best.copy()
                range_i = hi[i] - lo[i]
                step = 0.2 * range_i if range_i > 1e-6 else 2.0
                xi[i] += step
                simplex.append(np.clip(xi, lo, hi))
            f_vals = [objective(x) for x in simplex]
            stagnant_count = 0
            
    order = np.argsort(f_vals)
    best_idx = order[0]
    return np.clip(simplex[best_idx], lo, hi), f_vals[best_idx]

def design(problem):
    """
    Design a broadband antireflection coating.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    band_start, band_end = problem["band_nm"]
    wls = np.linspace(band_start, band_end, 401)
    n_mat = problem["materials"]
    
    best_val = float('inf')
    best_mats = None
    best_ths = None

    # Generate candidate material sequences
    # Prioritize alternating sequences but include others
    sequences = []
    
    # Alternating sequences
    for length in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
        if length > max_layers:
            break
        # Start with L
        seq_l = ['L' if i % 2 == 0 else 'H' for i in range(length)]
        sequences.append(seq_l)
        # Start with H
        seq_h = ['H' if i % 2 == 0 else 'L' for i in range(length)]
        sequences.append(seq_h)
        
    # Add some non-alternating sequences for diversity
    # e.g. LLHH, HHLL, LHLHHL, etc.
    for length in [4, 6, 8]:
        if length > max_layers:
            break
        # Generate a few random sequences of this length
        for _ in range(5):
            seq = [random.choice(['L', 'H']) for _ in range(length)]
            if seq not in sequences:
                sequences.append(seq)
                
    sequences = [seq for seq in sequences if len(seq) <= max_layers]
    
    # Optimization loop
    for mats in sequences:
        if time.time() > deadline - 5.0:
            break
            
        n_layers = len(mats)
        lo = np.zeros(n_layers)
        hi = np.full(n_layers, max_layer)
        
        # Generate initial guesses
        initial_guesses = []
        
        # Quarter-wave stack based on center wavelength
        center_wls = [500, 550, 600, 450, 650]
        for cw in center_wls:
            ths = []
            for m in mats:
                n = n_mat[m]
                q = cw / (4.0 * n)
                ths.append(q)
            initial_guesses.append(np.array(ths))
            
        # Random guesses
        for _ in range(3):
            ths = [random.uniform(50, 200) for _ in range(n_layers)]
            initial_guesses.append(np.array(ths))
            
        # Run Nelder-Mead for each initial guess
        for x0 in initial_guesses:
            if time.time() > deadline - 5.0:
                break
                
            x0 = _project_to_constraints(x0, max_layer, max_total)
            
            def objective(th, m=mats):
                return _mean_reflectance_percent(m, list(th), problem, wls, n_mat)
            
            x_opt, f_opt = _nelder_mead(objective, x0, (lo, hi), deadline, max_iter=300)
            
            x_opt = _project_to_constraints(x_opt, max_layer, max_total)
            val = _mean_reflectance_percent(mats, list(x_opt), problem, wls, n_mat)
            
            if val < best_val:
                best_val = val
                best_mats = mats
                best_ths = list(x_opt)
                
    # Final Polish using Nelder-Mead for higher precision
    if best_mats is not None and time.time() < deadline - 5.0:
        n_layers = len(best_mats)
        lo = np.zeros(n_layers)
        hi = np.full(n_layers, max_layer)
        
        def objective(th, m=best_mats):
            return _mean_reflectance_percent(m, list(th), problem, wls, n_mat)
        
        # Use Nelder-Mead with tighter tolerance for local polish
        try:
            x_polish, f_polish = _nelder_mead(objective, np.array(best_ths), (lo, hi), deadline, max_iter=200, tol=1e-10)
            x_polish = _project_to_constraints(x_polish, max_layer, max_total)
            val_polish = _mean_reflectance_percent(best_mats, list(x_polish), problem, wls, n_mat)
            if val_polish < best_val:
                best_val = val_polish
                best_ths = list(x_polish)
        except Exception:
            pass

    if best_mats is None:
        # Fallback: single quarter-wave layer
        best_mats = ['L']
        best_ths = [600.0 / (4.0 * n_L)]
        best_val = _mean_reflectance_percent(best_mats, best_ths, problem, wls, n_mat)
        
    # Final constraint check
    best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
    if sum(best_ths) > max_total + 1e-6:
        f = max_total / sum(best_ths)
        best_ths = [t * f for t in best_ths]
        
    return list(best_mats), list(best_ths)
