import time
import math
import numpy as np
import random

def _reflectance_array(materials, thicknesses, n_ambient, n_substrate, wls, n_mat):
    """
    Compute reflectance array for given materials and thicknesses.
    materials: list of 'L' or 'H'
    thicknesses: list of floats (nm)
    n_ambient: float
    n_substrate: float
    wls: array of wavelengths (nm)
    n_mat: dict {'L': n_L, 'H': n_H}
    Returns: array of reflectance values (0 to 1)
    """
    if len(materials) == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return np.full(len(wls), r * r)

    n_layers = len(materials)
    n_wls = len(wls)
    
    # Precompute beta = 2*pi*n*d/wl for all layers and wavelengths
    # beta shape: (n_layers, n_wls)
    beta = np.zeros((n_layers, n_wls))
    for i, m in enumerate(materials):
        n = n_mat[m]
        d = thicknesses[i]
        beta[i, :] = 2.0 * math.pi * n * d / wls

    cosb = np.cos(beta)
    sinb = np.sin(beta)

    # Characteristic matrix M = M1 * M2 * ... * M_N
    # M_i = [[cos(beta_i), (i/n_i)sin(beta_i)], [i*n_i*sin(beta_i), cos(beta_i)]]
    # We are working with complex numbers.
    
    M11 = np.ones(n_wls, dtype=complex)
    M12 = np.zeros(n_wls, dtype=complex)
    M21 = np.zeros(n_wls, dtype=complex)
    M22 = np.ones(n_wls, dtype=complex)

    for i, m in enumerate(materials):
        n = n_mat[m]
        c = cosb[i]
        s = sinb[i]
        
        # Matrix elements for layer i
        A = c
        B = 1j * s / n
        C = 1j * n * s
        D = c

        # Matrix multiplication: M_new = M_old * M_i
        # M_old is [[M11, M12], [M21, M22]]
        # M_i is [[A, B], [C, D]]
        
        nM11 = M11 * A + M12 * C
        nM12 = M11 * B + M12 * D
        nM21 = M21 * A + M22 * C
        nM22 = M21 * B + M22 * D
        
        M11, M12, M21, M22 = nM11, nM12, nM21, nM22

    # Admittance Y
    # Y = (M21 + n_substrate * M22) / (M11 + n_substrate * M12)
    Y = (M21 + n_substrate * M22) / (M11 + n_substrate * M12)
    
    # Reflectance r
    # r = (n_ambient - Y) / (n_ambient + Y)
    r = (n_ambient - Y) / (n_ambient + Y)
    
    # Reflectance R = |r|^2
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
        # Penalty proportional to violation
        return 1e6 + (total_th - max_total) * 10.0

    if len(materials) == 0:
        r = (problem["ambient_index"] - problem["substrate_index"]) / (
            problem["ambient_index"] + problem["substrate_index"])
        return r * r * 100.0

    R = _reflectance_array(materials, thicknesses, problem["ambient_index"],
                           problem["substrate_index"], wls, n_mat)
    return float(np.mean(R)) * 100.0

def _nelder_mead(objective, x0, bounds, deadline, max_iter=100, tol=1e-4):
    """
    Nelder-Mead optimization with bounds handling.
    Reduced iterations and looser convergence for speed.
    x0: initial point
    bounds: tuple (lo, hi) arrays
    """
    x0 = np.array(x0, dtype=float)
    n = len(x0)
    lo, hi = bounds
    
    # Initialize simplex
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        # Step size: 10% of range or 1.0 if range is small
        range_i = hi[i] - lo[i]
        step = 0.1 * range_i if range_i > 1e-6 else 1.0
        xi[i] += step
        simplex.append(np.clip(xi, lo, hi))
        
    f_vals = [objective(x) for x in simplex]
    
    # Alpha, Gamma, Rho, Sigma
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
            
        # Sort simplex
        order = np.argsort(f_vals)
        simplex = [simplex[i] for i in order]
        f_vals = [f_vals[i] for i in order]
        
        # Check convergence
        if abs(f_vals[0] - f_vals[-1]) < tol:
            break
        
        # Centroid of best n points
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = np.clip(centroid + alpha * (centroid - simplex[-1]), lo, hi)
        f_r = objective(x_r)
        
        if f_vals[0] <= f_r < f_vals[-2]:
            simplex[-1] = x_r
            f_vals[-1] = f_r
            continue
            
        # Expansion
        if f_r < f_vals[0]:
            x_e = np.clip(centroid + gamma * (x_r - centroid), lo, hi)
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_vals[-1] = f_e
            else:
                simplex[-1] = x_r
                f_vals[-1] = f_r
            continue
            
        # Contraction
        x_c = np.clip(centroid + rho * (simplex[-1] - centroid), lo, hi)
        f_c = objective(x_c)
        if f_c < f_vals[-1]:
            simplex[-1] = x_c
            f_vals[-1] = f_c
            continue
            
        # Shrink
        for i in range(1, n + 1):
            simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), lo, hi)
            f_vals[i] = objective(simplex[i])
            
    order = np.argsort(f_vals)
    best_idx = order[0]
    return np.clip(simplex[best_idx], lo, hi), f_vals[best_idx]

def _coordinate_descent(objective, x0, bounds, deadline, max_iter=50, tol=1e-4):
    """
    Coordinate descent with golden section search along each dimension.
    Reduced iterations and tighter convergence for speed.
    """
    x = np.array(x0, dtype=float)
    lo, hi = bounds
    n = len(x)
    f_best = objective(x)
    
    # Golden ratio
    gr = (1 + np.sqrt(5)) / 2.0
    inv_gr = 1.0 / gr
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            
            # Determine bounds for coordinate i
            lo_i = lo[i]
            hi_i = hi[i]
            
            # If current x[i] is at boundary, try expanding
            if x[i] <= lo_i + 1e-9:
                # Try moving up
                new_hi = min(x[i] + 30.0, hi_i)
                if new_hi <= x[i] + 1e-6:
                    continue
                a, b = x[i], new_hi
            elif x[i] >= hi_i - 1e-9:
                # Try moving down
                new_lo = max(x[i] - 30.0, lo_i)
                if new_lo >= x[i] - 1e-6:
                    continue
                a, b = new_lo, x[i]
            else:
                # Search around current position
                delta = min(30.0, (hi_i - lo_i) * 0.1)
                a = max(x[i] - delta, lo_i)
                b = min(x[i] + delta, hi_i)
                if b - a < 1e-6:
                    continue
            
            # Golden section search - reduced iterations
            c = b - inv_gr * (b - a)
            d = a + inv_gr * (b - a)
            
            x_test_c = x.copy()
            x_test_c[i] = c
            f_c = objective(x_test_c)
            
            x_test_d = x.copy()
            x_test_d[i] = d
            f_d = objective(x_test_d)
            
            for _ in range(10):  # Reduced from 20
                if time.time() > deadline:
                    break
                if f_c < f_d:
                    b = d
                    d = c
                    f_d = f_c
                    c = b - inv_gr * (b - a)
                    x_test_c = x.copy()
                    x_test_c[i] = c
                    f_c = objective(x_test_c)
                else:
                    a = c
                    c = d
                    f_c = f_d
                    d = a + inv_gr * (b - a)
                    x_test_d = x.copy()
                    x_test_d[i] = d
                    f_d = objective(x_test_d)
            
            # Best point in this coordinate search
            x_opt_i = x.copy()
            if f_c < f_d:
                x_opt_i[i] = c
                f_new = f_c
            else:
                x_opt_i[i] = d
                f_new = f_d
            
            if f_new < f_best - tol:
                x = x_opt_i
                f_best = f_new
                improved = True
        
        if not improved:
            break
    
    return x, f_best

def design(problem):
    """
    Design a broadband antireflection coating.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin: 150s limit - 40s buffer
    
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
    
    # Define a small set of physically plausible material sequences
    # Focus on alternating L/H which is known to work well for AR coatings
    sequences = []
    
    # Length 2: basic AR coating
    sequences.append(['L', 'H'])
    sequences.append(['H', 'L'])
    
    # Length 4: improved broadband
    sequences.append(['L', 'H', 'L', 'H'])
    sequences.append(['H', 'L', 'H', 'L'])
    
    # Length 6: even better broadband
    sequences.append(['L', 'H', 'L', 'H', 'L', 'H'])
    sequences.append(['H', 'L', 'H', 'L', 'H', 'L'])
    
    # Filter by max_layers
    sequences = [seq for seq in sequences if len(seq) <= max_layers]
    
    # For each sequence, try a few initial guesses
    # Use quarter-wave at central wavelength as starting point
    center_wl = (band_start + band_end) / 2.0  # 600 nm
    
    for mats in sequences:
        if time.time() > deadline:
            break
            
        n_layers = len(mats)
        lo = np.zeros(n_layers)
        hi = np.full(n_layers, max_layer)
        
        # Generate 2-3 initial guesses
        initial_guesses = []
        
        # Quarter-wave at center wavelength
        ths_qw = []
        for m in mats:
            n = n_mat[m]
            q = center_wl / (4.0 * n)
            ths_qw.append(min(q, max_layer))
        initial_guesses.append(ths_qw)
        
        # Quarter-wave with slight variation
        ths_var = []
        for m in mats:
            n = n_mat[m]
            q = center_wl / (4.0 * n) * random.uniform(0.9, 1.1)
            ths_var.append(min(q, max_layer))
        initial_guesses.append(ths_var)
        
        # Evaluate and optimize each guess
        for x0 in initial_guesses:
            if time.time() > deadline:
                break
                
            # Clip and scale to meet constraints
            x0 = np.clip(x0, lo, hi)
            s = sum(x0)
            if s > max_total:
                x0 = x0 * (max_total / s)
                x0 = np.clip(x0, 0.0, max_layer)
            
            # Define objective function
            def objective(th, m=mats):
                return _mean_reflectance_percent(m, list(th), problem, wls, n_mat)
            
            # Run coordinate descent with reduced iterations
            x_cd, f_cd = _coordinate_descent(objective, x0, (lo, hi), deadline, max_iter=50)
            
            # Check time before Nelder-Mead
            if time.time() > deadline:
                x_opt = x_cd
                f_opt = f_cd
            else:
                # Run Nelder-Mead with reduced iterations
                x_opt, f_opt = _nelder_mead(objective, x_cd, (lo, hi), deadline, max_iter=100)
            
            # Post-process: ensure constraints are strictly met
            x_opt = np.clip(x_opt, 0.0, max_layer)
            s = sum(x_opt)
            if s > max_total:
                x_opt = x_opt * (max_total / s)
                x_opt = np.clip(x_opt, 0.0, max_layer)
                
            # Calculate final objective value
            val = _mean_reflectance_percent(mats, list(x_opt), problem, wls, n_mat)
            
            if val < best_val:
                best_val = val
                best_mats = mats
                best_ths = list(x_opt)
                
    # Fallback: simple quarter-wave L layer
    if best_mats is None:
        best_mats = ['L']
        best_ths = [min(center_wl / (4.0 * n_L), max_layer)]
        best_val = _mean_reflectance_percent(best_mats, best_ths, problem, wls, n_mat)
        
    # Final safety check
    best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
    if sum(best_ths) > max_total + 1e-6:
        f = max_total / sum(best_ths)
        best_ths = [t * f for t in best_ths]
        
    return list(best_mats), list(best_ths)
