import time
import math
import numpy as np
import random

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

    # Vectorized beta computation
    ns = np.array([n_mat[m] for m in materials])
    ds = np.array(thicknesses, dtype=float)
    beta = 2.0 * math.pi * np.outer(ns * ds, 1.0 / wls)

    cosb = np.cos(beta)
    sinb = np.sin(beta)

    # Build the total characteristic matrix
    M11 = np.ones(n_wls, dtype=complex)
    M12 = np.zeros(n_wls, dtype=complex)
    M21 = np.zeros(n_wls, dtype=complex)
    M22 = np.ones(n_wls, dtype=complex)

    for i in range(n_layers):
        n = ns[i]
        c = cosb[i]
        s = sinb[i]

        A = c
        B = 1j * s / n
        C = 1j * n * s
        D = c

        nM11 = M11 * A + M12 * C
        nM12 = M11 * B + M12 * D
        nM21 = M21 * A + M22 * C
        nM22 = M21 * B + M22 * D

        M11, M12, M21, M22 = nM11, nM12, nM21, nM22

    Y = (M21 + n_substrate * M22) / (M11 + n_substrate * M12)
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.abs(r) ** 2
    return R

def _mean_reflectance_percent(materials, thicknesses, problem, wls, n_mat):
    """
    Compute mean reflectance in percent with constraint penalties.
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

def _project_to_feasible(x, lo, hi, max_total):
    """
    Project x to feasible region: bounds and total thickness constraint.
    """
    x = np.clip(x, lo, hi)
    s = np.sum(x)
    if s > max_total:
        if s > 1e-9:
            x = x * (max_total / s)
        else:
            x = np.zeros_like(x)
    x = np.clip(x, lo, hi)
    return x

def _optimize_nelder_mead(objective, x0, lo, hi, max_total, deadline, max_iter=300, tol=1e-7):
    """
    Nelder-Mead with bound handling, total thickness projection, and deadline check.
    """
    x0 = np.array(x0, dtype=float)
    n = len(x0)
    
    x0 = _project_to_feasible(x0, lo, hi, max_total)
    
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        step = 0.05 * (hi[i] - lo[i]) if hi[i] - lo[i] > 1e-6 else 1.0
        xi[i] += step
        simplex.append(_project_to_feasible(xi, lo, hi, max_total))

    f_vals = [objective(x) for x in simplex]
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break

        order = np.argsort(f_vals)
        simplex = [simplex[i] for i in order]
        f_vals = [f_vals[i] for i in order]

        if abs(f_vals[0] - f_vals[-1]) < tol:
            break

        centroid = np.mean(simplex[:-1], axis=0)

        x_r = _project_to_feasible(centroid + alpha * (centroid - simplex[-1]), lo, hi, max_total)
        f_r = objective(x_r)

        if f_vals[0] <= f_r < f_vals[-2]:
            simplex[-1] = x_r
            f_vals[-1] = f_r
            continue

        if f_r < f_vals[0]:
            x_e = _project_to_feasible(centroid + gamma * (x_r - centroid), lo, hi, max_total)
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_vals[-1] = f_e
            else:
                simplex[-1] = x_r
                f_vals[-1] = f_r
            continue

        x_c = _project_to_feasible(centroid + rho * (simplex[-1] - centroid), lo, hi, max_total)
        f_c = objective(x_c)
        if f_c < f_vals[-1]:
            simplex[-1] = x_c
            f_vals[-1] = f_c
            continue

        for i in range(1, n + 1):
            simplex[i] = _project_to_feasible(simplex[0] + sigma * (simplex[i] - simplex[0]), lo, hi, max_total)
            f_vals[i] = objective(simplex[i])

    order = np.argsort(f_vals)
    best_idx = order[0]
    return _project_to_feasible(simplex[best_idx], lo, hi, max_total), f_vals[best_idx]

def _coordinate_descent(objective, x0, lo, hi, max_total, deadline, max_rounds=100):
    """
    Cyclic coordinate descent with line search.
    """
    x = np.array(x0, dtype=float)
    x = _project_to_feasible(x, lo, hi, max_total)
    f_current = objective(x)
    n = len(x)
    
    for rnd in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            x_old = x[i]
            best_x = x_old
            best_f = f_current
            # Try several steps
            for step_frac in [0.5, -0.5, 0.25, -0.25, 0.1, -0.1, 0.75, -0.75]:
                x_new = x.copy()
                x_new[i] = np.clip(x_old + step_frac * (hi[i] - lo[i]) * 0.2, lo[i], hi[i])
                x_new = _project_to_feasible(x_new, lo, hi, max_total)
                f_new = objective(x_new)
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new[i]
            if best_x != x_old:
                x[i] = best_x
                x = _project_to_feasible(x, lo, hi, max_total)
                f_current = best_f
                improved = True
        if not improved:
            break
    
    return x, f_current

def design(problem):
    """
    Design a broadband antireflection coating using a search-based approach.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

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

    def update_best(mats, ths):
        nonlocal best_val, best_mats, best_ths
        ths_clipped = [min(max(t, 0.0), max_layer) for t in ths]
        s = sum(ths_clipped)
        if s > max_total:
            if s > 1e-9:
                f = max_total / s
                ths_clipped = [t * f for t in ths_clipped]
        val = _mean_reflectance_percent(mats, ths_clipped, problem, wls, n_mat)
        if val < best_val:
            best_val = val
            best_mats = list(mats)
            best_ths = list(ths_clipped)
        return val

    # Generate diverse initial designs and optimize them
    # Strategy: for each layer count, try multiple starting points
    
    # Common quarter-wave designs
    center_wls = [450, 500, 550, 600, 650, 700]
    
    # Build list of (materials, initial_thicknesses) candidates
    candidates = []
    
    for n_layers in [2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20]:
        if n_layers > max_layers:
            continue
        # Alternating starting with L or H
        for start_char in ['L', 'H']:
            mats = []
            for i in range(n_layers):
                if (i % 2 == 0) == (start_char == 'L'):
                    mats.append('L')
                else:
                    mats.append('H')
            
            # Multiple initial thickness guesses
            for cw in center_wls[:4]:
                ths = []
                for m in mats:
                    n = n_mat[m]
                    q = cw / (4.0 * n)
                    q *= random.uniform(0.7, 1.3)
                    ths.append(q)
                # Scale to fit total constraint
                s = sum(ths)
                if s > max_total:
                    f = max_total / s
                    ths = [t * f for t in ths]
                candidates.append((mats, ths))
            
            # Add random initial points
            for _ in range(2):
                ths = [random.uniform(20, max_layer * 0.6) for _ in range(n_layers)]
                s = sum(ths)
                if s > max_total:
                    f = max_total / s
                    ths = [t * f for t in ths]
                candidates.append((mats, ths))

    # Shuffle for diversity
    random.shuffle(candidates)
    
    # Limit number of candidates based on time
    max_candidates = 80
    candidates = candidates[:max_candidates]
    
    for mats, ths_init in candidates:
        if time.time() > deadline:
            break
        
        n_layers = len(mats)
        lo = np.zeros(n_layers)
        hi = np.full(n_layers, max_layer)
        
        x0 = np.clip(np.array(ths_init, dtype=float), lo, hi)
        x0 = _project_to_feasible(x0, lo, hi, max_total)
        
        def objective(th, m=mats):
            return _mean_reflectance_percent(m, list(th), problem, wls, n_mat)
        
        # First do coordinate descent for quick improvement
        x_cd, f_cd = _coordinate_descent(objective, x0, lo, hi, max_total, deadline, max_rounds=30)
        
        # Then Nelder-Mead for refinement
        x_nm, f_nm = _optimize_nelder_mead(objective, x_cd, lo, hi, max_total, deadline, max_iter=200)
        
        # Use the better of the two
        x_best = x_nm if f_nm <= f_cd else x_cd
        f_best = min(f_nm, f_cd)
        
        update_best(mats, x_best)
        
        # Also try the CD result separately in case NM got stuck
        if f_cd < f_nm:
            update_best(mats, x_cd)
    
    # Additional targeted search: refine the best found so far
    if best_mats is not None and time.time() < deadline - 10:
        n_layers = len(best_mats)
        lo = np.zeros(n_layers)
        hi = np.full(n_layers, max_layer)
        
        def objective(th):
            return _mean_reflectance_percent(best_mats, list(th), problem, wls, n_mat)
        
        # Multiple restarts from perturbed best
        for _ in range(5):
            if time.time() > deadline:
                break
            x0 = np.array(best_ths, dtype=float).copy()
            # Perturb
            for i in range(n_layers):
                if random.random() < 0.5:
                    x0[i] += random.uniform(-0.1, 0.1) * (hi[i] - lo[i])
                    x0[i] = np.clip(x0[i], lo[i], hi[i])
            
            x_opt, f_opt = _optimize_nelder_mead(objective, x0, lo, hi, max_total, deadline, max_iter=150)
            update_best(best_mats, x_opt)
    
    # Fallback
    if best_mats is None:
        best_mats = ['L']
        best_ths = [600.0 / (4.0 * n_L)]
        update_best(best_mats, best_ths)
    
    # Final constraint check
    best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
    s = sum(best_ths)
    if s > max_total + 1e-6:
        f = max_total / s
        best_ths = [t * f for t in best_ths]
        best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
    
    return list(best_mats), list(best_ths)
