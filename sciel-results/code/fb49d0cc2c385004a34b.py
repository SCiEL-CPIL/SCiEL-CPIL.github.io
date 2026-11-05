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

    ns = np.array([n_mat[m] for m in materials])
    ds = np.array(thicknesses, dtype=float)
    beta = 2.0 * math.pi * np.outer(ns * ds, 1.0 / wls)

    cosb = np.cos(beta)
    sinb = np.sin(beta)

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

def _project(x, lo, hi, max_total):
    """Project to feasible region."""
    x = np.clip(x, lo, hi)
    s = np.sum(x)
    if s > max_total:
        if s > 1e-9:
            x = x * (max_total / s)
        else:
            x = np.zeros_like(x)
    x = np.clip(x, lo, hi)
    return x

def _nelder_mead(objective, x0, lo, hi, max_total, deadline, max_iter=500, tol=1e-8):
    """Nelder-Mead with bound handling."""
    x0 = np.array(x0, dtype=float)
    n = len(x0)
    x0 = _project(x0, lo, hi, max_total)
    
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        step = 0.1 * (hi[i] - lo[i]) if hi[i] - lo[i] > 1e-6 else 1.0
        xi[i] += step
        simplex.append(_project(xi, lo, hi, max_total))

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

        x_r = _project(centroid + alpha * (centroid - simplex[-1]), lo, hi, max_total)
        f_r = objective(x_r)

        if f_vals[0] <= f_r < f_vals[-2]:
            simplex[-1] = x_r
            f_vals[-1] = f_r
            continue

        if f_r < f_vals[0]:
            x_e = _project(centroid + gamma * (x_r - centroid), lo, hi, max_total)
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_vals[-1] = f_e
            else:
                simplex[-1] = x_r
                f_vals[-1] = f_r
            continue

        x_c = _project(centroid + rho * (simplex[-1] - centroid), lo, hi, max_total)
        f_c = objective(x_c)
        if f_c < f_vals[-1]:
            simplex[-1] = x_c
            f_vals[-1] = f_c
            continue

        for i in range(1, n + 1):
            simplex[i] = _project(simplex[0] + sigma * (simplex[i] - simplex[0]), lo, hi, max_total)
            f_vals[i] = objective(simplex[i])

    order = np.argsort(f_vals)
    best_idx = order[0]
    return _project(simplex[best_idx], lo, hi, max_total), f_vals[best_idx]

def _design_quarter_wave_stack(n_layers, n_L, n_H, n_sub, wls, n_mat, problem):
    """
    Design a quarter-wave stack. For broadband, use effective thickness
    that balances performance across the band.
    """
    n_ambient = problem["ambient_index"]
    # For a single quarter-wave layer on substrate:
    # n_coating = sqrt(n_ambient * n_substrate) for zero reflection at design wavelength
    # For broadband, we use the geometric mean as a starting point
    
    # For alternating L/H layers, use quarter-wave at center wavelength
    wl_center = 600.0  # center of 400-800 band
    
    mats = []
    ths = []
    for i in range(n_layers):
        if i % 2 == 0:
            mats.append('L')
            ths.append(wl_center / (4.0 * n_L))
        else:
            mats.append('H')
            ths.append(wl_center / (4.0 * n_H))
    
    # Check constraints
    max_layer = problem["max_layer_nm"]
    max_total = problem["max_total_nm"]
    ths = [min(max(t, 0.0), max_layer) for t in ths]
    s = sum(ths)
    if s > max_total:
        f = max_total / s
        ths = [t * f for t in ths]
    
    return mats, ths

def _simulated_annealing(problem, wls, n_mat, deadline, max_layers=20, max_total=1000, max_layer=400):
    """
    Simulated annealing for global exploration.
    """
    n_L = n_mat['L']
    n_H = n_mat['H']
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    
    # Start with a quarter-wave stack
    n_init = random.choice([6, 8, 10, 12, 14])
    mats, ths = _design_quarter_wave_stack(n_init, n_L, n_H, n_substrate, wls, n_mat, problem)
    
    best_mats = list(mats)
    best_ths = list(ths)
    best_val = _mean_reflectance_percent(best_mats, best_ths, problem, wls, n_mat)
    curr_val = best_val
    
    T0 = 1.5
    T_end = 0.005
    n_iterations = 0
    
    start_time = time.time()
    
    while time.time() < deadline - 3:
        n_iterations += 1
        
        new_mats = list(mats)
        new_ths = list(ths)
        
        action = random.random()
        
        if action < 0.10:
            # Change number of layers
            if random.random() < 0.5 and len(mats) < max_layers:
                new_mats.append(random.choice(['L', 'H']))
                new_ths.append(random.uniform(20, 80))
            elif len(mats) > 2:
                idx = random.randint(0, len(mats) - 1)
                new_mats.pop(idx)
                new_ths.pop(idx)
        
        elif action < 0.25:
            # Flip material at some positions
            for _ in range(random.randint(1, 2)):
                idx = random.randint(0, len(new_mats) - 1)
                new_mats[idx] = 'L' if new_mats[idx] == 'H' else 'H'
        
        elif action < 0.45:
            # Single thickness perturbation
            idx = random.randint(0, len(new_ths) - 1)
            new_ths[idx] += random.uniform(-80, 80)
            new_ths[idx] = min(max(new_ths[idx], 0.0), max_layer)
        
        elif action < 0.65:
            # Multiple thickness perturbations
            for _ in range(random.randint(1, 3)):
                idx = random.randint(0, len(new_ths) - 1)
                new_ths[idx] += random.uniform(-60, 60)
                new_ths[idx] = min(max(new_ths[idx], 0.0), max_layer)
        
        elif action < 0.80:
            # Quarter-wave reset for some layers
            for _ in range(random.randint(1, 2)):
                idx = random.randint(0, len(new_ths) - 1)
                n = n_mat[new_mats[idx]]
                cw = random.choice([450, 500, 550, 600, 650, 700])
                new_ths[idx] = cw / (4.0 * n)
        
        elif action < 0.90:
            # Randomize a few thicknesses
            for _ in range(random.randint(1, 3)):
                idx = random.randint(0, len(new_ths) - 1)
                new_ths[idx] = random.uniform(10, 100)
        
        else:
            # Full random restart
            n_new = random.randint(4, min(max_layers, 16))
            new_mats, new_ths = _design_quarter_wave_stack(n_new, n_L, n_H, n_substrate, wls, n_mat, problem)
            for i in range(len(new_ths)):
                new_ths[i] += random.uniform(-20, 20)
                new_ths[i] = min(max(new_ths[i], 0.0), max_layer)
        
        # Project to feasible
        new_ths = [min(max(t, 0.0), max_layer) for t in new_ths]
        s = sum(new_ths)
        if s > max_total:
            f = max_total / s
            new_ths = [t * f for t in new_ths]
        
        new_val = _mean_reflectance_percent(new_mats, new_ths, problem, wls, n_mat)
        
        if new_val < curr_val:
            accept = True
        else:
            delta = new_val - curr_val
            T = T0 * (T_end / T0) ** (n_iterations / 8000.0)
            T = max(T, T_end)
            if T > 1e-10:
                accept = random.random() < math.exp(-delta / T)
            else:
                accept = False
        
        if accept:
            mats = new_mats
            ths = new_ths
            curr_val = new_val
            if curr_val < best_val:
                best_val = curr_val
                best_mats = list(mats)
                best_ths = list(ths)
        
        if n_iterations % 200 == 0 and time.time() > deadline - 5:
            break
    
    return best_mats, best_ths, best_val

def design(problem):
    """
    Design a broadband antireflection coating.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

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

    # Phase 1: Try several quarter-wave starting points
    for n_layers in [4, 6, 8, 10, 12, 14, 16]:
        if time.time() > deadline - 30:
            break
        mats, ths = _design_quarter_wave_stack(n_layers, n_L, n_H, n_substrate, wls, n_mat, problem)
        update_best(mats, ths)
    
    # Phase 2: Simulated annealing for global exploration
    sa_deadline = start_time + 50.0
    sa_mats, sa_ths, sa_val = _simulated_annealing(
        problem, wls, n_mat, sa_deadline, max_layers, max_total, max_layer
    )
    update_best(sa_mats, sa_ths)
    
    # Phase 3: Refine with Nelder-Mead
    if time.time() < deadline - 10:
        n_layers = len(best_mats)
        lo = np.zeros(n_layers)
        hi = np.full(n_layers, max_layer)
        
        def objective(th):
            return _mean_reflectance_percent(best_mats, list(th), problem, wls, n_mat)
        
        x0 = np.array(best_ths, dtype=float).copy()
        x0 = _project(x0, lo, hi, max_total)
        
        # Multiple Nelder-Mead attempts with different perturbations
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            
            if attempt == 0:
                x_start = x0
            else:
                x_start = x0.copy()
                for i in range(n_layers):
                    if random.random() < 0.4:
                        x_start[i] += random.uniform(-0.15, 0.15) * (hi[i] - lo[i])
                        x_start[i] = np.clip(x_start[i], lo[i], hi[i])
                x_start = _project(x_start, lo, hi, max_total)
            
            x_nm, f_nm = _nelder_mead(objective, x_start, lo, hi, max_total, deadline, max_iter=300)
            if f_nm < best_val:
                update_best(best_mats, x_nm)
    
    # Phase 4: More SA runs if time allows
    if time.time() < deadline - 20:
        for _ in range(2):
            if time.time() > deadline - 10:
                break
            sa_deadline2 = min(time.time() + 8, deadline - 10)
            sa_mats2, sa_ths2, sa_val2 = _simulated_annealing(
                problem, wls, n_mat, sa_deadline2, max_layers, max_total, max_layer
            )
            update_best(sa_mats2, sa_ths2)
            
            # Quick refine
            if time.time() < deadline - 5:
                n_layers2 = len(best_mats)
                lo2 = np.zeros(n_layers2)
                hi2 = np.full(n_layers2, max_layer)
                
                def objective2(th):
                    return _mean_reflectance_percent(best_mats, list(th), problem, wls, n_mat)
                
                x0_2 = np.array(best_ths, dtype=float).copy()
                x0_2 = _project(x0_2, lo2, hi2, max_total)
                x_nm2, f_nm2 = _nelder_mead(objective2, x0_2, lo2, hi2, max_total, deadline, max_iter=150)
                if f_nm2 < best_val:
                    update_best(best_mats, x_nm2)
    
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
