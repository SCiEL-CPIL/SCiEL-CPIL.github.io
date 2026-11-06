"""Broadband antireflection coating design via simulated annealing with adaptive restarts."""
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

def _objective(thicknesses, indices, wavelengths_nm, ambient_idx, substrate_idx, 
               max_layer_nm, max_total_nm):
    """
    Objective: mean reflectance in percent. Returns large penalty if constraints violated.
    """
    thicknesses = np.asarray(thicknesses, dtype=float)
    if np.any(thicknesses < -1e-6) or np.any(thicknesses > max_layer_nm + 1e-6):
        return 100.0
    if np.sum(thicknesses) > max_total_nm + 1e-6:
        return 100.0
    
    thicknesses_clipped = np.clip(thicknesses, 0.0, max_layer_nm)
    if np.sum(thicknesses_clipped) > max_total_nm:
        return 100.0
    
    R = _compute_reflectance(wavelengths_nm, indices, thicknesses_clipped, 
                             ambient_idx, substrate_idx)
    return np.mean(R) * 100.0

def _local_polish(f, x0, bounds, max_iter=500, tol=1e-8, time_deadline=None):
    """
    Fast local polish using a simplified Nelder-Mead with aggressive convergence criteria.
    """
    n = len(x0)
    lo_bounds = np.array([b[0] for b in bounds])
    hi_bounds = np.array([b[1] for b in bounds])
    x0 = np.clip(np.asarray(x0, dtype=float), lo_bounds, hi_bounds)
    
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    
    for i in range(n):
        simplex[i + 1] = x0.copy()
        lo, hi = bounds[i]
        step = 0.02 * (hi - lo) if hi > lo else 0.02
        simplex[i + 1][i] += step
    
    fvals = np.array([f(simplex[i]) for i in range(n + 1)])
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for it in range(max_iter):
        if time_deadline is not None and time.time() > time_deadline:
            break
        
        idx = np.argsort(fvals)
        simplex = simplex[idx]
        fvals = fvals[idx]
        
        if np.max(fvals) - np.min(fvals) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        x_r = centroid + alpha * (centroid - simplex[-1])
        x_r = np.clip(x_r, lo_bounds, hi_bounds)
        f_r = f(x_r)
        
        if f_r < fvals[0]:
            x_e = centroid + gamma * (x_r - centroid)
            x_e = np.clip(x_e, lo_bounds, hi_bounds)
            f_e = f(x_e)
            
            if f_e < f_r:
                simplex[-1] = x_e
                fvals[-1] = f_e
            else:
                simplex[-1] = x_r
                fvals[-1] = f_r
        elif f_r < fvals[-2]:
            simplex[-1] = x_r
            fvals[-1] = f_r
        else:
            x_c = centroid + rho * (simplex[-1] - centroid)
            x_c = np.clip(x_c, lo_bounds, hi_bounds)
            f_c = f(x_c)
            
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], lo_bounds, hi_bounds)
                    fvals[i] = f(simplex[i])
    
    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]

def _simulated_annealing(f, x0, bounds, time_budget, rng, 
                         T_start=50.0, T_end=0.01, cooling_rate=0.995,
                         max_steps_per_temp=50):
    """
    Simulated annealing with adaptive step size and constraint handling.
    """
    n = len(x0)
    lo_bounds = np.array([b[0] for b in bounds])
    hi_bounds = np.array([b[1] for b in bounds])
    
    x = np.clip(np.asarray(x0, dtype=float), lo_bounds, hi_bounds)
    f_x = f(x)
    
    best_x = x.copy()
    best_f = f_x
    
    T = T_start
    step_scale = 0.1  # Initial step scale
    
    start_time = time.time()
    
    while T > T_end:
        if time.time() - start_time > time_budget:
            break
        
        for _ in range(max_steps_per_temp):
            if time.time() - start_time > time_budget:
                break
            
            # Adaptive step: perturb a random subset of dimensions
            n_perturb = max(1, int(n * 0.3))
            dims = rng.choice(n, size=n_perturb, replace=False)
            
            x_new = x.copy()
            for d in dims:
                step = rng.uniform(-step_scale, step_scale) * (hi_bounds[d] - lo_bounds[d])
                x_new[d] += step
            
            x_new = np.clip(x_new, lo_bounds, hi_bounds)
            
            f_new = f(x_new)
            
            # Accept if better, or with probability exp(-delta/T)
            if f_new < f_x:
                x = x_new
                f_x = f_new
                if f_x < best_f:
                    best_f = f_x
                    best_x = x.copy()
            else:
                delta = f_new - f_x
                if delta < 100.0:  # Don't accept huge penalties
                    prob = np.exp(-delta / max(T, 1e-10))
                    if rng.random() < prob:
                        x = x_new
                        f_x = f_new
        
        T *= cooling_rate
        
        # Adaptive step scale: reduce if many rejections, increase if many accepts
        step_scale *= 0.95
    
    return best_x, best_f

def design(problem):
    """
    Design a broadband antireflection coating.
    Returns: (labels, thicknesses)
    """
    start_time = time.time()
    deadline = start_time + 145.0
    
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wavelengths_nm = np.linspace(band_nm[0], band_nm[1], 401)
    center_wl = 0.5 * (band_nm[0] + band_nm[1])
    
    best_labels = None
    best_thicknesses = None
    best_score = np.inf
    
    # Generate diverse layer patterns
    patterns = []
    
    # Alternating starting with L
    for n in range(3, min(max_layers + 1, 21), 2):
        labels = ['L' if i % 2 == 0 else 'H' for i in range(n)]
        patterns.append((n, labels, "alt_L"))
    
    # Alternating starting with H
    for n in range(3, min(max_layers + 1, 21), 2):
        labels = ['H' if i % 2 == 0 else 'L' for i in range(n)]
        patterns.append((n, labels, "alt_H"))
    
    # Some non-alternating patterns for broadband coverage
    # These can help with phase matching across the band
    special_patterns = [
        ['L', 'H', 'L', 'H', 'L', 'H', 'L'],
        ['H', 'L', 'H', 'L', 'H', 'L', 'H'],
        ['L', 'H', 'H', 'L', 'L', 'H', 'H', 'L'],
        ['H', 'L', 'L', 'H', 'H', 'L', 'L', 'H'],
        ['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'],
        ['H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'],
    ]
    for labels in special_patterns:
        if len(labels) <= max_layers:
            patterns.append((len(labels), labels, "special"))
    
    # Prioritize patterns: 5-15 layers first, then others
    def pattern_priority(p):
        n = p[0]
        if 5 <= n <= 15:
            return 0
        elif 3 <= n < 5 or 15 < n <= 19:
            return 1
        else:
            return 2
    
    patterns.sort(key=pattern_priority)
    
    rng = np.random.RandomState(12345)
    
    # For each pattern, use simulated annealing + local polish
    for n_layers, labels, ptype in patterns:
        if time.time() > deadline - 30:
            break
        
        indices = [materials[label] for label in labels]
        bounds = [(0.0, float(max_layer_nm))] * n_layers
        
        def make_f(indices):
            def f(x):
                return _objective(x, indices, wavelengths_nm, ambient_idx, substrate_idx,
                                  max_layer_nm, max_total_nm)
            return f
        
        f = make_f(indices)
        
        # Multiple starting points for SA
        n_starts = 3
        remaining_time = deadline - 25 - time.time()
        time_per_pattern = max(2.0, min(15.0, remaining_time / max(1, len(patterns))))
        time_per_start = time_per_pattern / n_starts
        
        best_local_x = None
        best_local_f = np.inf
        
        for start_i in range(n_starts):
            if time.time() > deadline - 20:
                break
            
            # Different initial guesses
            if start_i == 0:
                # Quarter-wave initial guess
                x0 = np.array([center_wl / (4.0 * n) for n in indices])
                x0 = np.clip(x0, 0, max_layer_nm)
                if np.sum(x0) > max_total_nm:
                    x0 = x0 * (max_total_nm / np.sum(x0))
            elif start_i == 1:
                # Random initialization
                x0 = rng.uniform(0, max_layer_nm, size=n_layers)
                if np.sum(x0) > max_total_nm:
                    x0 = x0 * (max_total_nm / np.sum(x0))
            else:
                # Thin film initial guess (optimized for center wavelength)
                x0 = np.array([center_wl / (2.0 * n) for n in indices])
                x0 = np.clip(x0, 0, max_layer_nm)
                if np.sum(x0) > max_total_nm:
                    x0 = x0 * (max_total_nm / np.sum(x0))
            
            # Simulated annealing
            sa_deadline = time.time() + time_per_start
            try:
                x_sa, f_sa = _simulated_annealing(f, x0, bounds, time_per_start, rng,
                                                   T_start=30.0, T_end=0.005,
                                                   cooling_rate=0.99, max_steps_per_temp=30)
            except Exception:
                x_sa = x0
                f_sa = f(x0)
            
            if f_sa < best_local_f:
                best_local_f = f_sa
                best_local_x = x_sa.copy()
            
            # Local polish from SA result
            if f_sa < 100.0 and time.time() < deadline - 15:
                polish_deadline = time.time() + max(0.5, min(3.0, (deadline - 15 - time.time())))
                try:
                    x_pol, f_pol = _local_polish(f, x_sa, bounds, max_iter=300, tol=1e-8,
                                                  time_deadline=polish_deadline)
                    if f_pol < best_local_f:
                        best_local_f = f_pol
                        best_local_x = x_pol.copy()
                except Exception:
                    pass
        
        # Update global best
        if best_local_x is not None and best_local_f < best_score:
            best_score = best_local_f
            best_labels = labels.copy()
            best_thicknesses = best_local_x.copy()
        
        # Early exit if very good
        if best_score < 0.15:
            break
    
    # Final fallback
    if best_labels is None:
        return ["L"], [center_wl / (4.0 * materials["L"])]
    
    # Ensure constraints on final result
    thicknesses = np.clip(best_thicknesses, 0, max_layer_nm)
    if np.sum(thicknesses) > max_total_nm:
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
    
    return best_labels, thicknesses.tolist()
