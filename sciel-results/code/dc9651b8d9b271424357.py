"""Broadband antireflection coating design via differential evolution + Nelder-Mead polish."""
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

def _objective(params, problem, wavelengths_nm):
    """
    Objective function: mean reflectance in percent.
    params: flat array of thicknesses
    """
    max_layer_nm = problem.get("max_layer_nm", 400)
    max_total_nm = problem.get("max_total_nm", 1000)
    
    thicknesses = np.clip(params, 0, max_layer_nm)
    
    if np.sum(thicknesses) > max_total_nm:
        return 100.0 + (np.sum(thicknesses) - max_total_nm)
    
    indices = [problem["materials"][label] for label in problem["labels"]]
    
    R = _compute_reflectance(wavelengths_nm, indices, thicknesses, 
                             problem["ambient_index"], problem["substrate_index"])
    
    return np.mean(R) * 100.0

def _nelder_mead(f, x0, bounds, max_iter=2000, tol=1e-10, time_limit=50.0, start_time=None):
    """
    Custom Nelder-Mead simplex algorithm with bounds handling.
    """
    if start_time is None:
        start_time = time.time()
    
    n = len(x0)
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    
    for i in range(n):
        simplex[i + 1] = x0.copy()
        step = 0.1 * (bounds[i][1] - bounds[i][0]) if bounds[i][1] > bounds[i][0] else 0.1
        simplex[i + 1][i] += step
    
    fvals = np.array([f(simplex[i], bounds, start_time) for i in range(n + 1)])
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        idx = np.argsort(fvals)
        simplex = simplex[idx]
        fvals = fvals[idx]
        
        if np.max(fvals) - np.min(fvals) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        x_r = centroid + alpha * (centroid - simplex[-1])
        x_r = np.clip(x_r, [b[0] for b in bounds], [b[1] for b in bounds])
        f_r = f(x_r, bounds, start_time)
        
        if f_r < fvals[0]:
            x_e = centroid + gamma * (x_r - centroid)
            x_e = np.clip(x_e, [b[0] for b in bounds], [b[1] for b in bounds])
            f_e = f(x_e, bounds, start_time)
            
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
            x_c = np.clip(x_c, [b[0] for b in bounds], [b[1] for b in bounds])
            f_c = f(x_c, bounds, start_time)
            
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], [b[0] for b in bounds], [b[1] for b in bounds])
                    fvals[i] = f(simplex[i], bounds, start_time)
    
    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]

def _differential_evolution(f, bounds, pop_size=50, max_iter=100, time_limit=30.0, 
                             start_time=None, cr=0.8, F=0.8):
    """
    Differential evolution for global optimization.
    """
    if start_time is None:
        start_time = time.time()
    
    n = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    
    # Initialize population
    pop = np.random.uniform(lo, hi, size=(pop_size, n))
    fvals = np.array([f(pop[i], bounds, start_time) for i in range(pop_size)])
    
    best_idx = np.argmin(fvals)
    best_x = pop[best_idx].copy()
    best_f = fvals[best_idx]
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        for i in range(pop_size):
            if time.time() - start_time > time_limit:
                break
            
            # Select three distinct random individuals
            a, b, c = np.random.choice(pop_size, size=3, replace=False)
            
            # Mutation
            mutant = pop[a] + F * (pop[b] - pop[c])
            mutant = np.clip(mutant, lo, hi)
            
            # Crossover
            trial = pop[i].copy()
            for j in range(n):
                if np.random.random() < cr:
                    trial[j] = mutant[j]
                else:
                    trial[j] = pop[i][j]
            
            # Evaluate
            f_trial = f(trial, bounds, start_time)
            
            # Selection
            if f_trial < fvals[i]:
                pop[i] = trial
                fvals[i] = f_trial
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()
    
    return best_x, best_f

def _polish(x0, bounds, f, start_time, time_budget=10.0, tol=1e-12):
    """
    Final high-precision polish pass using tight-tolerance Nelder-Mead.
    """
    if time.time() - start_time > start_time + time_budget:
        return x0, f(x0, bounds, start_time)
    
    n = len(x0)
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    
    for i in range(n):
        simplex[i + 1] = x0.copy()
        step = 0.01 * (bounds[i][1] - bounds[i][0]) if bounds[i][1] > bounds[i][0] else 0.01
        simplex[i + 1][i] += step
    
    fvals = np.array([f(simplex[i], bounds, start_time) for i in range(n + 1)])
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    max_iter = 500
    for it in range(max_iter):
        if time.time() - start_time > start_time + time_budget:
            break
        
        idx = np.argsort(fvals)
        simplex = simplex[idx]
        fvals = fvals[idx]
        
        if np.max(fvals) - np.min(fvals) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        x_r = centroid + alpha * (centroid - simplex[-1])
        x_r = np.clip(x_r, [b[0] for b in bounds], [b[1] for b in bounds])
        f_r = f(x_r, bounds, start_time)
        
        if f_r < fvals[0]:
            x_e = centroid + gamma * (x_r - centroid)
            x_e = np.clip(x_e, [b[0] for b in bounds], [b[1] for b in bounds])
            f_e = f(x_e, bounds, start_time)
            
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
            x_c = np.clip(x_c, [b[0] for b in bounds], [b[1] for b in bounds])
            f_c = f(x_c, bounds, start_time)
            
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], [b[0] for b in bounds], [b[1] for b in bounds])
                    fvals[i] = f(simplex[i], bounds, start_time)
    
    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]

def design(problem):
    """
    Design a broadband antireflection coating.
    Returns: (labels, thicknesses)
    """
    start_time = time.time()
    time_limit = 120.0  # Tighter safety margin
    
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wavelengths_nm = np.linspace(band_nm[0], band_nm[1], 401)
    
    best_result = None
    best_score = np.inf
    best_x = None
    best_labels = None
    best_bounds = None
    best_problem_local = None
    
    # Strategy: try different layer counts and patterns with DE + NM polish
    # Patterns: Alternating (LH...), Block (LLHH...), and some random variations
    # We will iterate over a few layer counts and for each, try a few patterns.
    
    layer_counts = [11, 13, 9, 15, 7, 17, 5, 19, 3]
    
    for n_layers in layer_counts:
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed - 10.0 # Reserve 10s for final checks
        
        if remaining < 5.0:
            break
        
        if n_layers > max_layers:
            continue
        
        # Generate diverse patterns
        patterns = []
        
        # 1. Standard Alternating (LHLH...)
        patterns.append(['L' if i % 2 == 0 else 'H' for i in range(n_layers)])
        
        # 2. Standard Alternating (HHLH... - starting with H)
        patterns.append(['H' if i % 2 == 0 else 'L' for i in range(n_layers)])
        
        # 3. Block pattern (LLHHLLHH...)
        patterns.append(['L' if (i // 2) % 2 == 0 else 'H' for i in range(n_layers)])
        
        # 4. Block pattern (HHLLHHLL...)
        patterns.append(['H' if (i // 2) % 2 == 0 else 'L' for i in range(n_layers)])
        
        # 5. Random pattern (seeded for reproducibility per run, but different per layer count)
        np.random.seed(42 + n_layers)
        random_pattern = list(np.random.choice(['L', 'H'], size=n_layers))
        patterns.append(random_pattern)
        
        # Remove duplicates if any
        unique_patterns = []
        for p in patterns:
            if p not in unique_patterns:
                unique_patterns.append(p)
        
        # Allocate time for each pattern
        time_per_pattern = (remaining - 2.0) / len(unique_patterns)
        
        for labels in unique_patterns:
            if time.time() - start_time > time_limit - 5.0:
                break
                
            problem_local = dict(problem)
            problem_local["labels"] = labels
            
            # Initial guess: quarter-wave at center wavelength
            center_wl = 0.5 * (band_nm[0] + band_nm[1])
            indices = [materials[label] for label in labels]
            init_thicknesses = np.array([center_wl / (4.0 * n) for n in indices])
            
            bounds = [(0, max_layer_nm)] * n_layers
            
            def obj_func(x, bounds_local, start_time_local):
                return _objective(x, problem_local, wavelengths_nm)
            
            # Time allocation within pattern: DE 50%, NM 30%, Polish 20%
            de_time = time_per_pattern * 0.5
            nm_time = time_per_pattern * 0.3
            polish_time = time_per_pattern * 0.2
            
            # Phase 1: Differential Evolution for global search
            try:
                x_de, f_de = _differential_evolution(obj_func, bounds, 
                                                      pop_size=30, max_iter=50, # Reduced pop/iter for more patterns
                                                      time_limit=de_time,
                                                      start_time=start_time)
            except Exception:
                x_de, f_de = init_thicknesses, obj_func(init_thicknesses, bounds, start_time)
            
            # Phase 2: Nelder-Mead for local refinement
            try:
                x_nm, f_nm = _nelder_mead(obj_func, x_de, bounds,
                                           max_iter=300, tol=1e-9,
                                           time_limit=nm_time,
                                           start_time=start_time)
            except Exception:
                x_nm, f_nm = x_de, f_de
            
            # Phase 3: Final polish
            try:
                x_pol, f_pol = _polish(x_nm, bounds, obj_func, start_time, 
                                        time_budget=polish_time)
            except Exception:
                x_pol, f_pol = x_nm, f_nm
            
            # Use the best of all phases
            candidates = [(x_de, f_de), (x_nm, f_nm), (x_pol, f_pol)]
            best_candidate = min(candidates, key=lambda t: t[1])
            x_opt, f_opt = best_candidate
            
            if f_opt < best_score:
                best_score = f_opt
                best_result = (labels, x_opt.tolist())
                best_x = x_opt.copy()
                best_labels = labels.copy()
                best_bounds = bounds
                best_problem_local = problem_local
            
            # Early termination if we've done very well
            if best_score < 0.3:
                break
    
    # Fallback
    if best_result is None:
        center_wl = 0.5 * (band_nm[0] + band_nm[1])
        return ["L"], [center_wl / (4.0 * materials["L"])]
    
    labels, thicknesses = best_result
    
    # Ensure constraints
    thicknesses = np.clip(thicknesses, 0, max_layer_nm)
    if np.sum(thicknesses) > max_total_nm:
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
    
    return labels, thicknesses.tolist()
