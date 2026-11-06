"""Broadband antireflection coating design via discrete GA + continuous Nelder-Mead polish."""
import numpy as np
import time
import random

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

def _discrete_ga(problem, wavelengths_nm, max_layers, max_total_nm, max_layer_nm, 
                 pop_size=100, generations=200, time_limit=40.0, start_time=None):
    """
    Discrete GA over layer count, labels, and quarter-wave grid thicknesses.
    Explores many valid multi-layer stacks leveraging known broadband AR structures.
    """
    if start_time is None:
        start_time = time.time()
    
    materials = problem["materials"]
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    
    # Grid of thicknesses: quarter-wave steps at center wavelength
    center_wl = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    # Use fine grid: 1/16 of quarter-wave at center
    qw_L = center_wl / (4.0 * materials["L"])
    qw_H = center_wl / (4.0 * materials["H"])
    # Grid spacing: 1/16 of the smaller quarter-wave
    grid_step = min(qw_L, qw_H) / 16.0
    # Number of grid points per layer
    n_grid_L = int(max_layer_nm / grid_step) + 1
    n_grid_H = int(max_layer_nm / grid_step) + 1
    
    # Encode individual: (n_layers, labels, thicknesses)
    # We'll use a simpler encoding: just the thicknesses for a fixed set of candidate structures
    
    # Candidate structures: various layer counts and label patterns
    candidate_structures = []
    for n_layers in range(1, min(max_layers + 1, 20)):
        # Generate several label patterns
        patterns = []
        # Alternating starting with L
        patterns.append(['L' if i % 2 == 0 else 'H' for i in range(n_layers)])
        # Alternating starting with H
        patterns.append(['H' if i % 2 == 0 else 'L' for i in range(n_layers)])
        # All L
        patterns.append(['L'] * n_layers)
        # All H
        patterns.append(['H'] * n_layers)
        # LHLH... or HLHL...
        if n_layers >= 2:
            patterns.append(['L' if i % 4 < 2 else 'H' for i in range(n_layers)])
            patterns.append(['H' if i % 4 < 2 else 'L' for i in range(n_layers)])
        
        for pat in patterns:
            candidate_structures.append(pat)
    
    # Deduplicate
    seen = set()
    unique_structures = []
    for pat in candidate_structures:
        key = tuple(pat)
        if key not in seen:
            seen.add(key)
            unique_structures.append(pat)
    
    # Initialize population: random thicknesses for random structures
    population = []
    for _ in range(pop_size):
        struct = random.choice(unique_structures)
        n_layers = len(struct)
        thicknesses = np.array([random.uniform(0, max_layer_nm) for _ in range(n_layers)])
        # Scale to fit total constraint
        total = np.sum(thicknesses)
        if total > max_total_nm:
            thicknesses = thicknesses * (max_total_nm / total)
        population.append((struct, thicknesses))
    
    # Evaluate initial population
    def evaluate_individual(individual):
        struct, thicknesses = individual
        problem_local = dict(problem)
        problem_local["labels"] = struct
        indices = [materials[label] for label in struct]
        R = _compute_reflectance(wavelengths_nm, indices, thicknesses, ambient_idx, substrate_idx)
        return np.mean(R) * 100.0
    
    fvals = [evaluate_individual(ind) for ind in population]
    
    best_idx = np.argmin(fvals)
    best_individual = population[best_idx]
    best_f = fvals[best_idx]
    
    # GA loop
    for gen in range(generations):
        if time.time() - start_time > time_limit:
            break
        
        # Sort by fitness
        sorted_idx = np.argsort(fvals)
        population = [population[i] for i in sorted_idx]
        fvals = [fvals[i] for i in sorted_idx]
        
        # Elitism: keep top 10%
        elite_size = max(1, pop_size // 10)
        new_population = [population[i] for i in range(elite_size)]
        new_fvals = [fvals[i] for i in range(elite_size)]
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            # Tournament selection
            def tournament_select():
                idxs = np.random.randint(0, pop_size, size=3)
                best = min(idxs, key=lambda i: fvals[i])
                return population[best]
            
            parent1 = tournament_select()
            parent2 = tournament_select()
            
            # Crossover: mix structure and thicknesses
            if random.random() < 0.5:
                # Use parent1's structure
                struct = parent1[0]
            else:
                struct = parent2[0]
            
            n_layers = len(struct)
            
            # Create offspring thicknesses
            # If structures differ, resize
            if len(parent1[1]) != n_layers:
                t1 = np.array([random.uniform(0, max_layer_nm) for _ in range(n_layers)])
            else:
                t1 = parent1[1].copy()
            
            if len(parent2[1]) != n_layers:
                t2 = np.array([random.uniform(0, max_layer_nm) for _ in range(n_layers)])
            else:
                t2 = parent2[1].copy()
            
            # Blend
            alpha = random.uniform(0.3, 0.7)
            offspring_thick = alpha * t1 + (1 - alpha) * t2
            
            # Mutation
            for j in range(n_layers):
                if random.random() < 0.1:
                    offspring_thick[j] = random.uniform(0, max_layer_nm)
            
            # Ensure constraints
            offspring_thick = np.clip(offspring_thick, 0, max_layer_nm)
            total = np.sum(offspring_thick)
            if total > max_total_nm:
                offspring_thick = offspring_thick * (max_total_nm / total)
            
            new_population.append((struct, offspring_thick))
            new_fvals.append(None)  # placeholder
        
        # Evaluate new individuals
        for i in range(elite_size, pop_size):
            new_fvals[i] = evaluate_individual(new_population[i])
        
        population = new_population
        fvals = new_fvals
        
        # Update best
        best_idx = np.argmin(fvals)
        if fvals[best_idx] < best_f:
            best_f = fvals[best_idx]
            best_individual = population[best_idx]
    
    return best_individual, best_f

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
    time_limit = 140.0
    
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
    best_labels = None
    best_thicknesses = None
    
    # Phase 1: Discrete GA to find good structure and rough thicknesses
    elapsed = time.time() - start_time
    ga_time = min(40.0, time_limit - elapsed - 80.0)
    
    if ga_time > 5.0:
        try:
            ga_struct, ga_thick, ga_score = _discrete_ga(
                problem, wavelengths_nm, max_layers, max_total_nm, max_layer_nm,
                pop_size=80, generations=150, time_limit=ga_time, start_time=start_time
            )
            if ga_score < best_score:
                best_score = ga_score
                best_labels = ga_struct
                best_thicknesses = ga_thick
        except Exception:
            pass
    
    # Phase 2: Continuous optimization on the best structure from GA
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed - 20.0
    
    if remaining < 10.0:
        # Not enough time, use GA result or fallback
        if best_labels is not None:
            thicknesses = np.clip(best_thicknesses, 0, max_layer_nm)
            if np.sum(thicknesses) > max_total_nm:
                thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
            return best_labels, thicknesses.tolist()
        else:
            center_wl = 0.5 * (band_nm[0] + band_nm[1])
            return ["L"], [center_wl / (4.0 * materials["L"])]
    
    # Try several layer counts with continuous Nelder-Mead
    # Prioritize the structure found by GA, then try others
    layer_counts = [len(best_labels)] if best_labels else []
    # Add other promising counts
    for nc in [11, 13, 9, 15, 7, 17, 5, 19, 3]:
        if nc not in layer_counts and nc <= max_layers:
            layer_counts.append(nc)
    
    for n_layers in layer_counts:
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed - 15.0
        
        if remaining < 10.0:
            break
        
        if n_layers > max_layers:
            continue
        
        # Create labels
        if n_layers == len(best_labels) and best_labels is not None:
            labels = best_labels.copy()
        else:
            labels = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        
        problem_local = dict(problem)
        problem_local["labels"] = labels
        
        # Initial guess
        if n_layers == len(best_labels) and best_labels is not None:
            init_thicknesses = np.clip(best_thicknesses, 0, max_layer_nm)
            if np.sum(init_thicknesses) > max_total_nm:
                init_thicknesses = init_thicknesses * (max_total_nm / np.sum(init_thicknesses))
        else:
            center_wl = 0.5 * (band_nm[0] + band_nm[1])
            indices = [materials[label] for label in labels]
            init_thicknesses = np.array([center_wl / (4.0 * n) for n in indices])
            if np.sum(init_thicknesses) > max_total_nm:
                init_thicknesses = init_thicknesses * (max_total_nm / np.sum(init_thicknesses))
        
        bounds = [(0, max_layer_nm)] * n_layers
        
        def obj_func(x, bounds_local, start_time_local):
            return _objective(x, problem_local, wavelengths_nm)
        
        # Nelder-Mead for local refinement
        nm_time = min(remaining * 0.6, 30.0)
        polish_time = min(remaining * 0.3, 15.0)
        
        try:
            x_nm, f_nm = _nelder_mead(obj_func, init_thicknesses, bounds,
                                       max_iter=500, tol=1e-9,
                                       time_limit=nm_time,
                                       start_time=start_time)
        except Exception:
            x_nm, f_nm = init_thicknesses, obj_func(init_thicknesses, bounds, start_time)
        
        # Final polish
        try:
            x_pol, f_pol = _polish(x_nm, bounds, obj_func, start_time, 
                                    time_budget=polish_time)
        except Exception:
            x_pol, f_pol = x_nm, f_nm
        
        # Use the best
        candidates = [(x_nm, f_nm), (x_pol, f_pol)]
        best_candidate = min(candidates, key=lambda t: t[1])
        x_opt, f_opt = best_candidate
        
        if f_opt < best_score:
            best_score = f_opt
            best_labels = labels.copy()
            best_thicknesses = x_opt.copy()
        
        # Early termination
        if best_score < 0.5:
            break
    
    # Fallback
    if best_labels is None:
        center_wl = 0.5 * (band_nm[0] + band_nm[1])
        return ["L"], [center_wl / (4.0 * materials["L"])]
    
    # Ensure constraints
    thicknesses = np.clip(best_thicknesses, 0, max_layer_nm)
    if np.sum(thicknesses) > max_total_nm:
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
    
    return best_labels, thicknesses.tolist()
