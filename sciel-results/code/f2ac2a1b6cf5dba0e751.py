"""Broadband antireflection coating design via discrete genetic algorithm over quarter-wave optical thicknesses."""
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

def _eval_chromosome(chrom, problem, wavelengths_nm, thickness_options, max_layer_nm, max_total_nm):
    """
    Evaluate a chromosome.
    chrom: array of length max_layers, first value is number of layers (1-20),
           remaining values encode layer type (0=L, 1=H) and thickness index
    Returns mean reflectance in percent, or 1000 if invalid.
    """
    n_layers = int(chrom[0])
    if n_layers < 1 or n_layers > 20:
        return 1000.0
    
    # Extract layer info from chromosome
    # For each layer i: chrom[1+i*2] = 0 or 1 (L or H), chrom[2+i*2] = thickness index
    labels = []
    thicknesses = []
    for i in range(n_layers):
        layer_type = int(chrom[1 + i * 2]) % 2  # 0=L, 1=H
        thick_idx = int(chrom[2 + i * 2]) % len(thickness_options)
        
        label = 'L' if layer_type == 0 else 'H'
        thick = thickness_options[thick_idx]
        
        if thick < 0 or thick > max_layer_nm:
            return 1000.0
        
        labels.append(label)
        thicknesses.append(thick)
    
    if sum(thicknesses) > max_total_nm:
        return 1000.0
    
    indices = [problem["materials"][label] for label in labels]
    thicknesses_arr = np.array(thicknesses, dtype=float)
    
    R = _compute_reflectance(wavelengths_nm, indices, thicknesses_arr,
                             problem["ambient_index"], problem["substrate_index"])
    
    return np.mean(R) * 100.0

def _crossover(parent1, parent2, cr=0.7):
    """Uniform crossover between two chromosomes."""
    n = len(parent1)
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    mask = np.random.random(n) < cr
    child1[mask] = parent2[mask]
    child2[mask] = parent1[mask]
    
    return child1, child2

def _mutate(chrom, mutation_rate=0.15, thickness_options=None):
    """Point mutation on a chromosome."""
    n = len(chrom)
    chrom = chrom.copy()
    
    for i in range(n):
        if np.random.random() < mutation_rate:
            if i == 0:
                # Mutate number of layers
                chrom[i] = np.random.randint(1, 21)
            else:
                # Mutate layer type or thickness index
                if i % 2 == 1:
                    chrom[i] = np.random.randint(0, 2)
                else:
                    chrom[i] = np.random.randint(0, len(thickness_options))
    
    return chrom

def _select_tournament(pop, fvals, k=3):
    """Tournament selection."""
    idx = np.random.choice(len(pop), size=k, replace=False)
    best_idx = idx[np.argmin(fvals[idx])]
    return pop[best_idx].copy()

def _genetic_algorithm(problem, wavelengths_nm, thickness_options, max_layer_nm, max_total_nm,
                       pop_size=60, max_iter=200, time_limit=120.0, start_time=None):
    """
    Genetic algorithm over discrete quarter-wave optical thicknesses.
    """
    if start_time is None:
        start_time = time.time()
    
    n_opts = len(thickness_options)
    chrom_len = 1 + 2 * 20  # 1 for num_layers + 2*20 for layer specs
    
    # Initialize population
    pop = np.zeros((pop_size, chrom_len), dtype=int)
    for i in range(pop_size):
        n_layers = np.random.randint(3, 18)
        pop[i, 0] = n_layers
        for j in range(n_layers):
            pop[i, 1 + j * 2] = np.random.randint(0, 2)
            pop[i, 2 + j * 2] = np.random.randint(0, n_opts)
    
    # Evaluate initial population
    fvals = np.array([_eval_chromosome(pop[i], problem, wavelengths_nm,
                                       thickness_options, max_layer_nm, max_total_nm)
                      for i in range(pop_size)])
    
    best_idx = np.argmin(fvals)
    best_chrom = pop[best_idx].copy()
    best_f = fvals[best_idx]
    
    # Elitism: keep top 2
    elite_count = 2
    elite_idx = np.argsort(fvals)[:elite_count]
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Create new population
        new_pop = [pop[idx].copy() for idx in elite_idx]
        
        while len(new_pop) < pop_size:
            # Selection
            p1 = _select_tournament(pop, fvals)
            p2 = _select_tournament(pop, fvals)
            
            # Crossover
            if np.random.random() < 0.8:
                c1, c2 = _crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            
            # Mutation
            c1 = _mutate(c1, mutation_rate=0.15, thickness_options=thickness_options)
            c2 = _mutate(c2, mutation_rate=0.15, thickness_options=thickness_options)
            
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)
        
        new_pop = np.array(new_pop[:pop_size])
        
        # Evaluate new population
        new_fvals = np.array([_eval_chromosome(new_pop[i], problem, wavelengths_nm,
                                               thickness_options, max_layer_nm, max_total_nm)
                              for i in range(pop_size)])
        
        # Update best
        new_best_idx = np.argmin(new_fvals)
        if new_fvals[new_best_idx] < best_f:
            best_f = new_fvals[new_best_idx]
            best_chrom = new_pop[new_best_idx].copy()
        
        # Replace population
        pop = new_pop
        fvals = new_fvals
    
    return best_chrom, best_f

def _decode_chromosome(chrom, problem, thickness_options):
    """
    Decode a chromosome into (labels, thicknesses).
    """
    n_layers = int(chrom[0])
    labels = []
    thicknesses = []
    
    for i in range(n_layers):
        layer_type = int(chrom[1 + i * 2]) % 2
        thick_idx = int(chrom[2 + i * 2]) % len(thickness_options)
        
        label = 'L' if layer_type == 0 else 'H'
        thick = thickness_options[thick_idx]
        
        labels.append(label)
        thicknesses.append(float(thick))
    
    return labels, thicknesses

def _local_refine(labels, thicknesses, problem, wavelengths_nm, time_limit=10.0, start_time=None):
    """
    Local refinement using Nelder-Mead on the continuous thickness values.
    """
    if start_time is None:
        start_time = time.time()
    
    n_layers = len(labels)
    if n_layers == 0:
        return labels, thicknesses
    
    max_layer_nm = problem.get("max_layer_nm", 400)
    max_total_nm = problem.get("max_total_nm", 1000)
    
    bounds = [(0, max_layer_nm)] * n_layers
    x0 = np.array(thicknesses, dtype=float)
    
    def obj_func(x):
        x = np.clip(x, 0, max_layer_nm)
        if np.sum(x) > max_total_nm:
            return 1000.0
        indices = [problem["materials"][label] for label in labels]
        R = _compute_reflectance(wavelengths_nm, indices, x,
                                 problem["ambient_index"], problem["substrate_index"])
        return np.mean(R) * 100.0
    
    # Simple Nelder-Mead
    simplex = np.zeros((n_layers + 1, n_layers))
    simplex[0] = x0
    for i in range(n_layers):
        simplex[i + 1] = x0.copy()
        step = 0.05 * max_layer_nm
        simplex[i + 1][i] += step
    
    fvals = np.array([obj_func(simplex[i]) for i in range(n_layers + 1)])
    
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    max_iter = 300
    
    for it in range(max_iter):
        if time.time() - start_time > start_time + time_limit:
            break
        
        idx = np.argsort(fvals)
        simplex = simplex[idx]
        fvals = fvals[idx]
        
        if np.max(fvals) - np.min(fvals) < 1e-10:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        x_r = centroid + alpha * (centroid - simplex[-1])
        x_r = np.clip(x_r, [b[0] for b in bounds], [b[1] for b in bounds])
        f_r = obj_func(x_r)
        
        if f_r < fvals[0]:
            x_e = centroid + gamma * (x_r - centroid)
            x_e = np.clip(x_e, [b[0] for b in bounds], [b[1] for b in bounds])
            f_e = obj_func(x_e)
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
            f_c = obj_func(x_c)
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                for i in range(1, n_layers + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], [b[0] for b in bounds], [b[1] for b in bounds])
                    fvals[i] = obj_func(simplex[i])
    
    best_idx = np.argmin(fvals)
    x_opt = np.clip(simplex[best_idx], 0, max_layer_nm)
    
    if np.sum(x_opt) > max_total_nm:
        x_opt = x_opt * (max_total_nm / np.sum(x_opt))
    
    return labels, x_opt.tolist()

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
    
    # Define discrete thickness options based on quarter-wave optical thicknesses
    # at various design wavelengths across the band
    # Quarter-wave optical thickness = n * d = lambda / 4
    # So d = lambda / (4 * n)
    
    design_wavelengths = np.arange(400, 801, 50)  # 400, 450, 500, ..., 800 nm
    thickness_options = []
    
    for wl in design_wavelengths:
        for n in [materials['L'], materials['H']]:
            d = wl / (4.0 * n)
            if 0 <= d <= max_layer_nm:
                thickness_options.append(round(d, 2))
    
    # Add some additional options for flexibility
    for wl in [420, 470, 520, 570, 620, 670, 720, 770]:
        for n in [materials['L'], materials['H']]:
            d = wl / (4.0 * n)
            if 0 <= d <= max_layer_nm and d not in thickness_options:
                thickness_options.append(round(d, 2))
    
    # Sort and deduplicate
    thickness_options = sorted(set(thickness_options))
    thickness_options = [t for t in thickness_options if 0 <= t <= max_layer_nm]
    
    # Ensure we have a reasonable number of options
    if len(thickness_options) < 5:
        thickness_options = list(np.arange(50, 401, 25))
    
    np.random.seed(42)
    
    best_result = None
    best_score = np.inf
    
    # Run genetic algorithm
    ga_time = min(time_limit - 15.0, 100.0)
    
    try:
        best_chrom, best_f = _genetic_algorithm(
            problem, wavelengths_nm, thickness_options,
            max_layer_nm, max_total_nm,
            pop_size=50, max_iter=150,
            time_limit=ga_time, start_time=start_time
        )
        
        labels, thicknesses = _decode_chromosome(best_chrom, problem, thickness_options)
        
        # Check constraints
        thicknesses = np.clip(thicknesses, 0, max_layer_nm)
        if np.sum(thicknesses) > max_total_nm:
            thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
        
        # Evaluate
        indices = [materials[label] for label in labels]
        R = _compute_reflectance(wavelengths_nm, indices, thicknesses,
                                 ambient_idx, substrate_idx)
        score = np.mean(R) * 100.0
        
        if score < best_score:
            best_score = score
            best_result = (labels, thicknesses.tolist())
            
    except Exception:
        pass
    
    # Local refinement of best result
    if best_result is not None and time.time() - start_time < time_limit - 5.0:
        labels, thicknesses = best_result
        try:
            refined_labels, refined_thicknesses = _local_refine(
                labels, thicknesses, problem, wavelengths_nm,
                time_limit=10.0, start_time=start_time
            )
            
            indices = [materials[label] for label in refined_labels]
            R = _compute_reflectance(wavelengths_nm, indices, refined_thicknesses,
                                     ambient_idx, substrate_idx)
            score = np.mean(R) * 100.0
            
            if score < best_score:
                best_score = score
                best_result = (refined_labels, refined_thicknesses)
                
        except Exception:
            pass
    
    # Fallback: try a few known good structures
    if best_result is None or time.time() - start_time < time_limit - 10.0:
        center_wl = 0.5 * (band_nm[0] + band_nm[1])
        
        # Try 9-layer quarter-wave stack at center wavelength
        n_layers = 9
        labels = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        thicknesses = []
        for label in labels:
            n = materials[label]
            d = center_wl / (4.0 * n)
            thicknesses.append(d)
        
        thicknesses = np.clip(thicknesses, 0, max_layer_nm)
        if np.sum(thicknesses) > max_total_nm:
            thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
        
        indices = [materials[label] for label in labels]
        R = _compute_reflectance(wavelengths_nm, indices, thicknesses,
                                 ambient_idx, substrate_idx)
        score = np.mean(R) * 100.0
        
        if score < best_score:
            best_score = score
            best_result = (labels, thicknesses.tolist())
    
    if best_result is None:
        return ["L"], [center_wl / (4.0 * materials["L"])]
    
    labels, thicknesses = best_result
    
    # Ensure constraints
    thicknesses = np.clip(thicknesses, 0, max_layer_nm)
    if np.sum(thicknesses) > max_total_nm:
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
    
    return labels, thicknesses.tolist()
