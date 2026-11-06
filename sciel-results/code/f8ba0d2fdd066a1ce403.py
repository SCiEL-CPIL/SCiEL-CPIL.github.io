"""Broadband antireflection coating design via genetic algorithm over discrete layer sequences."""
import numpy as np
import time
import math

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

def _evaluate_design(labels, thicknesses, problem, wavelengths_nm):
    """Evaluate a design and return mean reflectance in percent."""
    max_layer_nm = problem.get("max_layer_nm", 400)
    max_total_nm = problem.get("max_total_nm", 1000)
    
    thicknesses = np.array(thicknesses)
    if np.any(thicknesses < -1e-9) or np.any(thicknesses > max_layer_nm + 1e-9):
        return 100.0
    if np.sum(thicknesses) > max_total_nm + 1e-9:
        return 100.0
    
    indices = [problem["materials"][label] for label in labels]
    R = _compute_reflectance(wavelengths_nm, indices, thicknesses,
                             problem["ambient_index"], problem["substrate_index"])
    return np.mean(R) * 100.0

def _local_polish(labels, thicknesses, problem, wavelengths_nm, time_budget=5.0, start_time=None):
    """
    Local polish using Nelder-Mead on thicknesses with fixed layer sequence.
    """
    if start_time is None:
        start_time = time.time()
    
    max_layer_nm = problem.get("max_layer_nm", 400)
    max_total_nm = problem.get("max_total_nm", 1000)
    n = len(thicknesses)
    
    def obj(x):
        if np.any(x < -1e-9) or np.any(x > max_layer_nm + 1e-9):
            return 100.0
        if np.sum(x) > max_total_nm + 1e-9:
            return 100.0
        x = np.clip(x, 0, max_layer_nm)
        indices = [problem["materials"][label] for label in labels]
        R = _compute_reflectance(wavelengths_nm, indices, x,
                                 problem["ambient_index"], problem["substrate_index"])
        return np.mean(R) * 100.0
    
    x0 = np.array(thicknesses, dtype=float)
    
    # Initialize simplex
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    for i in range(n):
        simplex[i + 1] = x0.copy()
        step = 0.05 * max_layer_nm
        simplex[i + 1][i] += step
    
    fvals = np.array([obj(simplex[i]) for i in range(n + 1)])
    
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    max_iter = 300
    
    for it in range(max_iter):
        if time.time() - start_time > start_time + time_budget:
            break
        
        idx = np.argsort(fvals)
        simplex = simplex[idx]
        fvals = fvals[idx]
        
        if np.max(fvals) - np.min(fvals) < 1e-10:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        x_r = centroid + alpha * (centroid - simplex[-1])
        f_r = obj(x_r)
        
        if f_r < fvals[0]:
            x_e = centroid + gamma * (x_r - centroid)
            f_e = obj(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                fvals[-1] = f_e
            else:
                simplex[-1] = x_r
                fvals[-1] = f_r
        elif f_r < fvals[-1]:
            simplex[-1] = x_r
            fvals[-1] = f_r
        else:
            x_c = centroid + rho * (simplex[-1] - centroid)
            f_c = obj(x_c)
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = obj(simplex[i])
    
    best_idx = np.argmin(fvals)
    best_x = simplex[best_idx]
    best_x = np.clip(best_x, 0, max_layer_nm)
    if np.sum(best_x) > max_total_nm:
        best_x = best_x * (max_total_nm / np.sum(best_x))
    
    return best_x.tolist(), obj(best_x)

def design(problem):
    """
    Design a broadband antireflection coating using a genetic algorithm over
    discrete layer sequences (L/H) and continuous thickness optimization.
    """
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wavelengths_nm = np.linspace(band_nm[0], band_nm[1], 401)
    
    # GA parameters
    pop_size = 40
    max_generations = 100
    mutation_rate = 0.15
    crossover_rate = 0.8
    
    # Initialize population: random layer sequences with quarter-wave-ish thicknesses
    center_wl = 0.5 * (band_nm[0] + band_nm[1])
    
    def _random_design():
        n_layers = np.random.randint(4, max_layers + 1)
        labels = []
        for i in range(n_layers):
            labels.append('L' if np.random.rand() < 0.5 else 'H')
        indices = [materials[label] for label in labels]
        thicknesses = []
        for n in indices:
            # Quarter-wave at center wavelength with some random variation
            base = center_wl / (4.0 * n)
            t = base * np.random.uniform(0.5, 1.5)
            t = min(t, max_layer_nm)
            thicknesses.append(t)
        # Scale to fit total thickness constraint
        total = sum(thicknesses)
        if total > max_total_nm:
            scale = max_total_nm / total
            thicknesses = [t * scale for t in thicknesses]
        return labels, thicknesses
    
    population = []
    for _ in range(pop_size):
        labels, thicknesses = _random_design()
        score = _evaluate_design(labels, thicknesses, problem, wavelengths_nm)
        population.append((labels, thicknesses, score))
    
    best_labels, best_thicknesses, best_score = min(population, key=lambda x: x[2])
    
    for gen in range(max_generations):
        if time.time() - start_time > time_limit - 15:
            break
        
        # Sort by score
        population.sort(key=lambda x: x[2])
        
        # Elitism: keep top 2
        new_population = [population[0], population[1]]
        
        # Selection and crossover
        while len(new_population) < pop_size:
            # Tournament selection
            def _tournament():
                candidates = np.random.choice(len(population), size=3, replace=False)
                best = min(population[c] for c in candidates)
                return best
            
            parent1 = _tournament()
            parent2 = _tournament()
            
            if np.random.rand() < crossover_rate:
                # Crossover: combine layer sequences
                labels1, th1, _ = parent1
                labels2, th2, _ = parent2
                
                # Choose crossover point in the layer sequence
                max_len = max(len(labels1), len(labels2))
                xover_point = np.random.randint(1, max_len) if max_len > 1 else 0
                
                new_labels = []
                new_thicknesses = []
                for i in range(max_len):
                    if i < xover_point:
                        if i < len(labels1):
                            new_labels.append(labels1[i])
                            new_thicknesses.append(th1[i])
                    else:
                        if i < len(labels2):
                            new_labels.append(labels2[i])
                            new_thicknesses.append(th2[i])
                
                # Mutation
                if np.random.rand() < mutation_rate:
                    # Mutate: add, remove, or change a layer
                    mut_type = np.random.randint(3)
                    if mut_type == 0 and len(new_labels) < max_layers:
                        # Add a layer
                        pos = np.random.randint(len(new_labels) + 1)
                        new_labels.insert(pos, 'L' if np.random.rand() < 0.5 else 'H')
                        n_new = materials[new_labels[pos]]
                        new_thicknesses.insert(pos, center_wl / (4.0 * n_new) * np.random.uniform(0.5, 1.5))
                    elif mut_type == 1 and len(new_labels) > 2:
                        # Remove a layer
                        pos = np.random.randint(len(new_labels))
                        new_labels.pop(pos)
                        new_thicknesses.pop(pos)
                    else:
                        # Change a layer type
                        if new_labels:
                            pos = np.random.randint(len(new_labels))
                            new_labels[pos] = 'L' if new_labels[pos] == 'H' else 'H'
                
                # Randomize thicknesses slightly
                for i in range(len(new_thicknesses)):
                    new_thicknesses[i] *= np.random.uniform(0.9, 1.1)
                    new_thicknesses[i] = min(new_thicknesses[i], max_layer_nm)
                
                # Scale to fit constraint
                total = sum(new_thicknesses)
                if total > max_total_nm:
                    scale = max_total_nm / total
                    new_thicknesses = [t * scale for t in new_thicknesses]
            else:
                # No crossover: just copy parent1
                new_labels = parent1[0][:]
                new_thicknesses = parent1[1][:]
                # Slight mutation
                if new_thicknesses and np.random.rand() < mutation_rate:
                    pos = np.random.randint(len(new_thicknesses))
                    new_thicknesses[pos] *= np.random.uniform(0.8, 1.2)
                    new_thicknesses[pos] = min(new_thicknesses[pos], max_layer_nm)
                    total = sum(new_thicknesses)
                    if total > max_total_nm:
                        scale = max_total_nm / total
                        new_thicknesses = [t * scale for t in new_thicknesses]
            
            score = _evaluate_design(new_labels, new_thicknesses, problem, wavelengths_nm)
            new_population.append((new_labels, new_thicknesses, score))
        
        population = new_population
        
        # Update best
        gen_best = min(population, key=lambda x: x[2])
        if gen_best[2] < best_score:
            best_score = gen_best[2]
            best_labels = gen_best[0]
            best_thicknesses = gen_best[1]
        
        # Early stopping if very good
        if best_score < 0.3:
            break
    
    # Final polish on the best design
    remaining_time = time_limit - (time.time() - start_time) - 2
    if remaining_time > 3:
        try:
            polished_thicknesses, polished_score = _local_polish(
                best_labels, best_thicknesses, problem, wavelengths_nm,
                time_budget=min(remaining_time, 30.0), start_time=start_time
            )
            if polished_score < best_score:
                best_thicknesses = polished_thicknesses
                best_score = polished_score
        except Exception:
            pass
    
    # Ensure constraints
    best_thicknesses = [max(0, min(t, max_layer_nm)) for t in best_thicknesses]
    total = sum(best_thicknesses)
    if total > max_total_nm:
        scale = max_total_nm / total
        best_thicknesses = [t * scale for t in best_thicknesses]
    
    return best_labels, best_thicknesses
