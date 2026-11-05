import time
import math
import numpy as np
import random

def _reflectance_array(materials, thicknesses, n_ambient, n_substrate, wls, n_mat):
    """
    Compute reflectance array for given materials and thicknesses.
    """
    if len(materials) == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return np.full(len(wls), r * r)

    n_layers = len(materials)
    n_wls = len(wls)
    
    # Precompute beta = 2*pi*n*d/wl for all layers and wavelengths
    beta = np.zeros((n_layers, n_wls))
    for i, m in enumerate(materials):
        n = n_mat[m]
        d = thicknesses[i]
        if d == 0:
            continue
        beta[i, :] = 2.0 * math.pi * n * d / wls

    cosb = np.cos(beta)
    sinb = np.sin(beta)

    M11 = np.ones(n_wls, dtype=complex)
    M12 = np.zeros(n_wls, dtype=complex)
    M21 = np.zeros(n_wls, dtype=complex)
    M22 = np.ones(n_wls, dtype=complex)

    for i, m in enumerate(materials):
        n = n_mat[m]
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

def _nelder_mead(objective, x0, bounds, deadline, max_iter=200, tol=1e-5):
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
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
            
        order = np.argsort(f_vals)
        simplex = [simplex[i] for i in order]
        f_vals = [f_vals[i] for i in order]
        
        if abs(f_vals[0] - f_vals[-1]) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        x_r = np.clip(centroid + alpha * (centroid - simplex[-1]), lo, hi)
        f_r = objective(x_r)
        
        if f_vals[0] <= f_r < f_vals[-2]:
            simplex[-1] = x_r
            f_vals[-1] = f_r
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
            continue
            
        x_c = np.clip(centroid + rho * (simplex[-1] - centroid), lo, hi)
        f_c = objective(x_c)
        if f_c < f_vals[-1]:
            simplex[-1] = x_c
            f_vals[-1] = f_c
            continue
            
        for i in range(1, n + 1):
            simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), lo, hi)
            f_vals[i] = objective(simplex[i])
            
    order = np.argsort(f_vals)
    best_idx = order[0]
    return np.clip(simplex[best_idx], lo, hi), f_vals[best_idx]

def _ga_optimize(objective, x0, bounds, deadline, pop_size=30, max_gen=50):
    """
    Genetic algorithm optimization.
    """
    x0 = np.array(x0, dtype=float)
    n = len(x0)
    lo, hi = bounds
    
    # Initialize population
    population = [x0.copy()]
    for _ in range(pop_size - 1):
        x = np.random.uniform(lo, hi)
        population.append(x)
    
    # Evaluate initial population
    f_vals = [objective(x) for x in population]
    
    for gen in range(max_gen):
        if time.time() > deadline:
            break
            
        # Sort by fitness
        order = np.argsort(f_vals)
        population = [population[i] for i in order]
        f_vals = [f_vals[i] for i in order]
        
        # Elitism: keep top 2
        new_population = [population[0].copy(), population[1].copy()]
        new_f_vals = [f_vals[0], f_vals[1]]
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            # Tournament selection
            def tournament():
                idx1 = np.random.randint(0, min(10, len(population)))
                idx2 = np.random.randint(0, min(10, len(population)))
                return population[idx1] if f_vals[idx1] < f_vals[idx2] else population[idx2]
            
            parent1 = tournament()
            parent2 = tournament()
            
            # Crossover (uniform)
            mask = np.random.rand(n) > 0.5
            child = np.where(mask, parent1, parent2)
            
            # Mutation
            for i in range(n):
                if np.random.rand() < 0.3:
                    child[i] += np.random.uniform(-0.1, 0.1) * (hi[i] - lo[i])
                    child[i] = np.clip(child[i], lo[i], hi[i])
            
            new_population.append(child)
        
        # Evaluate new population
        new_f_vals = [objective(x) for x in new_population]
        
        # Combine and select
        combined_pop = population + new_population
        combined_f = f_vals + new_f_vals
        
        order = np.argsort(combined_f)
        population = [combined_pop[i] for i in order[:pop_size]]
        f_vals = [combined_f[i] for i in order[:pop_size]]
        
    order = np.argsort(f_vals)
    best_idx = order[0]
    return np.clip(population[best_idx], lo, hi), f_vals[best_idx]

def design(problem):
    """
    Design a broadband antireflection coating.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
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
    
    # Define candidate material sequences
    sequences = []
    
    # Length 2
    sequences.append(['L', 'H'])
    sequences.append(['H', 'L'])
    
    # Length 4
    sequences.append(['L', 'H', 'L', 'H'])
    sequences.append(['H', 'L', 'H', 'L'])
    
    # Length 6
    sequences.append(['L', 'H', 'L', 'H', 'L', 'H'])
    sequences.append(['H', 'L', 'H', 'L', 'H', 'L'])
    
    # Length 8
    sequences.append(['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'])
    sequences.append(['H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'])
    
    # Length 10
    sequences.append(['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'])
    sequences.append(['H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'])
    
    # Length 12
    sequences.append(['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'])
    
    sequences = [seq for seq in sequences if len(seq) <= max_layers]
    
    # Generate initial points
    center_wls = [500, 550, 600, 650, 700]
    
    for mats in sequences:
        if time.time() > deadline:
            break
            
        n_layers = len(mats)
        lo = np.zeros(n_layers)
        hi = np.full(n_layers, max_layer)
        
        initial_guesses = []
        
        for cw in center_wls:
            ths = []
            for m in mats:
                n = n_mat[m]
                q = cw / (4.0 * n)
                q = q * random.uniform(0.8, 1.2)
                ths.append(q)
            initial_guesses.append(ths)
            
        # Add random guesses
        for _ in range(3):
            ths = [random.uniform(10, 200) for _ in range(n_layers)]
            initial_guesses.append(ths)
        
        for x0 in initial_guesses:
            if time.time() > deadline:
                break
                
            x0 = np.clip(x0, lo, hi)
            s = sum(x0)
            if s > max_total:
                x0 = x0 * (max_total / s)
            
            def objective(th, m=mats):
                return _mean_reflectance_percent(m, list(th), problem, wls, n_mat)
            
            # Try both GA and Nelder-Mead
            use_ga = random.random() < 0.5
            
            if use_ga:
                x_opt, f_opt = _ga_optimize(objective, x0, (lo, hi), deadline, pop_size=20, max_gen=30)
            else:
                x_opt, f_opt = _nelder_mead(objective, x0, (lo, hi), deadline, max_iter=150)
            
            x_opt = np.clip(x_opt, 0.0, max_layer)
            s = sum(x_opt)
            if s > max_total:
                x_opt = x_opt * (max_total / s)
                x_opt = np.clip(x_opt, 0.0, max_layer)
                
            val = _mean_reflectance_percent(mats, list(x_opt), problem, wls, n_mat)
            
            if val < best_val:
                best_val = val
                best_mats = mats
                best_ths = list(x_opt)
                
    if best_mats is None:
        best_mats = ['L']
        best_ths = [600.0 / (4.0 * n_L)]
        best_val = _mean_reflectance_percent(best_mats, best_ths, problem, wls, n_mat)
        
    best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
    if sum(best_ths) > max_total + 1e-6:
        f = max_total / sum(best_ths)
        best_ths = [t * f for t in best_ths]
        
    return list(best_mats), list(best_ths)
