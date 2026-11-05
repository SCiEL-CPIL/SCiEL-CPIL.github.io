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
    ds = np.array(thicknesses)
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

def _project_to_feasible(x, max_layer, max_total):
    """Project x to feasible region: [0, max_layer] per layer, sum <= max_total."""
    x = np.clip(x, 0, max_layer)
    total = np.sum(x)
    if total > max_total:
        if total > 1e-12:
            x = x * (max_total / total)
        else:
            x = np.zeros_like(x)
    return x

def _ga_optimize(problem, wls, n_mat, deadline, max_layers, max_layer, max_total, pop_size=50):
    """
    Genetic Algorithm to optimize the stack.
    Chromosome: [n_layers, mat_1, mat_2, ..., mat_n, th_1, th_2, ..., th_n]
    n_layers is an integer 0..max_layers.
    mat_i is 0 (L) or 1 (H).
    th_i is float 0..max_layer.
    """
    start_time = time.time()
    
    # Initialize population
    population = []
    for _ in range(pop_size):
        n_layers = random.randint(4, max_layers)
        mats = [random.randint(0, 1) for _ in range(n_layers)]
        ths = np.random.uniform(0, max_layer, n_layers)
        ths = _project_to_feasible(ths, max_layer, max_total)
        chromosome = [n_layers] + mats + list(ths)
        population.append(chromosome)
    
    best_val = float('inf')
    best_chrom = None
    
    def evaluate(chrom):
        n_layers = int(chrom[0])
        mats_idx = chrom[1:1+n_layers]
        ths = np.array(chrom[1+n_layers:1+n_layers+n_layers])
        
        # Map indices to materials
        mats = ['L' if m == 0 else 'H' for m in mats_idx]
        
        # Filter out zero thicknesses to potentially reduce layer count if needed, 
        # but GA structure keeps fixed length for crossover simplicity.
        # We just evaluate as is.
        val = _mean_reflectance_percent(mats, list(ths), problem, wls, n_mat)
        return val

    # Evaluate initial population
    scores = [evaluate(ch) for ch in population]
    
    for i in range(pop_size):
        if scores[i] < best_val:
            best_val = scores[i]
            best_chrom = population[i]
            
    # Main GA loop
    generation = 0
    while time.time() < deadline - 5:
        generation += 1
        
        # Selection: Tournament
        new_population = []
        
        # Elitism: Keep the best individual
        best_idx = np.argmin(scores)
        new_population.append(population[best_idx])
        
        while len(new_population) < pop_size:
            # Pick two parents
            p1_idx = random.randint(0, pop_size - 1)
            p2_idx = random.randint(0, pop_size - 1)
            
            # Tournament selection
            if random.random() < 0.5:
                p1_idx = min(p1_idx, p2_idx, key=lambda i: scores[i])
            else:
                p2_idx = min(p1_idx, p2_idx, key=lambda i: scores[i])
            
            # Actually, let's do proper tournament
            def tournament_select():
                t1 = random.randint(0, pop_size - 1)
                t2 = random.randint(0, pop_size - 1)
                if scores[t1] < scores[t2]:
                    return t1
                else:
                    return t2
            
            p1_idx = tournament_select()
            p2_idx = tournament_select()
            
            parent1 = population[p1_idx]
            parent2 = population[p2_idx]
            
            # Crossover: Uniform
            # Determine child length (n_layers)
            # We need to handle different lengths. 
            # Strategy: Child n_layers is from one of the parents or a blend?
            # Simpler: Child n_layers is random from [min(p1, p2), max(p1, p2)] or just one of them.
            # Let's pick n_layers from parent1 or parent2.
            if random.random() < 0.5:
                child_n_layers = parent1[0]
                # Sample materials from parent1
                child_mats = parent1[1:1+child_n_layers]
                # Sample thicknesses from parent1
                child_ths = parent1[1+child_n_layers:1+child_n_layers+child_n_layers]
            else:
                child_n_layers = parent2[0]
                child_mats = parent2[1:1+child_n_layers]
                child_ths = parent2[1+child_n_layers:1+child_n_layers+child_n_layers]
            
            # Mutation
            # 1. Change n_layers with low probability
            if random.random() < 0.1:
                delta_n = random.randint(-2, 2)
                new_n = max(1, min(max_layers, child_n_layers + delta_n))
                if new_n != child_n_layers:
                    # Adjust mats and ths
                    if new_n < child_n_layers:
                        child_mats = child_mats[:new_n]
                        child_ths = child_ths[:new_n]
                    else:
                        # Extend with random
                        ext_mats = [random.randint(0, 1) for _ in range(new_n - child_n_layers)]
                        ext_ths = np.random.uniform(0, max_layer, new_n - child_n_layers)
                        child_mats = child_mats + ext_mats
                        child_ths = list(child_ths) + list(ext_ths)
                    child_n_layers = new_n
            
            # 2. Mutate materials
            for i in range(len(child_mats)):
                if random.random() < 0.1:
                    child_mats[i] = 1 - child_mats[i]
            
            # 3. Mutate thicknesses
            child_ths = np.array(child_ths)
            for i in range(len(child_ths)):
                if random.random() < 0.1:
                    # Gaussian mutation
                    child_ths[i] += random.gauss(0, 50)
            
            child_ths = _project_to_feasible(child_ths, max_layer, max_total)
            
            child_chrom = [child_n_layers] + child_mats + list(child_ths)
            new_population.append(child_chrom)
        
        population = new_population
        scores = [evaluate(ch) for ch in population]
        
        for i in range(pop_size):
            if scores[i] < best_val:
                best_val = scores[i]
                best_chrom = population[i]
                
    # Decode best chromosome
    n_layers = int(best_chrom[0])
    mats_idx = best_chrom[1:1+n_layers]
    ths = np.array(best_chrom[1+n_layers:1+n_layers+n_layers])
    
    mats = ['L' if m == 0 else 'H' for m in mats_idx]
    
    # Cleanup: remove zero layers
    final_mats = []
    final_ths = []
    for m, t in zip(mats, ths):
        if t > 1e-6:
            final_mats.append(m)
            final_ths.append(float(t))
            
    return final_mats, final_ths, best_val

def design(problem):
    """
    Design a broadband antireflection coating using a Genetic Algorithm.
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

    best_mats, best_ths, best_val = _ga_optimize(
        problem, wls, n_mat, deadline, max_layers, max_layer, max_total, pop_size=60
    )
    
    # Fallback if GA failed or returned empty
    if not best_mats:
        best_mats = ['L']
        best_ths = [600.0 / (4.0 * n_L)]
        best_val = _mean_reflectance_percent(best_mats, best_ths, problem, wls, n_mat)
        
    # Final constraint check
    best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
    s = sum(best_ths)
    if s > max_total + 1e-6:
        f = max_total / s
        best_ths = [t * f for t in best_ths]
        best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
    
    return list(best_mats), list(best_ths)
