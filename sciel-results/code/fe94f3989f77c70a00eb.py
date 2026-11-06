"""Program for broadband antireflection coating design using a Hybrid GA with Local Refinement."""
import numpy as np
import time
import random
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _compute_reflectances_vectorized(wavelengths_nm, ambient_idx, substrate_idx,
                                     material_indices, thicknesses_nm):
    """
    Compute normal-incidence reflectance (fraction, not percent) for each wavelength
    using the characteristic-matrix method, vectorized over wavelengths.
    
    Parameters:
    wavelengths_nm: array of wavelengths in nm (shape: N)
    ambient_idx: refractive index of ambient (air)
    substrate_idx: refractive index of substrate (glass)
    material_indices: list of refractive indices for each layer (from air to substrate)
    thicknesses_nm: list of physical thicknesses in nm for each layer
    
    Returns:
    array of reflectance values (0 to 1) for each wavelength (shape: N)
    """
    n_layers = len(material_indices)
    if n_layers == 0:
        # Bare glass
        r = (ambient_idx - substrate_idx) / (ambient_idx + substrate_idx)
        return np.full_like(wavelengths_nm, np.abs(r)**2)
    
    # Precompute constants for each layer
    # delta_j = 2*pi*n_j*d_j / lambda
    # We need to compute M = prod(L_j) for each wavelength
    # L_j = [[cos(delta), i*sin(delta)/n], [i*n*sin(delta), cos(delta)]]
    
    # To vectorize, we can't easily use matrix multiplication across the batch dimension 
    # unless we use einsum or loop over layers. Since N_layers is small (<20) and N_wavelengths is 401,
    # a loop over layers with vectorized operations over wavelengths is efficient.
    
    M00 = np.ones_like(wavelengths_nm, dtype=complex)
    M01 = np.zeros_like(wavelengths_nm, dtype=complex)
    M10 = np.zeros_like(wavelengths_nm, dtype=complex)
    M11 = np.ones_like(wavelengths_nm, dtype=complex)
    
    for j in range(n_layers):
        n_j = material_indices[j]
        d_j = thicknesses_nm[j]
        
        # delta for all wavelengths
        delta = 2.0 * np.pi * n_j * d_j / wavelengths_nm
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        # Layer matrix elements
        L00 = cos_d
        L01 = 1j * sin_d / n_j
        L10 = 1j * n_j * sin_d
        L11 = cos_d
        
        # Matrix multiplication: M_new = M_old @ L
        # M_old is [[M00, M01], [M10, M11]]
        # L is [[L00, L01], [L10, L11]]
        
        new_M00 = M00 * L00 + M01 * L10
        new_M01 = M00 * L01 + M01 * L11
        new_M10 = M10 * L00 + M11 * L10
        new_M11 = M10 * L01 + M11 * L11
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
        
    # Admittance Y = (M10 + M11 * n_sub) / (M00 + M01 * n_sub)
    n_sub = substrate_idx
    denom = M00 + M01 * n_sub
    
    # Handle potential division by zero (unlikely but safe)
    denom[abs(denom) < 1e-15] = 1e-15
    
    Y = (M10 + M11 * n_sub) / denom
    r = (ambient_idx - Y) / (ambient_idx + Y)
    reflectances = np.abs(r) ** 2
    
    return reflectances

def _evaluate_candidate(materials, thicknesses, problem):
    """
    Evaluate a candidate design. Returns mean reflectance in percent, or infinity if invalid.
    """
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    band_nm = problem.get("band_nm", [400, 800])
    materials_dict = problem.get("materials", {"L": 1.38, "H": 2.35})
    ambient_index = problem.get("ambient_index", 1.0)
    substrate_index = problem.get("substrate_index", 1.52)
    
    # Check constraints
    if len(materials) > max_layers:
        return float('inf')
    if len(materials) != len(thicknesses):
        return float('inf')
    
    total_thickness = sum(thicknesses)
    if total_thickness > max_total_nm:
        return float('inf')
    
    for t in thicknesses:
        if t < 0 or t > max_layer_nm:
            return float('inf')
            
    if len(materials) == 0:
        # Bare glass reflectance
        r = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return 100.0 * r ** 2
    
    material_indices = []
    for m in materials:
        if m not in materials_dict:
            return float('inf')
        material_indices.append(materials_dict[m])
    
    # Compute reflectance over 401 equally spaced wavelengths
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    reflectances = _compute_reflectances_vectorized(wavelengths, ambient_index, substrate_index,
                                                    material_indices, thicknesses)
    mean_R = np.mean(reflectances)
    return 100.0 * mean_R

def _local_refine(materials, thicknesses, problem, max_iter=50):
    """
    Refine thicknesses using scipy.optimize.minimize (L-BFGS-B).
    """
    if len(materials) == 0:
        return materials, thicknesses, _evaluate_candidate(materials, thicknesses, problem)
        
    max_layer_nm = problem.get("max_layer_nm", 400)
    max_total_nm = problem.get("max_total_nm", 1000)
    materials_dict = problem.get("materials", {"L": 1.38, "H": 2.35})
    ambient_index = problem.get("ambient_index", 1.0)
    substrate_index = problem.get("substrate_index", 1.52)
    band_nm = problem.get("band_nm", [400, 800])
    
    material_indices = [materials_dict[m] for m in materials]
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    # Objective function for scipy
    def objective(thick_array):
        # Check total thickness constraint
        if np.sum(thick_array) > max_total_nm:
            return float('inf')
        # Check individual layer constraints (bounds handle this, but just in case)
        if np.any(thick_array < 0) or np.any(thick_array > max_layer_nm):
            return float('inf')
            
        reflectances = _compute_reflectances_vectorized(wavelengths, ambient_index, substrate_index,
                                                        material_indices, thick_array)
        return np.mean(reflectances) * 100.0
        
    x0 = np.array(thicknesses, dtype=float)
    bounds = [(0.0, max_layer_nm) for _ in range(len(materials))]
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-8})
        if res.success and res.fun < float('inf'):
            new_thick = list(res.x)
            # Ensure no negative zeros or slight violations
            new_thick = [max(0.0, min(t, max_layer_nm)) for t in new_thick]
            # Check total thickness again
            if sum(new_thick) <= max_total_nm:
                return materials, new_thick, res.fun
    except Exception:
        pass
        
    return materials, thicknesses, _evaluate_candidate(materials, thicknesses, problem)

def design(problem):
    """
    Design a broadband antireflection coating using a Genetic Algorithm with Local Refinement.
    """
    start_time = time.time()
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    band_nm = problem.get("band_nm", [400, 800])
    materials_dict = problem.get("materials", {"L": 1.38, "H": 2.35})
    ambient_index = problem.get("ambient_index", 1.0)
    substrate_index = problem.get("substrate_index", 1.52)
    
    # GA Parameters
    pop_size = 60
    max_generations = 100
    elite_size = 5
    crossover_rate = 0.8
    mutation_rate = 0.2
    
    # Thickness ranges for initialization and mutation
    # We will use continuous thicknesses in the GA, but initialize with reasonable values.
    # Typical AR coating thicknesses are around lambda/4n.
    # For 600nm center: L: ~109nm, H: ~64nm.
    # Range: [10, 200] for L, [10, 150] for H.
    
    def random_individual():
        n_layers = random.randint(1, min(max_layers, 10)) # Start with smaller stacks
        mats = []
        thicks = []
        for _ in range(n_layers):
            m = random.choice(['L', 'H'])
            if m == 'L':
                t = random.uniform(20, 180)
            else:
                t = random.uniform(20, 140)
            mats.append(m)
            thicks.append(t)
        return mats, thicks
    
    # Initialize population
    population = []
    # Add some heuristic individuals
    # 1. Single layer quarter wave
    population.append((['L'], [600 / (4 * materials_dict['L'])]))
    population.append((['H'], [600 / (4 * materials_dict['H'])]))
    # 2. Two layer quarter wave
    population.append((['L', 'H'], [600 / (4 * materials_dict['L']), 600 / (4 * materials_dict['H'])]))
    population.append((['H', 'L'], [600 / (4 * materials_dict['H']), 600 / (4 * materials_dict['L'])]))
    
    while len(population) < pop_size:
        population.append(random_individual())
        
    # Evaluate initial population
    fitnesses = []
    for ind in population:
        mats, thicks = ind
        f = _evaluate_candidate(mats, thicks, problem)
        fitnesses.append(f)
        
    best_fitness = min(fitnesses)
    best_idx = fitnesses.index(best_fitness)
    best_individual = population[best_idx]
    
    # GA Loop
    for gen in range(max_generations):
        # Check time
        if time.time() - start_time > 50: # Leave 10s for final refinement
            break
            
        # Sort by fitness
        indexed = list(enumerate(fitnesses))
        indexed.sort(key=lambda x: x[1])
        sorted_pop = [population[i] for i, _ in indexed]
        sorted_fit = [f for _, f in indexed]
        
        # Update best
        if sorted_fit[0] < best_fitness:
            best_fitness = sorted_fit[0]
            best_individual = sorted_pop[0]
            
        # Elite selection
        new_population = [sorted_pop[i] for i in range(min(elite_size, len(sorted_pop)))]
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            # Tournament selection
            def tournament_select(pop, fit, k=3):
                indices = random.sample(range(len(pop)), min(k, len(pop)))
                best_i = indices[fit[indices].argmin()]
                return pop[best_i]
                
            parent1 = tournament_select(sorted_pop, sorted_fit)
            parent2 = tournament_select(sorted_pop, sorted_fit)
            
            # Crossover
            if random.random() < crossover_rate:
                child1, child2 = _crossover(parent1, parent2, max_layers)
            else:
                child1 = (list(parent1[0]), list(parent1[1]))
                child2 = (list(parent2[0]), list(parent2[1]))
            
            # Mutation
            child1 = _mutate(child1, mutation_rate, max_layers, materials_dict)
            child2 = _mutate(child2, mutation_rate, max_layers, materials_dict)
            
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)
                
        population = new_population
        fitnesses = [_evaluate_candidate(m, t, problem) for m, t in population]
        
        # Local Refinement on top candidates every 5 generations
        if gen % 5 == 0 and gen > 0:
            # Take top 3 individuals and refine them
            indexed = list(enumerate(fitnesses))
            indexed.sort(key=lambda x: x[1])
            top_indices = [i for i, _ in indexed[:3]]
            
            for idx in top_indices:
                m, t = population[idx]
                new_m, new_t, new_f = _local_refine(m, t, problem)
                if new_f < fitnesses[idx]:
                    population[idx] = (new_m, new_t)
                    fitnesses[idx] = new_f
                    if new_f < best_fitness:
                        best_fitness = new_f
                        best_individual = (new_m, new_t)

    # Final Refinement on the best individual found
    best_m, best_t = best_individual
    if len(best_m) > 0:
        final_m, final_t, final_f = _local_refine(best_m, best_t, problem, max_iter=100)
        if final_f < best_fitness:
            best_individual = (final_m, final_t)
            best_fitness = final_f
            
    return best_individual

def _crossover(parent1, parent2, max_layers):
    """
    Single-point crossover for variable-length chromosomes.
    """
    mat1, t1 = parent1
    mat2, t2 = parent2
    
    len1 = len(mat1)
    len2 = len(mat2)
    
    if max(len1, len2) <= 1:
        return (list(mat1), list(t1)), (list(mat2), list(t2))
        
    cp = random.randint(1, max(len1, len2))
    
    child1_mat = list(mat1[:cp])
    child1_t = list(t1[:cp])
    if len2 > cp:
        child1_mat.extend(mat2[cp:])
        child1_t.extend(t2[cp:])
        
    child2_mat = list(mat2[:cp])
    child2_t = list(t2[:cp])
    if len1 > cp:
        child2_mat.extend(mat1[cp:])
        child2_t.extend(t1[cp:])
        
    child1_mat = child1_mat[:max_layers]
    child1_t = child1_t[:max_layers]
    child2_mat = child2_mat[:max_layers]
    child2_t = child2_t[:max_layers]
    
    if len(child1_mat) == 0:
        child1_mat = ['L']
        child1_t = [100.0]
    if len(child2_mat) == 0:
        child2_mat = ['H']
        child2_t = [100.0]
        
    return (child1_mat, child1_t), (child2_mat, child2_t)

def _mutate(individual, mutation_rate, max_layers, materials_dict):
    """
    Mutate an individual: change material, change thickness, add/remove layer.
    """
    materials = list(individual[0])
    thicknesses = list(individual[1])
    
    n_layers = len(materials)
    
    for i in range(n_layers):
        if random.random() < mutation_rate:
            if random.random() < 0.5:
                # Change material
                materials[i] = 'L' if materials[i] != 'L' else 'H'
                # Reset thickness to a random value appropriate for the new material
                if materials[i] == 'L':
                    thicknesses[i] = random.uniform(20, 180)
                else:
                    thicknesses[i] = random.uniform(20, 140)
            else:
                # Change thickness
                # Small perturbation or random reset
                if random.random() < 0.5:
                    thicknesses[i] = max(10, min(max_layer_nm, thicknesses[i] * random.uniform(0.8, 1.2)))
                else:
                    if materials[i] == 'L':
                        thicknesses[i] = random.uniform(20, 180)
                    else:
                        thicknesses[i] = random.uniform(20, 140)
                        
    # Add a layer
    if random.random() < mutation_rate * 0.5 and n_layers < max_layers:
        mat = random.choice(['L', 'H'])
        if mat == 'L':
            t = random.uniform(20, 180)
        else:
            t = random.uniform(20, 140)
        pos = random.randint(0, n_layers)
        materials.insert(pos, mat)
        thicknesses.insert(pos, t)
        
    # Remove a layer
    elif random.random() < mutation_rate * 0.5 and n_layers > 1:
        pos = random.randint(0, n_layers)
        materials.pop(pos)
        thicknesses.pop(pos)
        
    # Clamp thicknesses
    for i in range(len(thicknesses)):
        thicknesses[i] = max(0.0, min(max_layer_nm, thicknesses[i]))
        
    return (materials, thicknesses)

# EVOLVE-BLOCK-END
