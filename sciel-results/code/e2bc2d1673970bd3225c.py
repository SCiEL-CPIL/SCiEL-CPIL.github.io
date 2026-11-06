# EVOLVE-BLOCK-START
import numpy as np
import random

def _compute_reflectance(materials, thicknesses, problem):
    """
    Compute mean reflectance in percent over the band.
    Uses characteristic matrix method.
    """
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    n_dict = problem["materials"]
    lam0, lam1 = problem["band_nm"]
    n_waves = 401
    lambdas = np.linspace(lam0, lam1, n_waves)
    
    # Pre-compute material indices and thicknesses arrays for faster access
    n_layers = len(materials)
    if n_layers == 0:
        # Bare glass
        r = (n0 - ns) / (n0 + ns)
        return 100.0 * r * r

    # Validate constraints inside to be safe, though GA should maintain them
    total_thick = sum(thicknesses)
    if total_thick > problem["max_total_nm"] or any(t < 0 or t > problem["max_layer_nm"] for t in thicknesses):
        return float('inf')
        
    n_vals = np.array([n_dict[m] for m in materials])
    d_vals = np.array(thicknesses)
    
    total_r = 0.0
    for lam in lambdas:
        # Initialize total matrix M
        A = 1.0 + 0j
        B = 0.0 + 0j
        C = 0.0 + 0j
        D = 1.0 + 0j
        
        for i in range(n_layers):
            n_i = n_vals[i]
            d_i = d_vals[i]
            delta = 2.0 * np.pi * n_i * d_i / lam
            cos_d = np.cos(delta)
            sin_d = np.sin(delta)
            
            # Layer matrix
            M11 = cos_d
            M12 = 1j * sin_d / n_i
            M21 = 1j * n_i * sin_d
            M22 = cos_d
            
            # M_new = M_old * M_layer
            nA = A * M11 + B * M21
            nB = A * M12 + B * M22
            nC = C * M11 + D * M21
            nD = C * M12 + D * M22
            
            A, B, C, D = nA, nB, nC, nD
        
        Y_in = (C + D * ns) / (A + B * ns)
        r = (n0 - Y_in) / (n0 + Y_in)
        R = np.abs(r) ** 2
        total_r += R
        
    return 100.0 * total_r / n_waves

def design(problem):
    """
    Design a stack using a Genetic Algorithm.
    """
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    pop_size = 50
    generations = 100
    mutation_rate = 0.1
    crossover_rate = 0.8
    
    # Initialize population
    population = []
    for _ in range(pop_size):
        # Random number of layers between 1 and max_layers
        n_layers = random.randint(1, max_layers)
        mats = [random.choice(['L', 'H']) for _ in range(n_layers)]
        # Random thicknesses, ensuring sum <= max_total_nm
        # Simple strategy: generate random weights, scale to fit
        weights = [random.random() for _ in range(n_layers)]
        total_weight = sum(weights)
        if total_weight == 0:
            thicks = [0.0] * n_layers
        else:
            # Leave some margin
            target_total = random.uniform(100, max_total_nm)
            thicks = [w / total_weight * target_total for w in weights]
            # Clip to max_layer_nm
            thicks = [min(t, max_layer_nm) for t in thicks]
            # If clipping caused sum to exceed, scale down
            s = sum(thicks)
            if s > max_total_nm:
                factor = max_total_nm / s
                thicks = [t * factor for t in thicks]
        
        population.append((mats, thicks))
    
    # Evaluate initial population
    scores = [_compute_reflectance(m, t, problem) for m, t in population]
    
    best_individual = min(zip(population, scores), key=lambda x: x[1])
    best_score = best_individual[1]
    
    for gen in range(generations):
        new_population = []
        
        # Elitism: keep the best individual
        new_population.append(best_individual[0])
        
        while len(new_population) < pop_size:
            # Selection: Tournament selection
            idx1 = random.randint(0, pop_size - 1)
            idx2 = random.randint(0, pop_size - 1)
            if scores[idx1] < scores[idx2]:
                parent1 = population[idx1]
            else:
                parent1 = population[idx2]
            
            idx3 = random.randint(0, pop_size - 1)
            idx4 = random.randint(0, pop_size - 1)
            if scores[idx3] < scores[idx4]:
                parent2 = population[idx3]
            else:
                parent2 = population[idx4]
            
            # Crossover
            if random.random() < crossover_rate:
                # Uniform crossover
                child_mats = []
                child_thicks = []
                # Take the max length of parents for structure, but we need to handle length changes
                # Simple approach: if lengths differ, truncate or pad? 
                # Better: Crossover on the sequence. If lengths differ, we can just pick the shorter one's structure or random.
                # Let's stick to fixed length crossover for simplicity in this GA, or allow length change via mutation.
                # For now, if lengths are different, we just pick one parent's structure.
                if len(parent1[0]) == len(parent2[0]):
                    for i in range(len(parent1[0])):
                        if random.random() < 0.5:
                            child_mats.append(parent1[0][i])
                            child_thicks.append(parent1[1][i])
                        else:
                            child_mats.append(parent2[0][i])
                            child_thicks.append(parent2[1][i])
                else:
                    # If lengths differ, just copy parent1
                    child_mats = list(parent1[0])
                    child_thicks = list(parent1[1])
            else:
                child_mats = list(parent1[0])
                child_thicks = list(parent1[1])
            
            # Mutation
            # 1. Change material
            if child_mats:
                for i in range(len(child_mats)):
                    if random.random() < mutation_rate:
                        child_mats[i] = 'L' if child_mats[i] == 'H' else 'H'
            
            # 2. Perturb thickness
            for i in range(len(child_thicks)):
                if random.random() < mutation_rate:
                    # Gaussian perturbation
                    perturbation = random.gauss(0, 20)
                    child_thicks[i] += perturbation
                    child_thicks[i] = max(0, min(child_thicks[i], max_layer_nm))
            
            # 3. Change number of layers (insertion/deletion)
            if random.random() < mutation_rate * 0.5:
                if random.random() < 0.5 and len(child_mats) < max_layers:
                    # Insert a layer
                    pos = random.randint(0, len(child_mats))
                    new_mat = random.choice(['L', 'H'])
                    new_thick = random.uniform(10, 100)
                    child_mats.insert(pos, new_mat)
                    child_thicks.insert(pos, new_thick)
                elif random.random() < 0.5 and len(child_mats) > 1:
                    # Delete a layer
                    pos = random.randint(0, len(child_mats) - 1)
                    child_mats.pop(pos)
                    child_thicks.pop(pos)
            
            # Constraint Repair: Ensure total thickness <= max_total_nm
            total_thick = sum(child_thicks)
            if total_thick > max_total_nm:
                factor = max_total_nm / total_thick
                child_thicks = [t * factor for t in child_thicks]
            
            new_population.append((child_mats, child_thicks))
        
        # Evaluate new population
        population = new_population
        scores = [_compute_reflectance(m, t, problem) for m, t in population]
        
        # Update best
        current_best_idx = np.argmin(scores)
        if scores[current_best_idx] < best_score:
            best_score = scores[current_best_idx]
            best_individual = (population[current_best_idx], scores[current_best_idx])
            
    return best_individual[0][0], best_individual[0][1]
# EVOLVE-BLOCK-END
