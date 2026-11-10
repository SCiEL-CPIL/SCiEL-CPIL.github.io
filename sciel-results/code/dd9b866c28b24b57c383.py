"""
Broadband AR coating design using a Genetic Algorithm (GA) over both 
material sequences and thicknesses, with a fast vectorized reflectance engine.
"""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _reflectance_fast(mat_indices, thicknesses, n0, ns, n_mat, lambdas):
    """
    Vectorized reflectance calculation for a single stack across all wavelengths.
    
    Parameters:
    - mat_indices: list of indices into n_mat (e.g., [0, 1, 0, 1] for L, H, L, H)
    - thicknesses: list of thicknesses in nm
    - n0: ambient index
    - ns: substrate index
    - n_mat: array of material indices [n_L, n_H]
    - lambdas: array of wavelengths
    
    Returns:
    - array of reflectance values (0 to 1)
    """
    N = len(thicknesses)
    if N == 0:
        r = (n0 - ns) / (n0 + ns)
        return np.full_like(lambdas, np.abs(r)**2)
        
    # Initialize M matrix components
    M11 = np.ones(len(lambdas), dtype=complex)
    M12 = np.zeros(len(lambdas), dtype=complex)
    M21 = np.zeros(len(lambdas), dtype=complex)
    M22 = np.ones(len(lambdas), dtype=complex)
    
    # Iterate layers from air side to substrate side
    for i in range(N):
        n_i = n_mat[mat_indices[i]]
        d_i = thicknesses[i]
        
        delta = 2.0 * np.pi * n_i * d_i / lambdas
        c = np.cos(delta)
        s = np.sin(delta)
        
        # Layer matrix M_i
        m11 = c
        m12 = 1j * s / n_i
        m21 = 1j * n_i * s
        m22 = c
        
        # Multiply M = M * M_i
        new_M11 = M11 * m11 + M12 * m21
        new_M12 = M11 * m12 + M12 * m22
        new_M21 = M21 * m11 + M22 * m21
        new_M22 = M21 * m12 + M22 * m22
        
        M11 = new_M11
        M12 = new_M12
        M21 = new_M21
        M22 = new_M22
        
    # Input admittance
    Y_in = (M21 + M22 * ns) / (M11 + M12 * ns)
    
    # Reflectance
    r = (Y_in - n0) / (Y_in + n0)
    R = np.abs(r)**2
    
    return R

def design(problem):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    band = problem["band_nm"]
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    materials = problem["materials"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Wavelengths
    lambdas = np.linspace(band[0], band[1], 401)
    
    # Material indices mapping
    # L: 1.38, H: 2.35
    n_mat = np.array([materials['L'], materials['H']])
    # Map labels to indices: L->0, H->1
    
    # Strategy:
    # 1. Initialize population with diverse random stacks + some heuristic quarter-wave stacks.
    # 2. Use a Genetic Algorithm to evolve both the sequence of materials and the thicknesses.
    # 3. Constraints: sum(thicknesses) <= max_total_nm, each thickness <= max_layer_nm.
    #    We will penalize violations heavily or repair them.
    #    Repairing is safer: if sum > max, scale down. If single > max, clip.
    
    pop_size = 100
    max_generations = 200
    mutation_rate = 0.1
    crossover_rate = 0.8
    
    # Initialize Population
    # Each individual: (seq, thicks)
    # seq: list of 0s and 1s (0=L, 1=H)
    # thicks: list of floats
    
    population = []
    fitness = []
    
    def evaluate_individual(seq, thicks):
        # Repair constraints
        thicks = np.array(thicks, dtype=float)
        thicks = np.clip(thicks, 0, max_layer_nm)
        total = np.sum(thicks)
        if total > max_total_nm:
            thicks = thicks * (max_total_nm / total)
            
        R = _reflectance_fast(seq, thicks, n0, ns, n_mat, lambdas)
        mean_R = np.mean(R) * 100.0
        return mean_R, thicks

    # Heuristic initialization: Quarter-wave stacks
    center_lambda = 0.5 * (band[0] + band[1])
    qw_L = center_lambda / (4.0 * n_mat[0])
    qw_H = center_lambda / (4.0 * n_mat[1])
    
    # Add a few good heuristic starts
    for depth in [4, 6, 8, 10, 12]:
        if depth > max_layers:
            continue
        # Alternating L, H
        seq = [i % 2 for i in range(depth)] # 0=L, 1=H
        thicks = [qw_L if i % 2 == 0 else qw_H for i in range(depth)]
        f, thicks = evaluate_individual(seq, thicks)
        population.append((seq, thicks))
        fitness.append(f)
        
    # Random initialization for rest
    while len(population) < pop_size:
        depth = np.random.randint(3, max_layers + 1)
        seq = np.random.randint(0, 2, depth).tolist()
        thicks = np.random.uniform(0, max_layer_nm, depth)
        f, thicks = evaluate_individual(seq, thicks)
        population.append((seq, thicks))
        fitness.append(f)
    
    best_idx = np.argmin(fitness)
    best_seq = population[best_idx][0]
    best_thicks = population[best_idx][1]
    best_f = fitness[best_idx]
    
    # GA Loop
    for gen in range(max_generations):
        if time.time() > deadline:
            break
            
        # Evaluate all (already done for initial, but need to re-eval if we modify)
        # Actually, we store fitness. We only re-eval new children.
        
        # Selection: Tournament Selection
        # Crossover: Uniform crossover for seq, Blend crossover for thicks
        # Mutation: Random change in seq (flip), Random change in thicks (add noise)
        
        new_population = []
        new_fitness = []
        
        # Elitism: keep top 5
        sorted_indices = np.argsort(fitness)
        for i in range(min(5, len(population))):
            idx = sorted_indices[i]
            new_population.append(population[idx])
            new_fitness.append(fitness[idx])
            
        # Generate offspring
        while len(new_population) < pop_size:
            # Select parents
            p1_idx = np.random.choice(pop_size, 1, p=1.0/np.arange(1, pop_size+1))[0] if pop_size > 1 else 0
            # Simpler: Tournament
            t1 = np.random.choice(pop_size, 2, replace=False)
            p1_idx = t1[np.argmin(fitness[t1])]
            
            t2 = np.random.choice(pop_size, 2, replace=False)
            p2_idx = t2[np.argmin(fitness[t2])]
            
            parent1 = population[p1_idx]
            parent2 = population[p2_idx]
            
            # Handle different lengths
            len1 = len(parent1[0])
            len2 = len(parent2[0])
            
            if np.random.rand() < crossover_rate:
                # Crossover
                # For simplicity, if lengths differ, we might just take one parent's structure 
                # or truncate/pad. 
                # Better: Define crossover that handles length.
                # Let's pick a random length for child? Or inherit from one parent?
                # To keep it simple and robust, let's use "Single Point Crossover" 
                # but since lengths vary, it's tricky.
                # Alternative: Use "Uniform Crossover" on a fixed max length?
                # No, variable length is a feature.
                
                # Strategy: With 50% prob, take structure from P1, 50% from P2.
                # Then mix thicknesses?
                # Or: Create a child by sampling layers from P1 and P2.
                
                # Let's do this: 
                # 1. Decide child length: min, max, or average? 
                #    Let's use the length of the better parent.
                if fitness[p1_idx] <= fitness[p2_idx]:
                    base_len = len1
                    base_seq = parent1[0]
                    base_thicks = parent1[1]
                else:
                    base_len = len2
                    base_seq = parent2[0]
                    base_thicks = parent2[1]
                
                # Mix thicknesses from the other parent if possible
                other_thicks = parent2[1] if base_len == len1 else parent1[1]
                # If lengths match, blend
                if base_len == len(other_thicks):
                    alpha = np.random.rand()
                    child_thicks = alpha * np.array(base_thicks) + (1-alpha) * np.array(other_thicks)
                else:
                    child_thicks = np.array(base_thicks)
                    
                child_seq = list(base_seq)
            else:
                # No crossover, just copy better parent
                if fitness[p1_idx] <= fitness[p2_idx]:
                    child_seq = list(parent1[0])
                    child_thicks = np.array(parent1[1])
                else:
                    child_seq = list(parent2[0])
                    child_thicks = np.array(parent2[1])
                    
            # Mutation
            # 1. Thickness Mutation: Add Gaussian noise
            noise = np.random.normal(0, 20.0, len(child_thicks))
            child_thicks = child_thicks + noise
            
            # 2. Sequence Mutation: Flip bits with probability mutation_rate
            for i in range(len(child_seq)):
                if np.random.rand() < mutation_rate:
                    child_seq[i] = 1 - child_seq[i]
            
            # 3. Length Mutation: Add or remove a layer
            if np.random.rand() < 0.1:
                if len(child_seq) < max_layers and np.random.rand() < 0.5:
                    # Add layer
                    new_mat = np.random.randint(0, 2)
                    new_thick = np.random.uniform(10, 100)
                    pos = np.random.randint(0, len(child_seq) + 1)
                    child_seq.insert(pos, new_mat)
                    child_thicks = np.insert(child_thicks, pos, new_thick)
                elif len(child_seq) > 2:
                    # Remove layer
                    pos = np.random.randint(0, len(child_seq))
                    child_seq.pop(pos)
                    child_thicks = np.delete(child_thicks, pos)
            
            # Evaluate
            f, repaired_thicks = evaluate_individual(child_seq, child_thicks)
            
            new_population.append((child_seq, repaired_thicks))
            new_fitness.append(f)
            
        # Replace population
        population = new_population
        fitness = np.array(new_fitness)
        
        # Update best
        gen_best_idx = np.argmin(fitness)
        if fitness[gen_best_idx] < best_f:
            best_f = fitness[gen_best_idx]
            best_seq = population[gen_best_idx][0]
            best_thicks = population[gen_best_idx][1]
            
    # Local Refinement on Best
    # Do a few steps of coordinate descent on the best solution
    if time.time() < deadline - 10:
        for step in range(50):
            improved = False
            for i in range(len(best_seq)):
                # Try small perturbations in thickness
                original_t = best_thicks[i]
                best_t = original_t
                best_f_local = best_f
                
                # Binary search for optimal thickness in range [max(0, original-50), min(max_layer, original+50)]
                # Or just try a few values
                candidates = [original_t, original_t + 5, original_t - 5, original_t + 10, original_t - 10]
                candidates = [c for c in candidates if 0 <= c <= max_layer_nm]
                
                for c in candidates:
                    # Check total thickness constraint
                    # If changing this layer violates total, skip or scale
                    # For simplicity, just evaluate and let evaluate_individual repair
                    # But repair might distort other layers. 
                    # Let's just evaluate the raw stack and check constraint manually for speed?
                    # No, use the fast evaluator with repair.
                    
                    temp_thicks = list(best_thicks)
                    temp_thicks[i] = c
                    f_temp, _ = evaluate_individual(best_seq, temp_thicks)
                    if f_temp < best_f_local:
                        best_f_local = f_temp
                        best_t = c
                        
                if best_t != original_t:
                    best_thicks[i] = best_t
                    best_f = best_f_local
                    improved = True
                    
            if not improved:
                break
                
    # Final Result
    # Map indices back to 'L' and 'H'
    final_materials = ['L' if m == 0 else 'H' for m in best_seq]
    final_thicknesses = [float(t) for t in best_thicks]
    
    return final_materials, final_thicknesses

# EVOLVE-BLOCK-END
