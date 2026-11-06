"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_reflectance_vectorized(materials, thicknesses, n_air, n_sub, wavelengths):
    """
    Compute reflectance for a stack across all wavelengths in parallel.
    materials: list of 'L' or 'H'
    thicknesses: list of physical thicknesses in nm
    n_air: ambient index
    n_sub: substrate index
    wavelengths: array of wavelengths in nm
    """
    n_layers = len(materials)
    if n_layers == 0:
        r = (n_air - n_sub) / (n_air + n_sub)
        return np.full_like(wavelengths, abs(r)**2)
    
    # M_total is 2x2 matrix for each wavelength
    # Initialize as identity
    M_total = np.zeros((len(wavelengths), 2, 2), dtype=complex)
    M_total[:, 0, 0] = 1.0
    M_total[:, 1, 1] = 1.0
    
    # Pre-compute indices for speed
    n_vals = np.array([problem_global["materials"][m] for m in materials])
    
    # Iterate through layers
    for i in range(n_layers):
        n = n_vals[i]
        d = thicknesses[i]
        
        # delta = 2 * pi * n * d / lambda
        # wavelengths is shape (401,)
        delta = 2.0 * np.pi * n * d / wavelengths
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        # M_i for each wavelength
        # M_i = [[cos, i*sin/n], [i*n*sin, cos]]
        M_i = np.zeros((len(wavelengths), 2, 2), dtype=complex)
        M_i[:, 0, 0] = cos_d
        M_i[:, 0, 1] = 1j * sin_d / n
        M_i[:, 1, 0] = 1j * n * sin_d
        M_i[:, 1, 1] = cos_d
        
        # M_total = M_total @ M_i
        # (N, 2, 2) @ (N, 2, 2) -> need to do matrix multiplication for each N
        # np.einsum or loop? Loop is 20 iterations, N is 401. 
        # M_total_new[i] = M_total[i] @ M_i[i]
        # We can use np.einsum('nij,njk->nik', M_total, M_i)
        M_total = np.einsum('nij,njk->nik', M_total, M_i)
        
    # Calculate Y_in
    # Y_in = (M[1,0] + M[1,1]*n_sub) / (M[0,0] + M[0,1]*n_sub)
    # M_total shape (N, 2, 2)
    m11 = M_total[:, 0, 0]
    m12 = M_total[:, 0, 1]
    m21 = M_total[:, 1, 0]
    m22 = M_total[:, 1, 1]
    
    Y_in = (m21 + m22 * n_sub) / (m11 + m12 * n_sub)
    
    # r = (n_air - Y_in) / (n_air + Y_in)
    r = (n_air - Y_in) / (n_air + Y_in)
    
    R = np.abs(r)**2
    return R

problem_global = None

def design(problem):
    global problem_global
    problem_global = problem
    
    n_air = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    # Genetic Algorithm Parameters
    pop_size = 60
    max_generations = 200
    mutation_rate = 0.1
    crossover_rate = 0.7
    
    start_time = time.time()
    time_budget = 55.0 # Leave 5s margin
    
    # Initialize population
    # Each individual: (materials_list, thicknesses_list)
    # We fix length to max_layers to simplify GA operations, using 0 thickness for "empty"
    population = []
    for _ in range(pop_size):
        mats = [np.random.choice(['L', 'H']) for _ in range(max_layers)]
        thicks = np.random.uniform(0, max_layer_nm, size=max_layers)
        # Ensure total thickness constraint
        total_t = np.sum(thicks)
        if total_t > max_total_nm:
            thicks = thicks * (max_total_nm / total_t)
        population.append((mats, thicks))
        
    best_individual = None
    best_fitness = float('inf')
    
    def evaluate_individual(ind):
        mats, thicks = ind
        # Filter out zero thickness layers for efficiency? 
        # The vectorized function handles zeros fine, but 20 layers is small.
        # Actually, if many are 0, we can skip them to save time in einsum?
        # Let's just run it. 20 layers * 401 wavelengths is fast.
        R = _compute_reflectance_vectorized(mats, thicks, n_air, n_sub, wavelengths)
        return np.mean(R) * 100.0
        
    # Initial evaluation
    fitnesses = []
    for ind in population:
        f = evaluate_individual(ind)
        fitnesses.append(f)
        if f < best_fitness:
            best_fitness = f
            best_individual = ind
            
    for gen in range(max_generations):
        if time.time() - start_time > time_budget:
            break
            
        # Sort by fitness
        indexed_pop = sorted(zip(fitnesses, range(pop_size)))
        new_population = []
        
        # Elitism: keep top 2
        for f, idx in indexed_pop[:2]:
            new_population.append(population[idx])
            
        # Generate rest
        while len(new_population) < pop_size:
            # Tournament selection
            p1_idx = np.random.randint(0, pop_size)
            p2_idx = np.random.randint(0, pop_size)
            
            # Crossover
            if np.random.random() < crossover_rate:
                # Cut point
                cut = np.random.randint(1, max_layers)
                
                # Get parents (copy to avoid mutation issues)
                p1_mats, p1_thicks = population[p1_idx]
                p2_mats, p2_thicks = population[p2_idx]
                
                # Create child
                c_mats = p1_mats[:cut] + p2_mats[cut:]
                c_thicks = np.concatenate([p1_thicks[:cut], p2_thicks[cut:]])
            else:
                # Duplicate
                p1_mats, p1_thicks = population[p1_idx]
                c_mats = p1_mats.copy()
                c_thicks = p1_thicks.copy()
                
            # Mutation
            if np.random.random() < mutation_rate:
                # Randomly mutate some layers
                for i in range(max_layers):
                    if np.random.random() < 0.2: # Per-layer mutation prob
                        # Change material
                        if np.random.random() < 0.5:
                            c_mats[i] = 'L' if c_mats[i] == 'H' else 'H'
                        # Change thickness
                        if np.random.random() < 0.5:
                            # Gaussian perturbation
                            delta_t = np.random.normal(0, 50)
                            c_thicks[i] += delta_t
                            c_thicks[i] = np.clip(c_thicks[i], 0, max_layer_nm)
                        
                        # Sometimes set to 0 or random
                        if np.random.random() < 0.05:
                            c_thicks[i] = 0.0
                            
            # Constraint Handling
            total_t = np.sum(c_thicks)
            if total_t > max_total_nm:
                c_thicks = c_thicks * (max_total_nm / total_t)
                
            new_population.append((c_mats, c_thicks))
            
        # Evaluate new population
        new_fitnesses = []
        for ind in new_population:
            f = evaluate_individual(ind)
            new_fitnesses.append(f)
            if f < best_fitness:
                best_fitness = f
                best_individual = ind
                
        population = new_population
        fitnesses = new_fitnesses
        
    # Extract best solution
    # Remove zero thickness layers for output? 
    # The problem says "list of layers". Empty layers (0 thickness) are valid but redundant.
    # It's cleaner to remove them.
    mats, thicks = best_individual
    final_mats = []
    final_thicks = []
    for m, t in zip(mats, thicks):
        if t > 1e-6: # Threshold
            final_mats.append(m)
            final_thicks.append(float(t))
            
    # If all zero, return single layer fallback?
    # Or just return empty? 
    # "Design a stack of at most twenty... layers". Empty stack is valid? 
    # Bare glass is 4.26%. An AR coating should be better.
    # If GA failed, we might have 0 layers.
    if len(final_mats) == 0:
        # Fallback to a simple quarter-wave L layer
        lam_c = 0.5 * (band_nm[0] + band_nm[1])
        n_L = problem["materials"]["L"]
        d_L = lam_c / (4.0 * n_L)
        final_mats = ["L"]
        final_thicks = [d_L]
        
    return final_mats, final_thicks
# EVOLVE-BLOCK-END
