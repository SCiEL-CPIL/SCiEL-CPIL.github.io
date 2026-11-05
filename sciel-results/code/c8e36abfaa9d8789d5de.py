"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def _compute_reflectance_vectorized(layer_types, thicknesses, materials, ambient_index, substrate_index, wavelengths):
    """
    Compute normal-incidence reflectance using the characteristic matrix method (vectorized).
    layer_types: list of 'L' or 'H'
    thicknesses: list of physical thicknesses in nm
    materials: dict with 'L' and 'H' indices
    wavelengths: array of wavelengths in nm
    Returns: array of reflectances (fraction, not percent)
    """
    n_sub = substrate_index
    n_amb = ambient_index
    
    num_layers = len(thicknesses)
    if num_layers == 0:
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return np.full(len(wavelengths), np.abs(r)**2)
    
    # Precompute indices for each layer
    n_layers_arr = np.array([materials[lt] for lt in layer_types])
    d_layers_arr = np.array(thicknesses)
    
    # wavelengths shape: (N,)
    # We'll compute the characteristic matrix for all wavelengths at once
    # M_total starts as identity for all wavelengths
    
    N = len(wavelengths)
    M00 = np.ones(N, dtype=complex)
    M01 = np.zeros(N, dtype=complex)
    M10 = np.zeros(N, dtype=complex)
    M11 = np.ones(N, dtype=complex)
    
    for j in range(num_layers):
        n = n_layers_arr[j]
        d = d_layers_arr[j]
        
        # delta = 2*pi*n*d/lambda for each wavelength
        delta = 2.0 * np.pi * n * d / wavelengths
        
        c = np.cos(delta)
        s = np.sin(delta)
        
        # M_layer = [[c, i*s/n], [i*n*s, c]]
        L00 = c
        L01 = 1j * s / n
        L10 = 1j * n * s
        L11 = c
        
        # M_total = M_total @ M_layer
        new_M00 = M00 * L00 + M01 * L10
        new_M01 = M00 * L01 + M01 * L11
        new_M10 = M10 * L00 + M11 * L10
        new_M11 = M10 * L01 + M11 * L11
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    # Admittance Y = M10 / M00
    Y = M10 / M00
    
    # Reflectance
    r = (n_amb - Y) / (n_amb + Y)
    reflectances = np.abs(r)**2
    
    return reflectances

def design(problem):
    """
    Design an antireflection coating using an improved genetic algorithm.
    """
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Wavelengths for evaluation
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    n_L = materials['L']
    n_H = materials['H']
    
    np.random.seed(42)
    
    def random_individual():
        """Generate a random valid individual with alternating layers."""
        num_layers = np.random.randint(2, min(max_layers, 12) + 1)
        # Start with L (air side should be low index for AR coating)
        start_type = 'L'
        layer_types = []
        for i in range(num_layers):
            if i % 2 == 0:
                layer_types.append(start_type)
            else:
                layer_types.append('H' if start_type == 'L' else 'L')
        
        # Generate thicknesses with Dirichlet-like distribution
        raw = np.random.exponential(1.0, size=num_layers)
        raw = raw / raw.sum()
        thicknesses = raw * max_total_nm * 0.8  # Leave some margin
        thicknesses = np.clip(thicknesses, 0.0, max_layer_nm)
        
        # Ensure total <= max_total_nm
        total = thicknesses.sum()
        if total > max_total_nm:
            thicknesses = thicknesses * (max_total_nm / total)
        
        return layer_types, thicknesses.tolist()
    
    def evaluate(individual):
        """Evaluate an individual: return mean reflectance in percent."""
        layer_types, thicknesses = individual
        refl = _compute_reflectance_vectorized(layer_types, thicknesses, materials, ambient_index, substrate_index, wavelengths)
        return np.mean(refl) * 100.0
    
    def is_valid(individual):
        """Check if individual is valid."""
        layer_types, thicknesses = individual
        if len(layer_types) != len(thicknesses):
            return False
        if len(layer_types) > max_layers:
            return False
        if len(layer_types) == 0:
            return False
        for t in thicknesses:
            if t < -1e-9 or t > max_layer_nm + 1e-9:
                return False
        if sum(thicknesses) > max_total_nm + 1e-9:
            return False
        # Check alternating pattern
        for i in range(1, len(layer_types)):
            if layer_types[i] == layer_types[i-1]:
                return False
        return True
    
    def crossover(ind1, ind2):
        """Uniform crossover with repair."""
        lt1, t1 = ind1
        lt2, t2 = ind2
        
        max_len = max(len(lt1), len(lt2))
        child_lt = []
        child_t = []
        
        for i in range(max_len):
            if np.random.rand() < 0.5:
                if i < len(lt1):
                    child_lt.append(lt1[i])
                    child_t.append(t1[i])
            else:
                if i < len(lt2):
                    child_lt.append(lt2[i])
                    child_t.append(t2[i])
        
        # Ensure alternating pattern
        fixed_lt = []
        fixed_t = []
        if len(child_lt) > 0:
            fixed_lt.append(child_lt[0])
            fixed_t.append(child_t[0])
            for i in range(1, len(child_lt)):
                if child_lt[i] != fixed_lt[-1]:
                    fixed_lt.append(child_lt[i])
                    fixed_t.append(child_t[i])
        
        # Truncate to max_layers
        if len(fixed_lt) > max_layers:
            fixed_lt = fixed_lt[:max_layers]
            fixed_t = fixed_t[:max_layers]
        
        # Scale thicknesses to fit within max_total_nm
        if len(fixed_t) > 0:
            total = sum(fixed_t)
            if total > max_total_nm:
                fixed_t = [t * (max_total_nm / total) for t in fixed_t]
            # Clip
            fixed_t = [min(max(t, 0.0), max_layer_nm) for t in fixed_t]
        
        return fixed_lt, fixed_t
    
    def mutate(individual, mutation_strength=0.3):
        """Mutate an individual with local perturbations."""
        lt, t = individual
        lt = list(lt)
        t = list(t)
        
        # Possibly change number of layers (less frequent)
        if np.random.rand() < 0.05:
            if len(lt) < max_layers and np.random.rand() < 0.5:
                # Add layer at random position
                pos = np.random.randint(0, len(lt) + 1)
                if pos == 0:
                    new_type = 'H' if lt[0] == 'L' else 'L'
                else:
                    new_type = 'H' if lt[pos-1] == 'L' else 'L'
                lt.insert(pos, new_type)
                t.insert(pos, np.random.uniform(10, max_layer_nm * 0.5))
            elif len(lt) > 2:
                # Remove a layer (not first or last to keep structure)
                pos = np.random.randint(1, len(lt) - 1)
                lt.pop(pos)
                t.pop(pos)
        
        # Mutate thicknesses with local perturbations
        for i in range(len(t)):
            if np.random.rand() < mutation_strength:
                # Local perturbation: small change around current value
                if np.random.rand() < 0.7:
                    # Small perturbation
                    perturbation = np.random.uniform(-0.3, 0.3) * t[i]
                    t[i] = max(0.0, min(t[i] + perturbation, max_layer_nm))
                else:
                    # Larger perturbation
                    t[i] = np.random.uniform(0, max_layer_nm)
        
        # Scale to fit
        total = sum(t)
        if total > max_total_nm and total > 0:
            t = [x * (max_total_nm / total) for x in t]
        
        return lt, t
    
    # Initialize population with some good starting points
    population = []
    
    # Add some known good designs
    # Quarter-wave stack: L H L H ... with optimal thicknesses
    for num_layers in [4, 6, 8, 10]:
        layer_types = []
        for i in range(num_layers):
            layer_types.append('L' if i % 2 == 0 else 'H')
        
        # Quarter-wave thicknesses at center wavelength (600 nm)
        center_lam = 600.0
        thicknesses = []
        for i, lt in enumerate(layer_types):
            n = n_L if lt == 'L' else n_H
            d = center_lam / (4.0 * n)
            thicknesses.append(d)
        
        # Scale to fit within max_total_nm
        total = sum(thicknesses)
        if total > max_total_nm:
            thicknesses = [t * (max_total_nm / total) for t in thicknesses]
        
        if is_valid((layer_types, thicknesses)):
            population.append((layer_types, thicknesses))
    
    # Add random individuals
    while len(population) < 60:
        ind = random_individual()
        if is_valid(ind):
            population.append(ind)
    
    # Evaluate initial population
    scores = [evaluate(ind) for ind in population]
    
    best_ind = population[np.argmin(scores)]
    best_score = min(scores)
    
    # Genetic algorithm with adaptive parameters
    pop_size = len(population)
    num_generations = 50
    elite_size = 5
    
    for gen in range(num_generations):
        # Sort by score
        sorted_indices = np.argsort(scores)
        sorted_pop = [population[i] for i in sorted_indices]
        sorted_scores = [scores[i] for i in sorted_indices]
        
        best_ind = sorted_pop[0]
        best_score = sorted_scores[0]
        
        # Adaptive mutation strength: decrease over generations
        mutation_strength = max(0.1, 0.4 - 0.005 * gen)
        
        # Create new population
        new_pop = []
        # Elitism
        for i in range(min(elite_size, len(sorted_pop))):
            new_pop.append((list(sorted_pop[i][0]), list(sorted_pop[i][1])))
        
        while len(new_pop) < pop_size:
            # Selection (tournament)
            def tournament_select(pop, sc, k=3):
                indices = np.random.choice(len(pop), k, replace=False)
                best_idx = indices[np.argmin([sc[i] for i in indices])]
                return pop[best_idx]
            
            parent1 = tournament_select(sorted_pop, sorted_scores)
            parent2 = tournament_select(sorted_pop, sorted_scores)
            
            if np.random.rand() < 0.6:
                child = crossover(parent1, parent2)
            else:
                child = (list(parent1[0]), list(parent1[1]))
            
            child = mutate(child, mutation_strength)
            
            # Ensure validity
            if not is_valid(child):
                child = random_individual()
            
            new_pop.append(child)
        
        # Evaluate new population
        population = new_pop
        scores = [evaluate(ind) for ind in population]
        
        # Early stopping if we've converged
        if gen > 10 and best_score < 0.5:
            break
    
    # Return best individual
    layer_types, thicknesses = best_ind
    return layer_types, thicknesses
# EVOLVE-BLOCK-END
