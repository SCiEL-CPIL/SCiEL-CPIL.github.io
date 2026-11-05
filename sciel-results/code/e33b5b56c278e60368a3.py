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
    Design an antireflection coating using a structured genetic algorithm
    that leverages the periodicity of alternating layer stacks.
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
    
    def make_alternating(num_layers, start_type='L'):
        """Create alternating layer type sequence."""
        layer_types = []
        for i in range(num_layers):
            if i % 2 == 0:
                layer_types.append(start_type)
            else:
                layer_types.append('H' if start_type == 'L' else 'L')
        return layer_types
    
    def quarter_wave_thicknesses(layer_types, center_wavelength):
        """Compute quarter-wave thicknesses for given layer types."""
        thicknesses = []
        for lt in layer_types:
            n = n_L if lt == 'L' else n_H
            d = center_wavelength / (4.0 * n)
            thicknesses.append(d)
        return thicknesses
    
    def scale_to_fit(thicknesses, max_total):
        """Scale thicknesses to fit within total constraint."""
        total = sum(thicknesses)
        if total > max_total and total > 0:
            factor = max_total / total
            return [t * factor for t in thicknesses]
        return thicknesses
    
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
    
    def evaluate(individual):
        """Evaluate an individual: return mean reflectance in percent."""
        layer_types, thicknesses = individual
        refl = _compute_reflectance_vectorized(layer_types, thicknesses, materials, ambient_index, substrate_index, wavelengths)
        return np.mean(refl) * 100.0
    
    def random_thicknesses(num_layers):
        """Generate random thicknesses with Dirichlet-like distribution."""
        raw = np.random.exponential(1.0, size=num_layers)
        raw = raw / raw.sum()
        thicknesses = raw * max_total_nm * 0.9
        thicknesses = np.clip(thicknesses, 0.0, max_layer_nm)
        # Ensure total <= max_total_nm
        total = thicknesses.sum()
        if total > max_total_nm:
            thicknesses = thicknesses * (max_total_nm / total)
        return thicknesses.tolist()
    
    def random_individual():
        """Generate a random valid individual with alternating layers."""
        num_layers = np.random.randint(2, min(max_layers, 14) + 1)
        start_type = np.random.choice(['L', 'H'])
        layer_types = make_alternating(num_layers, start_type)
        thicknesses = random_thicknesses(num_layers)
        return layer_types, thicknesses
    
    def crossover(ind1, ind2):
        """Uniform crossover with repair for alternating stacks."""
        lt1, t1 = ind1
        lt2, t2 = ind2
        
        # Choose a random number of layers for child
        min_len = min(len(lt1), len(lt2))
        max_len = max(len(lt1), len(lt2))
        child_len = np.random.randint(min(2, min_len), min(max_len + 1, max_layers + 1))
        
        # Determine starting type (from one of the parents)
        if np.random.rand() < 0.5:
            child_start = lt1[0]
        else:
            child_start = lt2[0]
        
        child_lt = make_alternating(child_len, child_start)
        
        # For each position, pick thickness from parent1 or parent2
        child_t = []
        for i in range(child_len):
            if i < len(t1) and (i >= len(t2) or np.random.rand() < 0.5):
                child_t.append(t1[i])
            elif i < len(t2):
                child_t.append(t2[i])
            else:
                child_t.append(np.random.uniform(0, max_layer_nm * 0.5))
        
        # Scale to fit
        child_t = scale_to_fit(child_t, max_total_nm)
        child_t = [min(max(t, 0.0), max_layer_nm) for t in child_t]
        
        return child_lt, child_t
    
    def mutate(individual, mutation_strength=0.3):
        """Mutate an individual with local perturbations."""
        lt, t = individual
        lt = list(lt)
        t = list(t)
        
        # Possibly change number of layers (less frequent)
        if np.random.rand() < 0.08:
            if len(lt) < max_layers and np.random.rand() < 0.5:
                # Add layer at end
                new_type = 'H' if lt[-1] == 'L' else 'L'
                lt.append(new_type)
                t.append(np.random.uniform(10, max_layer_nm * 0.5))
            elif len(lt) > 2:
                # Remove last layer
                lt.pop()
                t.pop()
        
        # Mutate thicknesses with local perturbations
        for i in range(len(t)):
            if np.random.rand() < mutation_strength:
                # Local perturbation: small change around current value
                if np.random.rand() < 0.7:
                    # Small perturbation
                    perturbation = np.random.uniform(-0.3, 0.3) * max(t[i], 10)
                    t[i] = max(0.0, min(t[i] + perturbation, max_layer_nm))
                else:
                    # Larger perturbation
                    t[i] = np.random.uniform(0, max_layer_nm)
        
        # Scale to fit
        total = sum(t)
        if total > max_total_nm and total > 0:
            t = [x * (max_total_nm / total) for x in t]
        
        return lt, t
    
    # Initialize population with physically-motivated designs
    population = []
    
    # Quarter-wave stacks at different center wavelengths
    center_wavelengths = [500, 550, 600, 650, 700]
    for center_lam in center_wavelengths:
        for num_layers in [3, 4, 5, 6, 7, 8]:
            for start_type in ['L', 'H']:
                layer_types = make_alternating(num_layers, start_type)
                thicknesses = quarter_wave_thicknesses(layer_types, center_lam)
                thicknesses = scale_to_fit(thicknesses, max_total_nm)
                thicknesses = [min(max(t, 0.0), max_layer_nm) for t in thicknesses]
                
                if is_valid((layer_types, thicknesses)):
                    population.append((layer_types, thicknesses))
    
    # Add some random individuals for diversity
    while len(population) < 80:
        ind = random_individual()
        if is_valid(ind):
            population.append(ind)
    
    # Evaluate initial population
    scores = [evaluate(ind) for ind in population]
    
    best_ind = population[np.argmin(scores)]
    best_score = min(scores)
    
    # Genetic algorithm with adaptive parameters
    pop_size = len(population)
    num_generations = 80
    elite_size = 8
    
    for gen in range(num_generations):
        # Sort by score
        sorted_indices = np.argsort(scores)
        sorted_pop = [population[i] for i in sorted_indices]
        sorted_scores = [scores[i] for i in sorted_indices]
        
        best_ind = sorted_pop[0]
        best_score = sorted_scores[0]
        
        # Adaptive mutation strength: decrease over generations
        mutation_strength = max(0.05, 0.35 - 0.004 * gen)
        
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
            
            if np.random.rand() < 0.5:
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
        if gen > 20 and best_score < 0.3:
            break
    
    # Final local refinement around best solution
    best_lt, best_t = best_ind
    best_t = list(best_t)
    
    # Try small perturbations around best
    for _ in range(50):
        perturbed_t = [max(0, min(t + np.random.normal(0, 5), max_layer_nm)) for t in best_t]
        perturbed_t = scale_to_fit(perturbed_t, max_total_nm)
        perturbed_t = [min(max(t, 0.0), max_layer_nm) for t in perturbed_t]
        
        candidate = (best_lt, perturbed_t)
        if is_valid(candidate):
            cand_score = evaluate(candidate)
            if cand_score < best_score:
                best_t = perturbed_t
                best_score = cand_score
    
    return best_lt, best_t
# EVOLVE-BLOCK-END
