"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _reflectance_stack(labels, thicknesses, wavelengths, n_ambient, n_substrate, materials):
    """
    Compute reflectance spectrum using the characteristic matrix method.
    labels: list of 'L' or 'H'
    thicknesses: list of physical thicknesses in nm
    wavelengths: array of wavelengths in nm
    n_ambient: refractive index of ambient (air)
    n_substrate: refractive index of substrate
    materials: dict mapping 'L' -> n_L, 'H' -> n_H
    Returns: array of reflectance fractions (0 to 1)
    """
    n_L = materials['L']
    n_H = materials['H']
    
    # Build the characteristic matrices
    # For normal incidence, the characteristic matrix for a layer of index n
    # and physical thickness d at wavelength lambda is:
    # M = [[cos(delta), i*sin(delta)/n], [i*n*sin(delta), cos(delta)]]
    # where delta = 2*pi*n*d/lambda
    
    total_M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    
    for label, d in zip(labels, thicknesses):
        n = n_L if label == 'L' else n_H
        delta = 2.0 * np.pi * n * d / wavelengths
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        M_layer = np.array([[cos_d, 1j * sin_d / n],
                            [1j * n * sin_d, cos_d]], dtype=complex)
        total_M = total_M @ M_layer
    
    # The admittance Y = (M11 + M12 * n_substrate) / (M21 + M22 * n_substrate)
    # Wait, let me be careful with the characteristic matrix convention.
    # Standard: [B, C]^T = M * [n_0, 1]^T where n_0 is substrate index? 
    # Actually, the standard characteristic matrix method:
    # Let the stack be between ambient (index n_a) and substrate (index n_s).
    # The characteristic matrix M_total is the product of individual layer matrices.
    # Then B = M11 + M12 * n_s
    #        C = M21 + M22 * n_s
    # The reflectance is R = | (n_a * B - C) / (n_a * B + C) |^2
    
    n_s = n_substrate
    n_a = n_ambient
    
    B = total_M[0, 0] + total_M[0, 1] * n_s
    C = total_M[1, 0] + total_M[1, 1] * n_s
    
    numerator = n_a * B - C
    denominator = n_a * B + C
    
    R = np.abs(numerator / denominator)**2
    
    return R

def _mean_reflectance_percent(labels, thicknesses, problem):
    """Compute mean reflectance in percent over the band."""
    band = problem['band_nm']
    wavelengths = np.linspace(band[0], band[1], 401)
    n_ambient = problem['ambient_index']
    n_substrate = problem['substrate_index']
    materials = problem['materials']
    
    if len(labels) == 0:
        # Bare substrate
        n_a = n_ambient
        n_s = n_substrate
        r = (n_a - n_s) / (n_a + n_s)
        R = r**2
        return R * 100.0
    
    R = _reflectance_stack(labels, thicknesses, wavelengths, n_ambient, n_substrate, materials)
    return np.mean(R) * 100.0

def _evaluate_design(labels, thicknesses, problem):
    """
    Evaluate a design. Returns (is_valid, score).
    Lower score is better.
    """
    max_layers = problem['max_layers']
    max_total_nm = problem['max_total_nm']
    max_layer_nm = problem['max_layer_nm']
    
    if len(labels) > max_layers:
        return False, float('inf')
    
    if len(labels) != len(thicknesses):
        return False, float('inf')
    
    total_thick = sum(thicknesses)
    if total_thick > max_total_nm + 1e-9:
        return False, float('inf')
    
    for t in thicknesses:
        if t < -1e-9 or t > max_layer_nm + 1e-9:
            return False, float('inf')
    
    # Clamp thicknesses to valid range
    clamped_thicknesses = [max(0.0, min(max_layer_nm, t)) for t in thicknesses]
    
    score = _mean_reflectance_percent(labels, clamped_thicknesses, problem)
    return True, score

def _crossover(parent1, parent2, labels1, labels2, problem):
    """
    Crossover: combine two designs.
    We'll use a simple approach: take a random prefix from parent1 and suffix from parent2,
    or vice versa. We need to ensure the labels alternate properly if needed, but the problem
    doesn't strictly require alternating layers, just at most 20 layers.
    Actually, the problem says "stack of at most twenty low- and high-index layers", it doesn't
    say they must alternate. So we can have any sequence.
    """
    len1 = len(labels1)
    len2 = len(labels2)
    
    if len1 == 0 and len2 == 0:
        return [], []
    
    if len1 == 0:
        return list(labels2), list(parent2)
    
    if len2 == 0:
        return list(labels1), list(parent1)
    
    # Random crossover point
    point = np.random.randint(0, len1 + 1)
    
    child_labels = list(labels1[:point]) + list(labels2[point:])
    child_thicknesses = list(parent1[:point]) + list(parent2[point:])
    
    # Truncate to max_layers
    max_layers = problem['max_layers']
    if len(child_labels) > max_layers:
        child_labels = child_labels[:max_layers]
        child_thicknesses = child_thicknesses[:max_layers]
    
    return child_labels, child_thicknesses

def _mutation(labels, thicknesses, problem):
    """
    Mutate a design: randomly change some thicknesses, add/remove layers, change labels.
    """
    max_layers = problem['max_layers']
    max_layer_nm = problem['max_layer_nm']
    
    new_labels = list(labels)
    new_thicknesses = list(thicknesses)
    
    # Mutate thicknesses
    for i in range(len(new_thicknesses)):
        if np.random.random() < 0.3:
            new_thicknesses[i] = np.random.uniform(0, max_layer_nm)
    
    # Occasionally add a layer
    if len(new_labels) < max_layers and np.random.random() < 0.2:
        pos = np.random.randint(0, len(new_labels) + 1)
        label = np.random.choice(['L', 'H'])
        thick = np.random.uniform(0, max_layer_nm)
        new_labels.insert(pos, label)
        new_thicknesses.insert(pos, thick)
    
    # Occasionally remove a layer
    if len(new_labels) > 0 and np.random.random() < 0.1:
        pos = np.random.randint(0, len(new_labels))
        new_labels.pop(pos)
        new_thicknesses.pop(pos)
    
    # Occasionally change a label
    for i in range(len(new_labels)):
        if np.random.random() < 0.1:
            new_labels[i] = 'L' if new_labels[i] == 'H' else 'H'
    
    return new_labels, new_thicknesses

def design(problem):
    """
    Design an antireflection coating using a genetic algorithm.
    """
    max_layers = problem['max_layers']
    max_total_nm = problem['max_total_nm']
    max_layer_nm = problem['max_layer_nm']
    materials = problem['materials']
    band = problem['band_nm']
    
    # Start with a population of random designs
    pop_size = 20
    generations = 30
    
    # Initial population: mix of random designs and some structured ones
    population = []
    
    # Add a single quarter-wave layer
    centre = 0.5 * (band[0] + band[1])
    population.append((['L'], [centre / (4.0 * materials['L'])]))
    
    # Add a two-layer quarter-wave stack
    population.append((['L', 'H'], [
        centre / (4.0 * materials['L']),
        centre / (4.0 * materials['H'])
    ]))
    
    # Add random designs
    for _ in range(pop_size - 2):
        n_layers = np.random.randint(1, max_layers + 1)
        labels = [np.random.choice(['L', 'H']) for _ in range(n_layers)]
        # Generate thicknesses that sum to at most max_total_nm
        raw_thick = np.random.uniform(0, max_layer_nm, n_layers)
        total = np.sum(raw_thick)
        if total > max_total_nm:
            raw_thick *= max_total_nm / total
        thicknesses = list(raw_thick)
        population.append((labels, thicknesses))
    
    # Evaluate initial population
    best_labels = None
    best_thicknesses = None
    best_score = float('inf')
    
    for labels, thicknesses in population:
        is_valid, score = _evaluate_design(labels, thicknesses, problem)
        if is_valid and score < best_score:
            best_score = score
            best_labels = list(labels)
            best_thicknesses = list(thicknesses)
    
    # Genetic algorithm loop
    for gen in range(generations):
        # Sort population by score
        scored_pop = []
        for labels, thicknesses in population:
            is_valid, score = _evaluate_design(labels, thicknesses, problem)
            scored_pop.append((score if is_valid else float('inf'), labels, thicknesses))
        
        scored_pop.sort(key=lambda x: x[0])
        
        # Keep the best
        if scored_pop[0][0] < best_score:
            best_score = scored_pop[0][0]
            best_labels = list(scored_pop[0][1])
            best_thicknesses = list(scored_pop[0][2])
        
        # Selection: keep top half
        elite_size = max(2, pop_size // 2)
        new_population = []
        for i in range(elite_size):
            new_population.append((list(scored_pop[i][1]), list(scored_pop[i][2])))
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Crossover
            p1_idx = np.random.randint(0, elite_size)
            p2_idx = np.random.randint(0, elite_size)
            parent1_labels = scored_pop[p1_idx][1]
            parent1_thick = scored_pop[p1_idx][2]
            parent2_labels = scored_pop[p2_idx][1]
            parent2_thick = scored_pop[p2_idx][2]
            
            child_labels, child_thick = _crossover(
                parent1_thick, parent2_thick,
                parent1_labels, parent2_labels,
                problem
            )
            
            # Mutation
            child_labels, child_thick = _mutation(child_labels, child_thick, problem)
            
            new_population.append((child_labels, child_thick))
        
        population = new_population
    
    # Final local optimization of the best design
    if best_labels is not None and best_thicknesses is not None:
        # Use scipy.optimize to fine-tune the thicknesses
        def objective(thick_array):
            labels = best_labels
            thick_list = list(thick_array)
            is_valid, score = _evaluate_design(labels, thick_list, problem)
            if not is_valid:
                return 1e10
            return score
        
        x0 = np.array(best_thicknesses)
        bounds = [(0, max_layer_nm)] * len(x0)
        # Add constraint that sum <= max_total_nm
        constraints = [
            {
                'type': 'ineq',
                'fun': lambda x: max_total_nm - np.sum(x)
            }
        ]
        
        try:
            result = minimize(
                objective, x0, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            if result.success:
                is_valid, score = _evaluate_design(best_labels, list(result.x), problem)
                if is_valid and score < best_score:
                    best_score = score
                    best_thicknesses = list(result.x)
        except Exception:
            pass
    
    # Ensure all thicknesses are within bounds
    final_thicknesses = [max(0.0, min(max_layer_nm, t)) for t in best_thicknesses]
    
    return best_labels, final_thicknesses
# EVOLVE-BLOCK-END
