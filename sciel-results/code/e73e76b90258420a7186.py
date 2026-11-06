"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

# EVOLVE-BLOCK-START

def _compute_reflectance(labels, thicknesses, materials, ambient_index, substrate_index, band_nm):
    """Compute mean reflectance in percent over the band using characteristic matrix method."""
    if len(labels) == 0:
        r0 = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return r0 * r0 * 100.0
    
    n_L = materials['L']
    n_H = materials['H']
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    n_layers = len(labels)
    
    # Build index array
    n_indices = np.empty(n_layers + 2, dtype=float)
    n_indices[0] = ambient_index
    n_indices[-1] = substrate_index
    for i, lab in enumerate(labels):
        n_indices[i + 1] = n_L if lab == 'L' else n_H
    
    d = np.array(thicknesses, dtype=float)
    
    # Vectorized computation over all wavelengths
    delta = 2.0 * np.pi * np.outer(n_indices[1:-1] * d, 1.0 / wavelengths)
    cos_d = np.cos(delta)
    sin_d = np.sin(delta)
    
    m11 = np.ones(401, dtype=complex)
    m12 = np.zeros(401, dtype=complex)
    m21 = np.zeros(401, dtype=complex)
    m22 = np.ones(401, dtype=complex)
    
    for j in range(n_layers):
        n_j = n_indices[j + 1]
        lm11 = cos_d[j]
        lm12 = 1j * sin_d[j] / n_j
        lm21 = 1j * n_j * sin_d[j]
        lm22 = cos_d[j]
        
        new_m11 = m11 * lm11 + m12 * lm21
        new_m12 = m11 * lm12 + m12 * lm22
        new_m21 = m21 * lm11 + m22 * lm21
        new_m22 = m21 * lm12 + m22 * lm22
        
        m11 = new_m11
        m12 = new_m12
        m21 = new_m21
        m22 = new_m22
    
    Y = (m21 + m22 * substrate_index) / (m11 + m12 * substrate_index)
    r = (ambient_index - Y) / (ambient_index + Y)
    reflectances = np.abs(r) ** 2
    
    return np.mean(reflectances) * 100.0


def _nelder_mead(objective, x0, step=50.0, max_iter=200, tol=1e-6):
    """Simple Nelder-Mead optimizer for continuous variables."""
    n = len(x0)
    # Initialize simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        xi = np.array(x0, dtype=float)
        xi[i] += step
        simplex.append(xi)
    
    # Evaluate
    fvals = [objective(x) for x in simplex]
    
    for _ in range(max_iter):
        # Sort by function value
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[-1] - fvals[0]) < tol:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        alpha = 1.0
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = objective(xr)
        
        if fr < fvals[0]:
            # Expansion
            gamma = 2.0
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            rho = 0.5
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = objective(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = objective(simplex[i])
    
    # Return best
    order = np.argsort(fvals)
    return simplex[order[0]], fvals[order[0]]


def design(problem):
    """Design an antireflection coating stack using genetic algorithm over discrete thicknesses."""
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    materials = problem["materials"]
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    n_L = materials['L']
    n_H = materials['H']
    
    # Discrete thickness options: quarter-wave at various center wavelengths
    # Optical thickness = n*d, so d = (lambda/4)/n for quarter-wave
    L_thick_options = []
    H_thick_options = []
    for cw in range(400, 801, 20):  # 400 to 800 nm in steps of 20
        t_L = cw / (4.0 * n_L)
        t_H = cw / (4.0 * n_H)
        if 0 < t_L <= max_layer_nm:
            L_thick_options.append(round(t_L, 3))
        if 0 < t_H <= max_layer_nm:
            H_thick_options.append(round(t_H, 3))
    
    L_thick_options = sorted(set(L_thick_options))
    H_thick_options = sorted(set(H_thick_options))
    
    # Ensure minimum options
    if not L_thick_options:
        L_thick_options = [600 / (4.0 * n_L)]
    if not H_thick_options:
        H_thick_options = [600 / (4.0 * n_H)]
    
    def evaluate_design(labels, thicknesses):
        """Evaluate a design, returning infinity if invalid."""
        if len(labels) > max_layers:
            return float('inf')
        if len(labels) != len(thicknesses):
            return float('inf')
        if any(t < 0 or t > max_layer_nm for t in thicknesses):
            return float('inf')
        if sum(thicknesses) > max_total_nm:
            return float('inf')
        return _compute_reflectance(labels, thicknesses, materials, ambient_index, substrate_index, band_nm)
    
    # Genetic Algorithm
    pop_size = 100
    max_gen = 50
    elite_size = 10
    
    # Initialize population
    population = []
    for _ in range(pop_size):
        n_layers = random.randint(1, min(10, max_layers))
        labels = [random.choice(['L', 'H']) for _ in range(n_layers)]
        thicknesses = []
        for lab in labels:
            if lab == 'L':
                thicknesses.append(random.choice(L_thick_options))
            else:
                thicknesses.append(random.choice(H_thick_options))
        population.append((labels, thicknesses))
    
    # Evaluate initial population
    scores = [evaluate_design(l, t) for l, t in population]
    
    # Track best
    best_idx = np.argmin(scores)
    best_labels = population[best_idx][0][:]
    best_thicknesses = population[best_idx][1][:]
    best_score = scores[best_idx]
    
    # Genetic algorithm loop
    for gen in range(max_gen):
        if time.time() > deadline:
            break
        
        # Sort by score
        order = np.argsort(scores)
        population = [population[i] for i in order]
        scores = [scores[i] for i in order]
        
        # Update best
        if scores[0] < best_score:
            best_score = scores[0]
            best_labels = population[0][0][:]
            best_thicknesses = population[0][1][:]
        
        # Create new population
        new_population = []
        
        # Elitism
        for i in range(min(elite_size, len(population))):
            new_population.append(population[i])
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            if time.time() > deadline:
                break
            
            # Select parents via tournament
            def tournament_select():
                idx1 = random.randint(0, len(population) - 1)
                idx2 = random.randint(0, len(population) - 1)
                if scores[idx1] <= scores[idx2]:
                    return population[idx1]
                else:
                    return population[idx2]
            
            parent1 = tournament_select()
            parent2 = tournament_select()
            
            # Crossover
            if random.random() < 0.7:
                # Uniform crossover on labels
                p1_labels, p1_thicks = parent1
                p2_labels, p2_thicks = parent2
                
                # Determine max length
                max_len = max(len(p1_labels), len(p2_labels))
                min_len = min(len(p1_labels), len(p2_labels))
                
                # Pad shorter with random
                def pad_design(labels, thick_options_L, thick_options_H):
                    while len(labels) < max_len:
                        lab = random.choice(['L', 'H'])
                        labels.append(lab)
                        if lab == 'L':
                            thick_options = thick_options_L
                        else:
                            thick_options = thick_options_H
                        labels[-1] = lab
                    return labels
                
                # Simple crossover: take prefix from one, suffix from other
                cross_point = random.randint(0, min_len)
                
                child_labels = p1_labels[:cross_point] + p2_labels[cross_point:]
                child_thicks = p1_thicks[:cross_point] + p2_thicks[cross_point:]
                
                # If lengths differ, truncate or pad
                if len(child_labels) > max_layers:
                    child_labels = child_labels[:max_layers]
                    child_thicks = child_thicks[:max_layers]
                
                # Ensure validity
                if sum(child_thicks) > max_total_nm:
                    # Scale down
                    scale = max_total_nm / sum(child_thicks)
                    child_thicks = [t * scale for t in child_thicks]
                    child_thicks = [round(t, 3) for t in child_thicks]
                
                new_population.append((child_labels, child_thicks))
            else:
                # Clone one parent
                parent = random.choice([parent1, parent2])
                new_population.append((parent[0][:], parent[1][:]))
            
            # Mutation
            child_labels, child_thicks = new_population[-1]
            if random.random() < 0.3:
                # Mutate a random layer
                if len(child_labels) > 0:
                    idx = random.randint(0, len(child_labels) - 1)
                    if random.random() < 0.5:
                        # Change material
                        child_labels[idx] = 'L' if child_labels[idx] == 'H' else 'H'
                    # Change thickness
                    if child_labels[idx] == 'L':
                        child_thicks[idx] = random.choice(L_thick_options)
                    else:
                        child_thicks[idx] = random.choice(H_thick_options)
            
            # Mutate number of layers
            if random.random() < 0.1 and len(child_labels) < max_layers:
                # Add a layer
                new_lab = random.choice(['L', 'H'])
                if new_lab == 'L':
                    new_thick = random.choice(L_thick_options)
                else:
                    new_thick = random.choice(H_thick_options)
                
                if sum(child_thicks) + new_thick <= max_total_nm:
                    pos = random.randint(0, len(child_labels))
                    child_labels.insert(pos, new_lab)
                    child_thicks.insert(pos, new_thick)
            elif random.random() < 0.1 and len(child_labels) > 1:
                # Remove a layer
                idx = random.randint(0, len(child_labels) - 1)
                child_labels.pop(idx)
                child_thicks.pop(idx)
        
        # Evaluate new population
        population = new_population
        scores = [evaluate_design(l, t) for l, t in population]
    
    # Try some known good symmetric designs
    known_patterns = [
        ['L', 'H', 'L'],
        ['H', 'L', 'H'],
        ['L', 'H', 'L', 'H'],
        ['H', 'L', 'H', 'L'],
        ['L', 'H', 'L', 'H', 'L'],
        ['H', 'L', 'H', 'L', 'H'],
    ]
    
    for pattern in known_patterns:
        if time.time() > deadline:
            break
        if len(pattern) > max_layers:
            continue
        
        # Try various thickness combinations
        for t_L in L_thick_options[:5]:
            for t_H in H_thick_options[:5]:
                if time.time() > deadline:
                    break
                thicknesses = []
                for lab in pattern:
                    if lab == 'L':
                        thicknesses.append(t_L)
                    else:
                        thicknesses.append(t_H)
                
                if sum(thicknesses) > max_total_nm:
                    continue
                
                score = evaluate_design(pattern, thicknesses)
                if score < best_score:
                    best_score = score
                    best_labels = pattern[:]
                    best_thicknesses = thicknesses[:]
    
    # Final validation
    if not best_labels or len(best_labels) > max_layers:
        best_labels = ["L"]
        best_thicknesses = [600 / (4.0 * n_L)]
    
    if any(t < 0 or t > max_layer_nm for t in best_thicknesses):
        best_labels = ["L"]
        best_thicknesses = [600 / (4.0 * n_L)]
    
    if sum(best_thicknesses) > max_total_nm:
        best_labels = ["L"]
        best_thicknesses = [600 / (4.0 * n_L)]
    
    return best_labels, best_thicknesses

# EVOLVE-BLOCK-END
