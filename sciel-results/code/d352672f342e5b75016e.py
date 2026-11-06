"""Design a broadband antireflection coating using discrete genetic search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _reflectance_spectrum(materials, thicknesses, ambient_index, substrate_index, wavelengths):
    """
    Compute reflectance spectrum using characteristic-matrix method.
    Returns array of reflectance values (0 to 1) for each wavelength.
    """
    n_w = len(wavelengths)
    n_layers = len(materials)
    
    if n_layers == 0:
        r = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return np.full(n_w, r ** 2)
    
    # Precompute optical thicknesses: n_j * d_j for each layer
    n_arr = np.array(materials, dtype=float)
    d_arr = np.array(thicknesses, dtype=float)
    optical_thick = n_arr * d_arr
    
    # delta[j, i] = 2*pi*optical_thick[j]/wavelengths[i]  shape (n_layers, n_w)
    delta = 2.0 * np.pi * optical_thick[:, None] / wavelengths[None, :]
    
    cos_d = np.cos(delta)  # (n_layers, n_w)
    sin_d = np.sin(delta)  # (n_layers, n_w)
    
    # Initialize the running product as identity for each wavelength
    M00 = np.ones(n_w, dtype=complex)
    M01 = np.zeros(n_w, dtype=complex)
    M10 = np.zeros(n_w, dtype=complex)
    M11 = np.ones(n_w, dtype=complex)
    
    for j in range(n_layers):
        c = cos_d[j, :]
        s = sin_d[j, :]
        n_j = n_arr[j]
        
        # New M = M_old @ M_j
        # M_j = [[c, 1j*s/n_j], [1j*n_j*s, c]]
        new_M00 = M00 * c + M01 * (1j * s / n_j)
        new_M01 = M00 * (1j * s / n_j) + M01 * c
        new_M10 = M10 * c + M11 * (1j * n_j * s)
        new_M11 = M10 * (1j * n_j * s) + M11 * c
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    # Now compute B and C:
    # [B, C]^T = M @ [1, n_s]^T
    B = M00 * 1.0 + M01 * substrate_index
    C = M10 * 1.0 + M11 * substrate_index
    
    # Reflectance:
    # r = (n_0 * B - C) / (n_0 * B + C)
    # R = |r|^2
    r = (ambient_index * B - C) / (ambient_index * B + C)
    R = np.abs(r) ** 2
    
    return R

def design(problem):
    """
    Design a broadband antireflection coating using a discrete genetic algorithm
    over quarter-wave optical thickness levels and L/H layer sequences.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    materials_dict = problem["materials"]
    n_L = materials_dict["L"]
    n_H = materials_dict["H"]
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    band = problem["band_nm"]
    
    # Precompute wavelength grid for fast evaluation
    wavelengths = np.linspace(band[0], band[1], 401)
    
    def fast_reflectance(materials, thicknesses):
        """Fast reflectance computation using precomputed wavelengths."""
        n_layers = len(materials)
        if n_layers == 0:
            R = ((ambient_index - substrate_index) / 
                 (ambient_index + substrate_index)) ** 2
            return R * 100.0
        R = _reflectance_spectrum(materials, thicknesses, n_layers, 
                                   ambient_index, substrate_index, wavelengths)
        return np.mean(R) * 100.0
    
    # Define discrete thickness levels based on quarter-wave at center wavelength
    centre = 0.5 * (band[0] + band[1])
    # Quarter-wave optical thickness at center: n*d = lambda/4 => d = lambda/(4*n)
    # We'll define thickness levels in terms of optical thickness (n*d)
    # Optical thickness levels: from lambda/8 to lambda/2 in steps of lambda/16
    # This gives us discrete choices for each layer's optical thickness
    
    # Let's define optical thickness levels as multiples of lambda/16
    # At center wavelength, lambda/16 = centre/16
    # Optical thickness = k * centre/16, where k is an integer
    # Physical thickness d = optical_thickness / n
    
    # For a layer with index n, physical thickness = (k * centre/16) / n
    # We need 0 <= d <= max_layer_nm, so 0 <= k*centre/(16*n) <= max_layer_nm
    # k <= 16*n*max_layer_nm/centre
    
    # Let's use a simpler approach: define a set of discrete physical thickness values
    # that correspond to common optical thicknesses (quarter-wave, half-wave, etc.)
    # at various wavelengths in the band
    
    # Actually, let's use optical thickness levels directly.
    # For each layer, we choose an optical thickness from a discrete set.
    # The optical thickness determines the phase shift at each wavelength.
    
    # Define optical thickness levels: multiples of lambda_center/16
    # k from 1 to some max, where k*lambda_center/16 <= n*max_layer_nm
    # For n=1.38: max k = 16*1.38*400/600 ≈ 14.7, so k up to 14
    # For n=2.35: max k = 16*2.35*400/600 ≈ 25.1, so k up to 25
    
    # But we also have the total thickness constraint. Let's be conservative.
    # Use k from 1 to 12 for both materials (this limits optical thickness to 12*centre/16 = 0.75*centre)
    
    opt_thick_levels = [k * centre / 16.0 for k in range(1, 13)]  # 12 levels
    
    def opt_thick_to_phys(opt_thick, n):
        """Convert optical thickness to physical thickness."""
        return opt_thick / n
    
    # Genetic algorithm setup
    # Genome: (n_layers, layer_types, layer_opt_thick_indices)
    # layer_types: 0 for L, 1 for H
    # layer_opt_thick_indices: index into opt_thick_levels
    
    def decode_genome(genome):
        """Decode a genome into materials and thicknesses."""
        n_layers = genome[0]
        if n_layers == 0:
            return [], []
        
        layer_types = genome[1]
        layer_opt_indices = genome[2]
        
        materials = []
        thicknesses = []
        for i in range(n_layers):
            if layer_types[i] == 0:
                n = n_L
            else:
                n = n_H
            opt_thick = opt_thick_levels[layer_opt_indices[i]]
            phys_thick = opt_thick_to_phys(opt_thick, n)
            materials.append(n)
            thicknesses.append(phys_thick)
        
        return materials, thicknesses
    
    def genome_to_score(genome):
        """Compute the score for a genome."""
        materials, thicknesses = decode_genome(genome)
        if not materials:
            return ((ambient_index - substrate_index) / 
                   (ambient_index + substrate_index)) ** 2 * 100.0
        
        # Check total thickness constraint
        total = sum(thicknesses)
        if total > max_total_nm:
            return 100.0  # Penalty
        
        # Check individual layer thickness
        for t in thicknesses:
            if t > max_layer_nm:
                return 100.0  # Penalty
        
        return fast_reflectance(materials, thicknesses)
    
    def random_genome():
        """Generate a random genome."""
        n_layers = random.randint(1, max_layers)
        layer_types = [random.randint(0, 1) for _ in range(n_layers)]
        layer_opt_indices = [random.randint(0, len(opt_thick_levels) - 1) for _ in range(n_layers)]
        return (n_layers, layer_types, layer_opt_indices)
    
    def crossover(g1, g2):
        """Crossover two genomes."""
        # Uniform crossover
        n1, lt1, lo1 = g1
        n2, lt2, lo2 = g2
        
        n = random.choice([n1, n2, min(n1, n2), max(n1, n2)])
        n = max(1, min(n, max_layers))
        
        lt = []
        lo = []
        for i in range(n):
            if random.random() < 0.5:
                if i < n1:
                    lt.append(lt1[i])
                    lo.append(lo1[i])
                else:
                    lt.append(lt2[i % n2] if n2 > 0 else 0)
                    lo.append(lo2[i % n2] if n2 > 0 else 0)
            else:
                if i < n2:
                    lt.append(lt2[i])
                    lo.append(lo2[i])
                else:
                    lt.append(lt1[i % n1] if n1 > 0 else 0)
                    lo.append(lo1[i % n1] if n1 > 0 else 0)
        
        return (n, lt, lo)
    
    def mutate(genome, mutation_rate=0.1):
        """Mutate a genome."""
        n, lt, lo = genome
        
        # Mutate n_layers
        if random.random() < mutation_rate:
            n = random.randint(1, max_layers)
            lt = [random.randint(0, 1) for _ in range(n)]
            lo = [random.randint(0, len(opt_thick_levels) - 1) for _ in range(n)]
        
        # Mutate individual layers
        for i in range(n):
            if random.random() < mutation_rate:
                lt[i] = random.randint(0, 1)
            if random.random() < mutation_rate:
                lo[i] = random.randint(0, len(opt_thick_levels) - 1)
        
        return (n, lt, lo)
    
    # Initialize population
    pop_size = 200
    population = [random_genome() for _ in range(pop_size)]
    
    # Evaluate initial population
    scores = [genome_to_score(g) for g in population]
    
    # Track best
    best_idx = np.argmin(scores)
    best_genome = population[best_idx]
    best_score = scores[best_idx]
    
    # Genetic algorithm loop
    elite_size = 10
    tournament_size = 5
    max_generations = 2000
    
    for gen in range(max_generations):
        if time.time() > deadline:
            break
        
        # Sort population by score
        indices = np.argsort(scores)
        population = [population[i] for i in indices]
        scores = [scores[i] for i in indices]
        
        # Update best
        if scores[0] < best_score:
            best_score = scores[0]
            best_genome = population[0]
        
        # Create new population
        new_population = []
        
        # Elitism
        for i in range(elite_size):
            new_population.append(population[i])
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            if time.time() > deadline:
                break
            
            # Tournament selection
            def tournament_select():
                candidates = random.sample(range(pop_size), min(tournament_size, pop_size))
                best_c = min(candidates, key=lambda i: scores[i])
                return population[best_c]
            
            parent1 = tournament_select()
            parent2 = tournament_select()
            
            child = crossover(parent1, parent2)
            child = mutate(child)
            
            new_population.append(child)
        
        # Evaluate new population
        for i, genome in enumerate(new_population):
            if time.time() > deadline:
                break
            new_population[i] = genome
            scores[i] = genome_to_score(genome)
        
        population = new_population
    
    # Final polish: try local search on the best solution
    if time.time() < deadline - 10.0:
        # Try adjusting the best solution by small changes
        best_n, best_lt, best_lo = best_genome
        
        # Try different numbers of layers around the best
        for delta_n in range(-2, 3):
            if time.time() > deadline:
                break
            
            n_try = best_n + delta_n
            if n_try < 1 or n_try > max_layers:
                continue
            
            # Try all combinations of layer types and thickness indices
            # This is too expensive, so just try random perturbations
            for _ in range(50):
                if time.time() > deadline:
                    break
                
                lt_try = best_lt[:]
                lo_try = best_lo[:]
                
                # Adjust to new length
                if n_try < len(lt_try):
                    lt_try = lt_try[:n_try]
                    lo_try = lo_try[:n_try]
                elif n_try > len(lt_try):
                    lt_try.extend([random.randint(0, 1) for _ in range(n_try - len(lt_try))])
                    lo_try.extend([random.randint(0, len(opt_thick_levels) - 1) for _ in range(n_try - len(lo_try))])
                
                # Randomly perturb some layers
                for i in range(n_try):
                    if random.random() < 0.3:
                        lt_try[i] = random.randint(0, 1)
                    if random.random() < 0.3:
                        lo_try[i] = random.randint(0, len(opt_thick_levels) - 1)
                
                genome_try = (n_try, lt_try, lo_try)
                score_try = genome_to_score(genome_try)
                
                if score_try < best_score:
                    best_score = score_try
                    best_genome = genome_try
    
    # Decode best genome
    materials, thicknesses = decode_genome(best_genome)
    
    if not materials:
        # Fallback: single quarter-wave L layer
        return ["L"], [centre / (4.0 * n_L)]
    
    # Convert materials from indices back to labels
    material_labels = []
    for n in materials:
        if abs(n - n_L) < 1e-6:
            material_labels.append("L")
        else:
            material_labels.append("H")
    
    # Final constraint check
    total_thickness = sum(thicknesses)
    if total_thickness > max_total_nm:
        scale = max_total_nm / total_thickness
        thicknesses = [t * scale for t in thicknesses]
    
    thicknesses = [min(max(t, 0.0), max_layer_nm) for t in thicknesses]
    
    return material_labels, thicknesses
# EVOLVE-BLOCK-END
