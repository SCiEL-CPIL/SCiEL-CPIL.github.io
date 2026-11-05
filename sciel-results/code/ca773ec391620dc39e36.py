"""Broadband antireflection coating design via Differential Evolution with arbitrary layer sequences."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _vectorized_reflectance(thicknesses, materials, n_ambient, n_substrate, n_L, n_H, wls):
    """
    Compute mean reflectance fraction for a single design across all wavelengths.
    Uses vectorized operations over the wavelength axis.
    
    thicknesses: array of physical thicknesses in nm
    materials: array of 'L' or 'H' strings
    wls: array of wavelengths in nm
    """
    n_layers = len(thicknesses)
    if n_layers == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return float(r * r)
    
    # Pre-compute refractive indices for each layer
    n_layers_arr = np.array([n_H if m == 'H' else n_L for m in materials])
    
    # M = Identity
    M11 = np.ones_like(wls, dtype=complex)
    M12 = np.zeros_like(wls, dtype=complex)
    M21 = np.zeros_like(wls, dtype=complex)
    M22 = np.ones_like(wls, dtype=complex)
    
    for i in range(n_layers):
        n = n_layers_arr[i]
        d = thicknesses[i]
        beta = 2.0 * np.pi * n * d / wls
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        L11 = cos_b
        L12 = -1j * sin_b / n
        L21 = 1j * n * sin_b
        L22 = cos_b
        
        # M = M @ L
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
    
    # Admittance: Y = (M11 * n_substrate + M12) / (M21 * n_substrate + M22)
    Y = (M11 * n_substrate + M12) / (M21 * n_substrate + M22)
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.real(r * r.conjugate())
    
    return float(np.mean(R))

def _evaluate_design(thicknesses, materials, n_ambient, n_substrate, n_L, n_H, wls, max_total_nm, max_layer_nm):
    """
    Evaluate a design. Returns reflectance percentage if valid, inf otherwise.
    """
    if len(thicknesses) == 0:
        return float('inf')
    
    # Check constraints
    if any(t < 0 or t > max_layer_nm for t in thicknesses):
        return float('inf')
    if sum(thicknesses) > max_total_nm:
        return float('inf')
        
    R_frac = _vectorized_reflectance(thicknesses, materials, n_ambient, n_substrate, n_L, n_H, wls)
    return R_frac * 100.0

def design(problem):
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    wl_min, wl_max = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 401 equally spaced wavelengths
    wls = np.linspace(wl_min, wl_max, 401)
    wl_center = 0.5 * (wl_min + wl_max)
    
    # Initial best: Single quarter-wave L layer
    qw_L = wl_center / (4.0 * n_L)
    qw_H = wl_center / (4.0 * n_H)
    
    # Fallback solution
    best_score = float('inf')
    best_thicknesses = [qw_L]
    best_materials = ['L']
    
    # Evaluate fallback
    score = _evaluate_design(best_thicknesses, best_materials, n_ambient, n_substrate, n_L, n_H, wls, max_total_nm, max_layer_nm)
    if score < best_score:
        best_score = score
    
    # Differential Evolution Parameters
    pop_size = 40
    max_generations = 200
    mutation_rate = 0.8
    crossover_rate = 0.9
    
    # We will optimize for a fixed number of layers to keep the search space manageable.
    # We try a few different layer counts.
    # Let's start with 5 layers, which is often a good balance.
    # We can also try 7 or 9 if time permits.
    
    layer_counts_to_try = [5, 7, 9]
    
    for n_layers in layer_counts_to_try:
        if time.time() > deadline - 10.0:
            break
            
        # Initialize population
        # Each individual is a vector of length n_layers
        # We also need to assign materials. 
        # Strategy: Randomly assign materials but ensure valid sequences?
        # Actually, the problem allows any sequence. 
        # However, alternating H-L is usually better. 
        # To keep it simple and effective, let's optimize thicknesses for a FIXED alternating pattern H-L-H-L...
        # This reduces the search space significantly compared to optimizing materials too.
        # If we allow arbitrary materials, the search space is 2^n_layers * max_layer_nm^n_layers.
        # Let's stick to alternating H-L-H-L... starting with H, as it's physically motivated for AR coatings.
        # Wait, the current program assumes alternating. 
        # The prompt says "Design a stack of at most twenty low- and high-index layers".
        # It does not force alternation. 
        # However, non-alternating layers (e.g., H-H) are rarely optimal for broadband AR unless very specific.
        # Let's try to optimize thicknesses for alternating patterns of different starting materials and counts.
        
        # Let's try a few specific patterns:
        # Pattern 1: H L H L ... (n_layers)
        # Pattern 2: L H L H ... (n_layers)
        
        for start_mat in ['H', 'L']:
            if time.time() > deadline - 10.0:
                break
                
            materials = []
            for i in range(n_layers):
                if i % 2 == 0:
                    materials.append(start_mat)
                else:
                    materials.append('L' if start_mat == 'H' else 'H')
            
            # Initialize population of thicknesses
            # Uniform random in [0, max_layer_nm] is a bad start.
            # Better: Around quarter-wave thicknesses.
            pop = np.zeros((pop_size, n_layers))
            for i in range(pop_size):
                for j in range(n_layers):
                    if materials[j] == 'H':
                        pop[i, j] = qw_H + np.random.uniform(-20, 20)
                    else:
                        pop[i, j] = qw_L + np.random.uniform(-20, 20)
                # Clamp to [0, max_layer_nm]
                pop[i, :] = np.clip(pop[i, :], 0, max_layer_nm)
            
            # Evaluate initial population
            pop_scores = np.zeros(pop_size)
            for i in range(pop_size):
                pop_scores[i] = _evaluate_design(pop[i, :], materials, n_ambient, n_substrate, n_L, n_H, wls, max_total_nm, max_layer_nm)
            
            # Sort population
            order = np.argsort(pop_scores)
            pop = pop[order]
            pop_scores = pop_scores[order]
            
            # Update best
            if pop_scores[0] < best_score:
                best_score = pop_scores[0]
                best_thicknesses = pop[0, :].tolist()
                best_materials = materials[:]
            
            # Differential Evolution Loop
            for gen in range(max_generations):
                if time.time() > deadline - 5.0:
                    break
                
                # Mutation and Crossover
                for i in range(pop_size):
                    # Select 3 distinct random individuals
                    idx = np.random.choice(pop_size, 3, replace=False)
                    a, b, c = pop[idx[0]], pop[idx[1]], pop[idx[2]]
                    
                    # Mutation: v = a + F * (b - c)
                    v = a + mutation_rate * (b - c)
                    
                    # Crossover
                    trial = np.zeros_like(v)
                    j_rand = np.random.randint(0, n_layers)
                    for j in range(n_layers):
                        if np.random.rand() < crossover_rate or j == j_rand:
                            trial[j] = v[j]
                        else:
                            trial[j] = pop[i, j]
                    
                    # Clamp
                    trial = np.clip(trial, 0, max_layer_nm)
                    
                    # Evaluate
                    trial_score = _evaluate_design(trial, materials, n_ambient, n_substrate, n_L, n_H, wls, max_total_nm, max_layer_nm)
                    
                    # Selection
                    if trial_score < pop_scores[i]:
                        pop[i] = trial
                        pop_scores[i] = trial_score
                
                # Sort population
                order = np.argsort(pop_scores)
                pop = pop[order]
                pop_scores = pop_scores[order]
                
                # Update best
                if pop_scores[0] < best_score:
                    best_score = pop_scores[0]
                    best_thicknesses = pop[0, :].tolist()
                    best_materials = materials[:]
                    
                # Early stopping if converged
                if pop_scores[0] < 0.01: # 0.01% reflectance is excellent
                    break

    # Final validity check and cleanup
    # Ensure we return a valid design
    if not best_thicknesses or any(t < 0 or t > max_layer_nm for t in best_thicknesses) or sum(best_thicknesses) > max_total_nm:
        # Fallback to single layer
        return ["L"], [qw_L]
        
    # Remove zero-thickness layers if any (though they should be clamped to >=0, tiny values are okay)
    # The problem allows 0 thickness? "every thickness must lie in [0, max_layer_nm]"
    # So 0 is allowed.
    
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
