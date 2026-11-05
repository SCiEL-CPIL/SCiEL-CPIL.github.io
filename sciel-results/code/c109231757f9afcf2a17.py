"""Broadband antireflection coating design via GA + Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def _reflectance_vectorized(materials, thicknesses, problem, wls):
    """
    Compute reflectance array (0 to 1) over all wavelengths simultaneously.
    materials: list of 'L' or 'H'
    thicknesses: list/array of thicknesses in nm
    problem: problem dict
    wls: array of wavelengths in nm
    Returns: array of reflectance values
    """
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    mat_indices = []
    for m in materials:
        mat_indices.append(n_L if m == 'L' else n_H)
    
    # Start with identity matrices: shape (n_wl, 2, 2)
    M11 = np.ones(len(wls), dtype=complex)
    M12 = np.zeros(len(wls), dtype=complex)
    M21 = np.zeros(len(wls), dtype=complex)
    M22 = np.ones(len(wls), dtype=complex)
    
    for idx, (mat, d) in enumerate(zip(materials, thicknesses)):
        n = mat_indices[idx]
        beta = 2.0 * np.pi * n * d / wls
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        # Layer matrix elements
        L11 = cos_b
        L12 = -1j * sin_b / n
        L21 = 1j * n * sin_b
        L22 = cos_b
        
        # M_total = M_total @ M_layer
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        
        M11 = new_M11
        M12 = new_M12
        M21 = new_M21
        M22 = new_M22
    
    # Admittance Y = (M21 + M22 * n_sub) / (M11 + M12 * n_sub)
    Y = (M21 + M22 * n_substrate) / (M11 + M12 * n_substrate)
    
    # Reflectance
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.abs(r) ** 2
    return R

def _mean_reflectance_percent(materials, thicknesses, problem, wls):
    """Compute mean reflectance in percent over the band."""
    R = _reflectance_vectorized(materials, thicknesses, problem, wls)
    return float(np.mean(R)) * 100.0

def _is_valid(materials, thicknesses, problem):
    """Check if the design is valid."""
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > problem["max_layers"]:
        return False
    if any(t < -1e-9 or t > problem["max_layer_nm"] + 1e-9 for t in thicknesses):
        return False
    if sum(max(0, t) for t in thicknesses) > problem["max_total_nm"] + 1e-9:
        return False
    if any(m not in ["L", "H"] for m in materials):
        return False
    return True

def _clamp_and_check(x, problem):
    """Clamp thicknesses and check validity. Returns (clamped_array, is_valid)."""
    x = np.clip(x, 0, problem["max_layer_nm"])
    if np.sum(x) > problem["max_total_nm"]:
        return x, False
    return x, True

def _ga_optimize(materials, x0, problem, wls, deadline, pop_size=40, n_gen=30, seed=None):
    """
    Genetic algorithm to optimize thicknesses for a fixed layer structure.
    """
    rng = random.Random(seed)
    n_params = len(x0)
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    # Initialize population
    population = []
    for i in range(pop_size):
        if i == 0:
            x = np.array(x0, dtype=float)
        elif i == 1:
            # Quarter-wave at center wavelength
            centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
            n_L = problem["materials"]["L"]
            n_H = problem["materials"]["H"]
            x = np.array([centre/(4*(n_L if m=='L' else n_H)) for m in materials], dtype=float)
        else:
            # Random within constraints
            n_layers = n_params
            # Generate random thicknesses that sum to <= max_total
            # Use Dirichlet-like sampling
            raw = np.array([rng.random() for _ in range(n_layers)])
            total = raw.sum()
            if total == 0:
                x = np.array([max_total / n_layers] * n_layers)
            else:
                x = (raw / total) * max_total * rng.uniform(0.5, 1.0)
            x = np.clip(x, 0, max_layer)
            # If still over, scale down
            if x.sum() > max_total:
                x = x * (max_total / x.sum())
        population.append(x)
    
    # Evaluate initial population
    f_vals = []
    for x in population:
        x_c, valid = _clamp_and_check(x, problem)
        if not valid:
            f_vals.append(100.0)
        else:
            f_vals.append(_mean_reflectance_percent(materials, list(x_c), problem, wls))
    
    best_x = population[0].copy()
    best_f = f_vals[0]
    
    for gen in range(n_gen):
        if time.time() > deadline:
            break
        
        # Sort by fitness
        order = np.argsort(f_vals)
        population = [population[i] for i in order]
        f_vals = [f_vals[i] for i in order]
        
        if f_vals[0] < best_f:
            best_f = f_vals[0]
            best_x = population[0].copy()
        
        # Elitism: keep top 2
        new_population = [population[0].copy(), population[1].copy()]
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Tournament selection
            def tournament():
                idxs = rng.sample(range(pop_size), min(3, pop_size))
                best_idx = min(idxs, key=lambda i: f_vals[i])
                return population[best_idx].copy()
            
            if len(new_population) < pop_size:
                p1 = tournament()
            if len(new_population) < pop_size:
                p2 = tournament()
            
            # Crossover (uniform)
            if rng.random() < 0.7 and n_params > 1:
                mask = np.array([rng.random() < 0.5 for _ in range(n_params)])
                child = np.where(mask, p1, p2)
            else:
                child = p1.copy()
            
            # Mutation
            if rng.random() < 0.8:
                # Gaussian mutation
                sigma = 20.0 * (1.0 - gen / n_gen * 0.7)  # Decreasing sigma
                child = child + rng.gauss(0, sigma) * np.ones(n_params)
            
            # Also try some larger mutations
            if rng.random() < 0.1:
                idx = rng.randint(0, n_params - 1)
                child[idx] = rng.uniform(0, max_layer)
            
            new_population.append(child)
        
        population = new_population
        
        # Evaluate new population
        f_vals = []
        for x in population:
            if time.time() > deadline:
                f_vals.append(100.0)
                continue
            x_c, valid = _clamp_and_check(x, problem)
            if not valid:
                f_vals.append(100.0)
            else:
                f_vals.append(_mean_reflectance_percent(materials, list(x_c), problem, wls))
        
        # Update best
        order = np.argsort(f_vals)
        if f_vals[order[0]] < best_f:
            best_f = f_vals[order[0]]
            best_x = population[order[0]].copy()
    
    return best_x, best_f

def _nm_optimize(materials, thicknesses, problem, wls, deadline, max_iter=300):
    """
    Nelder-Mead optimization on the thicknesses, keeping materials fixed.
    """
    x0 = np.array(thicknesses, dtype=float)
    n_params = len(x0)
    
    # Initial simplex
    simplex = [x0.copy()]
    for i in range(n_params):
        x = x0.copy()
        x[i] += 15.0
        simplex.append(x)
    
    # Evaluate initial simplex
    f_vals = []
    for x in simplex:
        if time.time() > deadline:
            f_vals.append(100.0)
            continue
        x_c, valid = _clamp_and_check(x, problem)
        if not valid:
            f_vals.append(100.0)
        else:
            f_vals.append(_mean_reflectance_percent(materials, list(x_c), problem, wls))
    
    # Nelder-Mead iterations
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Sort simplex by function value
        order = np.argsort(f_vals)
        simplex = [simplex[i] for i in order]
        f_vals = [f_vals[i] for i in order]
        
        # Check convergence
        if abs(f_vals[0] - f_vals[-1]) < 1e-8:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        alpha = 1.0
        x_r = centroid + alpha * (centroid - simplex[-1])
        x_r_c, valid_r = _clamp_and_check(x_r, problem)
        if not valid_r:
            f_r = 100.0
        else:
            f_r = _mean_reflectance_percent(materials, list(x_r_c), problem, wls)
        
        if f_vals[0] <= f_r < f_vals[-2]:
            simplex[-1] = x_r_c
            f_vals[-1] = f_r
        elif f_r < f_vals[0]:
            # Expansion
            gamma = 2.0
            x_e = centroid + gamma * (x_r - centroid)
            x_e_c, valid_e = _clamp_and_check(x_e, problem)
            if not valid_e:
                f_e = 100.0
            else:
                f_e = _mean_reflectance_percent(materials, list(x_e_c), problem, wls)
            if f_e < f_r:
                simplex[-1] = x_e_c
                f_vals[-1] = f_e
            else:
                simplex[-1] = x_r_c
                f_vals[-1] = f_r
        else:
            # Contraction
            rho = 0.5
            x_c = centroid + rho * (simplex[-1] - centroid)
            x_c_c, valid_c = _clamp_and_check(x_c, problem)
            if not valid_c:
                f_c = 100.0
            else:
                f_c = _mean_reflectance_percent(materials, list(x_c_c), problem, wls)
            if f_c < f_vals[-1]:
                simplex[-1] = x_c_c
                f_vals[-1] = f_c
            else:
                # Shrink
                for i in range(1, n_params + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    x_clamped, valid = _clamp_and_check(simplex[i], problem)
                    if not valid:
                        f_vals[i] = 100.0
                    else:
                        f_vals[i] = _mean_reflectance_percent(materials, list(x_clamped), problem, wls)
    
    # Return best
    order = np.argsort(f_vals)
    best_idx = order[0]
    best_x = np.clip(simplex[best_idx], 0, problem["max_layer_nm"])
    best_x, valid = _clamp_and_check(best_x, problem)
    if not valid:
        # Fall back to original
        return list(thicknesses), _mean_reflectance_percent(materials, thicknesses, problem, wls)
    return list(best_x), f_vals[best_idx]

def design(problem):
    """
    Design a broadband antireflection coating using GA + NM polish.
    """
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    band = problem["band_nm"]
    wls = np.linspace(band[0], band[1], 401)
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    # Precompute index arrays for common structures
    def make_materials(n_layers, start='L'):
        """Create alternating material labels."""
        if start == 'L':
            return ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        else:
            return ['H' if i % 2 == 0 else 'L' for i in range(n_layers)]
    
    def qw_thicknesses(materials, centre_wl):
        """Quarter-wave thicknesses at center wavelength."""
        return [centre_wl / (4.0 * (n_L if m == 'L' else n_H)) for m in materials]
    
    # Try multiple structures with GA optimization
    structures = []
    # Various layer counts and starting indices
    for n_layers in [4, 5, 6, 7, 8, 10, 12]:
        for start in ['L', 'H']:
            mats = make_materials(n_layers, start)
            # Check total thickness at quarter-wave
            centre = 0.5 * (band[0] + band[1])
            thicks = qw_thicknesses(mats, centre)
            if sum(thicks) <= max_total:
                structures.append((mats, thicks))
    
    # Also try some non-quarter-wave initial guesses
    for n_layers in [4, 6, 8]:
        for start in ['L', 'H']:
            mats = make_materials(n_layers, start)
            # Random initial
            raw = np.random.RandomState(42 + n_layers + (0 if start=='L' else 1)).rand(n_layers)
            thicks = (raw / raw.sum()) * max_total * 0.8
            thicks = np.clip(thicks, 0, max_layer)
            if thicks.sum() > max_total:
                thicks *= max_total / thicks.sum()
            structures.append((mats, list(thicks)))
    
    best_materials = None
    best_thicknesses = None
    best_score = float('inf')
    
    # Phase 1: GA optimization for each structure
    ga_results = []
    for idx, (mats, thicks) in enumerate(structures):
        if time.time() > deadline - 30.0:
            break
        
        # GA with time budget
        ga_deadline = deadline - 30.0  # Leave time for NM polish
        opt_thicks, score = _ga_optimize(mats, thicks, problem, wls, ga_deadline, 
                                          pop_size=30, n_gen=25, seed=idx)
        ga_results.append((mats, opt_thicks, score))
        
        if score < best_score:
            best_score = score
            best_materials = mats
            best_thicknesses = opt_thicks
    
    # Phase 2: NM polish on top GA results
    ga_results.sort(key=lambda x: x[2])
    n_polish = min(5, len(ga_results))
    for i in range(n_polish):
        if time.time() > deadline - 5.0:
            break
        
        mats, thicks, score = ga_results[i]
        polish_thicks, polish_score = _nm_optimize(mats, thicks, problem, wls, deadline - 5.0, max_iter=200)
        
        if polish_score < best_score:
            best_score = polish_score
            best_materials = mats
            best_thicknesses = polish_thicks
    
    # Fallback
    if best_materials is None:
        centre = 0.5 * (band[0] + band[1])
        return ["L"], [centre / (4.0 * n_L)]
    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
