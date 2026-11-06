"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

# EVOLVE-BLOCK-START

def _reflectance_spectrum(materials, thicknesses, wls, ambient_index=1.0, substrate_index=1.52, materials_dict=None):
    """
    Compute reflectance at all wavelengths using the characteristic matrix method (vectorized).
    """
    if materials_dict is None:
        materials_dict = {'L': 1.38, 'H': 2.35}
    
    n_layers = len(materials)
    if n_layers == 0:
        r = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return np.full_like(wls, r*r)
        
    M11 = np.ones_like(wls, dtype=complex)
    M12 = np.zeros_like(wls, dtype=complex)
    M21 = np.zeros_like(wls, dtype=complex)
    M22 = np.ones_like(wls, dtype=complex)
    
    for mat, thick in zip(materials, thicknesses):
        n = materials_dict[mat]
        if thick < 1e-9:
            continue
            
        beta = 2.0 * np.pi * n * thick / wls
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        L11 = cos_b
        L12 = 1.0j * sin_b / n
        L21 = n * 1.0j * sin_b
        L22 = cos_b
        
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        
        M11 = new_M11
        M12 = new_M12
        M21 = new_M21
        M22 = new_M22
        
    Y_s = substrate_index
    Y = (M21 + M22 * Y_s) / (M11 + M12 * Y_s)
    
    Y_0 = ambient_index
    r = (Y_0 - Y) / (Y_0 + Y)
    R = np.abs(r) ** 2
    return R

def _mean_reflectance(materials, thicknesses, band_nm=(400, 800), n_points=401,
                       ambient_index=1.0, substrate_index=1.52, materials_dict=None):
    """Compute mean reflectance in percent over the band."""
    if materials_dict is None:
        materials_dict = {'L': 1.38, 'H': 2.35}
    
    wls = np.linspace(band_nm[0], band_nm[1], n_points)
    refls = _reflectance_spectrum(materials, thicknesses, wls, ambient_index, substrate_index, materials_dict)
    return 100.0 * np.mean(refls)

def _is_valid(materials, thicknesses, problem):
    """Check if the design satisfies all constraints."""
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    if len(materials) > max_layers:
        return False
    if len(materials) != len(thicknesses):
        return False
    for t in thicknesses:
        if t < -1e-9 or t > max_layer_nm + 1e-9:
            return False
    if sum(thicknesses) > max_total_nm + 1e-9:
        return False
    return True

def _penalized_score(materials, thicknesses, problem, band_nm=(400, 800), n_points=401):
    """
    Compute a penalized mean reflectance score.
    Adds penalty for constraint violations.
    """
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    materials_dict = problem["materials"]
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    
    penalty = 0.0
    total = sum(thicknesses)
    if total > max_total_nm:
        penalty += (total - max_total_nm) * 0.1
    for t in thicknesses:
        if t < 0:
            penalty += abs(t) * 0.1
        elif t > max_layer_nm:
            penalty += (t - max_layer_nm) * 0.1
    
    base_score = _mean_reflectance(materials, thicknesses, band_nm, n_points, ambient_index, substrate_index, materials_dict)
    return base_score + penalty

def _ga(problem, wls, deadline, pop_size=50, generations=200):
    """
    Genetic Algorithm to optimize both material sequence and thicknesses.
    """
    materials_dict = problem["materials"]
    nL = materials_dict["L"]
    nH = materials_dict["H"]
    n_sub = problem["substrate_index"]
    n_amb = problem["ambient_index"]
    band_nm = problem["band_nm"]
    wl_center = 0.5 * (band_nm[0] + band_nm[1])
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    max_layers = problem.get("max_layers", 20)
    
    # Initialize population
    # Mix of random and quarter-wave inspired
    pop = []
    for i in range(pop_size):
        if i < 10:
            # Quarter wave stack
            n_layers = random.randint(3, 8)
            mats = []
            thicks = []
            for j in range(n_layers):
                m = 'L' if j % 2 == 0 else 'H'
                n = nL if m == 'L' else nH
                d = wl_center / (4.0 * n)
                mats.append(m)
                thicks.append(d)
        else:
            # Random
            n_layers = random.randint(2, max_layers)
            mats = [random.choice(['L', 'H']) for _ in range(n_layers)]
            thicks = [random.uniform(10, max_layer_nm) for _ in range(n_layers)]
            # Ensure total thickness is reasonable
            total = sum(thicks)
            if total > max_total_nm:
                scale = max_total_nm / total
                thicks = [t * scale for t in thicks]
        
        pop.append((mats, thicks))
        
    # Evaluate initial population
    scores = []
    for mats, thicks in pop:
        s = _penalized_score(mats, thicks, problem, band_nm)
        scores.append(s)
        
    best_idx = np.argmin(scores)
    best_score = scores[best_idx]
    best_mats = pop[best_idx][0]
    best_thicks = pop[best_idx][1]
    
    for gen in range(generations):
        if time.time() > deadline:
            break
            
        # Selection: Tournament selection
        new_pop = []
        for _ in range(pop_size):
            # Pick 3 random individuals
            idxs = np.random.randint(0, pop_size, 3)
            best_local = idxs[np.argmin([scores[i] for i in idxs])]
            
            # Crossover
            if random.random() < 0.7 and len(pop[best_local][0]) > 1:
                # Find another parent
                parent2_idx = random.choice([i for i in range(pop_size) if i != best_local])
                if not parent2_idx:
                    parent2_idx = 0
                
                mats1 = list(pop[best_local][0])
                thicks1 = list(pop[best_local][1])
                mats2 = list(pop[parent2_idx][0])
                thicks2 = list(pop[parent2_idx][1])
                
                # Uniform crossover for materials and thicknesses
                # Pad shorter list with None
                len1 = len(mats1)
                len2 = len(mats2)
                max_len = max(len1, len2)
                
                child_mats = []
                child_thicks = []
                for i in range(max_len):
                    if random.random() < 0.5:
                        if i < len1:
                            child_mats.append(mats1[i])
                            child_thicks.append(thicks1[i])
                    else:
                        if i < len2:
                            child_mats.append(mats2[i])
                            child_thicks.append(thicks2[i])
                            
                # Remove None if any (shouldn't happen with this logic, but safe)
                # Actually, if one list is shorter, we just take from the other.
                # If we want strict length, we can truncate.
                # Let's truncate to max_layers
                if len(child_mats) > max_layers:
                    child_mats = child_mats[:max_layers]
                    child_thicks = child_thicks[:max_layers]
            else:
                child_mats = list(pop[best_local][0])
                child_thicks = list(pop[best_local][1])
                
            # Mutation
            # 1. Material flip
            if random.random() < 0.3 and len(child_mats) > 0:
                idx = random.randint(0, len(child_mats) - 1)
                child_mats[idx] = 'H' if child_mats[idx] == 'L' else 'L'
                
            # 2. Add/Remove layer
            if random.random() < 0.2:
                if len(child_mats) < max_layers and random.random() < 0.5:
                    # Add layer
                    pos = random.randint(0, len(child_mats))
                    m = random.choice(['L', 'H'])
                    n = nL if m == 'L' else nH
                    d = wl_center / (4.0 * n) * random.uniform(0.8, 1.2)
                    child_mats.insert(pos, m)
                    child_thicks.insert(pos, d)
                elif len(child_mats) > 2:
                    # Remove layer
                    pos = random.randint(0, len(child_mats) - 1)
                    child_mats.pop(pos)
                    child_thicks.pop(pos)
                    
            # 3. Thickness mutation
            if len(child_thicks) > 0:
                for i in range(len(child_thicks)):
                    if random.random() < 0.5:
                        # Gaussian mutation
                        step = random.gauss(0, 20)
                        child_thicks[i] = max(0, min(max_layer_nm, child_thicks[i] + step))
                        
            # Enforce total thickness constraint
            total = sum(child_thicks)
            if total > max_total_nm:
                scale = max_total_nm / total
                child_thicks = [t * scale for t in child_thicks]
                
            new_pop.append((child_mats, child_thicks))
            
        # Evaluate new population
        new_scores = []
        for mats, thicks in new_pop:
            s = _penalized_score(mats, thicks, problem, band_nm)
            new_scores.append(s)
            
        # Elitism: keep best individual
        best_new_idx = np.argmin(new_scores)
        if new_scores[best_new_idx] < best_score:
            best_score = new_scores[best_new_idx]
            best_mats = new_pop[best_new_idx][0]
            best_thicks = new_pop[best_new_idx][1]
            
        # Replace worst 20% with new population
        # Sort new population by score
        sorted_idxs = np.argsort(new_scores)
        top_count = int(pop_size * 0.2)
        top_new = [new_pop[i] for i in sorted_idxs[:top_count]]
        
        # Remove worst from current pop
        sorted_current_idxs = np.argsort(scores)
        worst_current = sorted_current_idxs[-top_count:]
        
        for i in worst_current:
            pop[i] = top_new.pop(0)
            scores[i] = new_scores[sorted_current_idxs[sorted_current_idxs.tolist().index(i)]] # This is messy, let's just re-evaluate or track properly
            
        # Simpler replacement: just take the best of new_pop and replace worst of pop
        # Let's do a simpler elitism: keep best 20% of combined pop
        # Actually, let's just replace the worst 20% of the current pop with the best 20% of the new pop
        # We need to update scores correctly.
        
        # Let's restart the replacement logic to be cleaner
        # Current pop: pop, scores
        # New pop: new_pop, new_scores
        # Sort current by score
        curr_order = np.argsort(scores)
        worst_curr_indices = curr_order[-int(pop_size * 0.2):]
        
        # Sort new by score
        new_order = np.argsort(new_scores)
        best_new_indices = new_order[:int(pop_size * 0.2)]
        
        for i, new_i in zip(worst_curr_indices, best_new_indices):
            pop[i] = new_pop[new_i]
            scores[i] = new_scores[new_i]
            
        # Update global best
        global_best_idx = np.argmin(scores)
        if scores[global_best_idx] < best_score:
            best_score = scores[global_best_idx]
            best_mats = pop[global_best_idx][0]
            best_thicks = pop[global_best_idx][1]
            
    return best_mats, best_thicks, best_score

def _nelder_mead(f, x0, bounds, max_iter=200, tol=1e-5, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
    """
    Simple Nelder-Mead implementation with box constraints.
    """
    n = len(x0)
    simplex = [x0.copy()]
    for i in range(n):
        x_new = x0.copy()
        step = 0.05 * (bounds[i][1] - bounds[i][0]) + 0.01
        x_new[i] += step
        x_new[i] = np.clip(x_new[i], bounds[i][0], bounds[i][1])
        simplex.append(x_new)
        
    simplex_f = [f(x) for x in simplex]
    
    for it in range(max_iter):
        order = np.argsort(simplex_f)
        simplex = [simplex[i] for i in order]
        simplex_f = [simplex_f[i] for i in order]
        
        if abs(simplex_f[-1] - simplex_f[0]) < tol:
            break
            
        centroid = np.mean(simplex[:-1], axis=0)
        
        x_r = centroid + alpha * (centroid - simplex[-1])
        x_r = np.clip(x_r, [b[0] for b in bounds], [b[1] for b in bounds])
        f_r = f(x_r)
        
        if f_r < simplex_f[0]:
            x_e = centroid + gamma * (x_r - centroid)
            x_e = np.clip(x_e, [b[0] for b in bounds], [b[1] for b in bounds])
            f_e = f(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                simplex_f[-1] = f_e
            else:
                simplex[-1] = x_r
                simplex_f[-1] = f_r
        elif f_r < simplex_f[-2]:
            simplex[-1] = x_r
            simplex_f[-1] = f_r
        else:
            x_c = centroid + rho * (simplex[-1] - centroid)
            x_c = np.clip(x_c, [b[0] for b in bounds], [b[1] for b in bounds])
            f_c = f(x_c)
            if f_c < simplex_f[-1]:
                simplex[-1] = x_c
                simplex_f[-1] = f_c
            else:
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], [b[0] for b in bounds], [b[1] for b in bounds])
                    simplex_f[i] = f(simplex[i])
                    
    best_idx = np.argmin(simplex_f)
    return simplex[best_idx], simplex_f[best_idx]

def design(problem):
    """
    Design using Genetic Algorithm followed by local refinement.
    """
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s kill
    
    materials_dict = problem["materials"]
    nL = materials_dict["L"]
    nH = materials_dict["H"]
    n_sub = problem["substrate_index"]
    n_amb = problem["ambient_index"]
    band_nm = problem["band_nm"]
    wl_center = 0.5 * (band_nm[0] + band_nm[1])
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    wls = np.linspace(band_nm[0], band_nm[1], 401)
    
    # Run GA
    best_mats, best_thicks, best_score = _ga(problem, wls, deadline - 20)
    
    # Local refinement on the best GA result
    if time.time() < deadline - 5 and len(best_mats) > 0:
        current_materials = best_mats
        bounds = [(0.0, max_layer_nm) for _ in range(len(best_mats))]
        x0 = np.array(best_thicks)
        
        def objective_final(thicknesses):
            return _penalized_score(best_mats, thicknesses, problem, band_nm)
        
        try:
            x_opt, f_opt = _nelder_mead(objective_final, x0, bounds, max_iter=500, tol=1e-7)
            if f_opt < best_score:
                best_score = f_opt
                best_thicknesses = list(x_opt)
                best_thicks = best_thicknesses
        except Exception:
            pass
            
    # Ensure validity
    if not _is_valid(best_mats, best_thicks, problem):
        # Fallback to a valid single layer
        best_mats = ["L"]
        best_thicks = [wl_center / (4.0 * nL)]
        
    # Clip thicknesses to valid range
    best_thicks = [max(0.0, min(max_layer_nm, t)) for t in best_thicks]
    
    # Check total thickness constraint and scale if necessary
    total = sum(best_thicks)
    if total > max_total_nm:
        scale = max_total_nm / total
        best_thicks = [t * scale for t in best_thicks]
        
    return best_mats, best_thicks

# EVOLVE-BLOCK-END
