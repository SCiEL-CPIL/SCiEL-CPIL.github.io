"""Broadband antireflection coating design using characteristic-matrix method."""
import time
import numpy as np
import random

# EVOLVE-BLOCK-START

def _reflectance_vectorized(materials, thicknesses, wls, ambient_index=1.0, substrate_index=1.52, materials_dict=None):
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
    refls = _reflectance_vectorized(materials, thicknesses, wls, ambient_index, substrate_index, materials_dict)
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
        if t < 0 or t > max_layer_nm:
            return False
    if sum(thicknesses) > max_total_nm:
        return False
    return True

def _evaluate(materials, thicknesses, wls, ambient_index, substrate_index, materials_dict):
    """Compute mean reflectance in percent."""
    refls = _reflectance_vectorized(materials, thicknesses, wls, ambient_index, substrate_index, materials_dict)
    return 100.0 * np.mean(refls)

def _local_refine(materials, thicknesses, problem, wls, materials_dict, 
                  ambient_index, substrate_index, band_nm, deadline, max_iter=5000):
    """
    Adaptive step-size local refinement for thicknesses.
    Uses multiple step sizes and coordinate descent.
    """
    n_layers = len(materials)
    if n_layers == 0:
        return materials, thicknesses
    
    max_layer_nm = problem.get("max_layer_nm", 400)
    max_total_nm = problem.get("max_total_nm", 1000)
    
    current_thicks = list(thicknesses)
    current_score = _evaluate(materials, current_thicks, wls, ambient_index, substrate_index, materials_dict)
    
    # Initialize step sizes
    step_sizes = [15.0] * n_layers
    min_step = 0.05
    max_step = 50.0
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(n_layers):
            # Try several step sizes around current value
            best_local_score = current_score
            best_local_thick = current_thicks[i]
            
            for factor in [1.0, 0.5, 0.25, 0.1, 2.0, 4.0]:
                step = step_sizes[i] * factor
                
                # Try increasing
                new_thick = min(current_thicks[i] + step, max_layer_nm)
                new_thicks = list(current_thicks)
                new_thicks[i] = new_thick
                if _is_valid(materials, new_thicks, problem):
                    new_score = _evaluate(materials, new_thicks, wls, ambient_index, substrate_index, materials_dict)
                    if new_score < best_local_score:
                        best_local_score = new_score
                        best_local_thick = new_thick
                
                # Try decreasing
                new_thick = max(current_thicks[i] - step, 0.0)
                new_thicks = list(current_thicks)
                new_thicks[i] = new_thick
                if _is_valid(materials, new_thicks, problem):
                    new_score = _evaluate(materials, new_thicks, wls, ambient_index, substrate_index, materials_dict)
                    if new_score < best_local_score:
                        best_local_score = new_score
                        best_local_thick = new_thick
            
            if best_local_score < current_score:
                current_thicks[i] = best_local_thick
                current_score = best_local_score
                improved = True
        
        if not improved:
            step_sizes = [s * 0.7 for s in step_sizes]
            if min(step_sizes) < min_step:
                break
        else:
            step_sizes = [min(s * 1.15, max_step) for s in step_sizes]
    
    return materials, current_thicks

def _sa_step(materials, thicknesses, problem, wls, materials_dict, 
             ambient_index, substrate_index, band_nm, T, rng):
    """
    Perform one simulated annealing step. Returns new materials, thicknesses, and whether it was accepted.
    """
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    n_layers = len(materials)
    
    # Choose a move type
    if n_layers < 2:
        move_type = rng.choice([1, 1, 1, 2, 3])
    elif n_layers >= max_layers:
        move_type = rng.choice([1, 1, 2, 4, 5])
    else:
        move_type = rng.choice([1, 1, 1, 2, 3, 4, 5])
    
    # Current score
    current_score = _evaluate(materials, thicknesses, wls, ambient_index, substrate_index, materials_dict)
    
    # Apply move
    new_materials = list(materials)
    new_thicknesses = list(thicknesses)
    
    if move_type == 1:  # Perturb a random thickness
        if n_layers == 0:
            return materials, thicknesses, False
        idx = rng.randint(0, n_layers - 1)
        base = max(thicknesses[idx], 50.0)
        delta = rng.gauss(0, base * 0.3)
        new_thicknesses[idx] = max(0.0, min(max_layer_nm, thicknesses[idx] + delta))
        
    elif move_type == 2:  # Flip a random layer type
        if n_layers == 0:
            return materials, thicknesses, False
        idx = rng.randint(0, n_layers - 1)
        new_materials[idx] = 'H' if materials[idx] == 'L' else 'L'
        
    elif move_type == 3:  # Add a layer at a random position
        if n_layers >= max_layers:
            return materials, thicknesses, False
        pos = rng.randint(0, n_layers)
        mat = rng.choice(['L', 'H'])
        n_mat = materials_dict[mat]
        wl_center = 0.5 * (band_nm[0] + band_nm[1])
        thick = wl_center / (4.0 * n_mat)
        thick = max(10.0, min(max_layer_nm, thick * rng.uniform(0.5, 1.5)))
        new_materials.insert(pos, mat)
        new_thicknesses.insert(pos, thick)
        
    elif move_type == 4:  # Remove a random layer
        if n_layers <= 1:
            return materials, thicknesses, False
        idx = rng.randint(0, n_layers - 1)
        new_materials.pop(idx)
        new_thicknesses.pop(idx)
        
    elif move_type == 5:  # Swap adjacent layers
        if n_layers < 2:
            return materials, thicknesses, False
        idx = rng.randint(0, n_layers - 2)
        new_materials[idx], new_materials[idx+1] = new_materials[idx+1], new_materials[idx]
        new_thicknesses[idx], new_thicknesses[idx+1] = new_thicknesses[idx+1], new_thicknesses[idx]
    
    # New score
    new_score = _evaluate(new_materials, new_thicknesses, wls, ambient_index, substrate_index, materials_dict)
    
    # Acceptance criterion
    if new_score < current_score:
        return new_materials, new_thicknesses, True
    else:
        delta_e = new_score - current_score
        if delta_e == 0:
            return new_materials, new_thicknesses, True
        if T > 1e-8:
            prob = np.exp(-delta_e / T)
            if rng.random() < prob:
                return new_materials, new_thicknesses, True
    
    return materials, thicknesses, False

def _nelder_mead(materials, thicknesses, problem, wls, materials_dict,
                 ambient_index, substrate_index, band_nm, deadline, max_iter=3000):
    """
    Nelder-Mead simplex optimization for thicknesses with fixed material sequence.
    """
    n_layers = len(materials)
    if n_layers == 0:
        return materials, thicknesses
    
    max_layer_nm = problem.get("max_layer_nm", 400)
    max_total_nm = problem.get("max_total_nm", 1000)
    
    # Initial simplex: current point + perturbations
    simplex = [list(thicknesses)]
    for i in range(n_layers):
        perturbed = list(thicknesses)
        perturbed[i] = min(max(perturbed[i] + 20.0, 0.0), max_layer_nm)
        simplex.append(perturbed)
    
    # Evaluate initial simplex
    scores = []
    for simplex_point in simplex:
        if _is_valid(materials, simplex_point, problem):
            scores.append(_evaluate(materials, simplex_point, wls, ambient_index, substrate_index, materials_dict))
        else:
            scores.append(1000.0)
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Sort simplex by score
        order = np.argsort(scores)
        simplex = [simplex[i] for i in order]
        scores = [scores[i] for i in order]
        
        # If convergence achieved
        if abs(scores[-1] - scores[0]) < 1e-10:
            break
        
        # Centroid of all but worst point
        centroid = [0.0] * n_layers
        for i in range(len(simplex) - 1):
            for j in range(n_layers):
                centroid[j] += simplex[i][j]
        centroid = [c / (len(simplex) - 1) for c in centroid]
        
        # Reflection
        alpha = 1.0
        reflected = [centroid[j] + alpha * (centroid[j] - simplex[-1][j]) for j in range(n_layers)]
        if _is_valid(materials, reflected, problem):
            ref_score = _evaluate(materials, reflected, wls, ambient_index, substrate_index, materials_dict)
        else:
            ref_score = 1000.0
        
        if ref_score < scores[0]:
            # Expansion
            gamma = 2.0
            expanded = [centroid[j] + gamma * (reflected[j] - centroid[j]) for j in range(n_layers)]
            if _is_valid(materials, expanded, problem):
                exp_score = _evaluate(materials, expanded, wls, ambient_index, substrate_index, materials_dict)
            else:
                exp_score = 1000.0
            
            if exp_score < ref_score:
                simplex[-1] = expanded
                scores[-1] = exp_score
            else:
                simplex[-1] = reflected
                scores[-1] = ref_score
        elif ref_score < scores[-2]:
            simplex[-1] = reflected
            scores[-1] = ref_score
        else:
            # Contraction
            rho = 0.5
            contracted = [centroid[j] + rho * (simplex[-1][j] - centroid[j]) for j in range(n_layers)]
            if _is_valid(materials, contracted, problem):
                con_score = _evaluate(materials, contracted, wls, ambient_index, substrate_index, materials_dict)
            else:
                con_score = 1000.0
            
            if con_score < scores[-1]:
                simplex[-1] = contracted
                scores[-1] = con_score
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    for j in range(n_layers):
                        simplex[i][j] = simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j])
                    if _is_valid(materials, simplex[i], problem):
                        scores[i] = _evaluate(materials, simplex[i], wls, ambient_index, substrate_index, materials_dict)
                    else:
                        scores[i] = 1000.0
    
    # Return best point
    best_idx = np.argmin(scores)
    return materials, simplex[best_idx]

def design(problem):
    """
    Design using simulated annealing over both layer sequence and thicknesses,
    with Nelder-Mead and adaptive step-size refinement.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
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
    
    rng = random.Random(42)
    
    # Initial solution: a few quarter-wave layers
    init_materials = []
    init_thicknesses = []
    for mat in ['L', 'H', 'L']:
        n = nL if mat == 'L' else nH
        d = wl_center / (4.0 * n)
        init_materials.append(mat)
        init_thicknesses.append(d)
    
    best_materials = list(init_materials)
    best_thicknesses = list(init_thicknesses)
    best_score = _evaluate(best_materials, best_thicknesses, wls, n_amb, n_sub, materials_dict)
    
    # Simulated annealing
    T_start = 5.0
    T_end = 0.01
    
    # Estimate time per iteration
    t0 = time.time()
    _sa_step(best_materials, best_thicknesses, problem, wls, materials_dict,
             n_amb, n_sub, band_nm, T_start, rng)
    t1 = time.time()
    time_per_iter = max(t1 - t0, 0.001)
    
    # Adjust iterations to fit within time budget
    time_budget = deadline - (time.time() - start_time) - 20.0  # Reserve 20s for refinement
    max_iter_sa = int(time_budget / time_per_iter)
    max_iter_sa = max(100, min(max_iter_sa, 10000))
    
    current_materials = list(init_materials)
    current_thicknesses = list(init_thicknesses)
    current_score = best_score
    
    for i in range(max_iter_sa):
        if time.time() - start_time > deadline - 25:
            break
        
        # Cooling schedule
        T = T_start * (T_end / T_start) ** (i / max_iter_sa)
        
        new_materials, new_thicknesses, accepted = _sa_step(
            current_materials, current_thicknesses, problem, wls, materials_dict,
            n_amb, n_sub, band_nm, T, rng)
        
        if accepted:
            current_materials = new_materials
            current_thicknesses = new_thicknesses
            current_score = _evaluate(current_materials, current_thicknesses, wls, n_amb, n_sub, materials_dict)
            
            if current_score < best_score:
                best_score = current_score
                best_materials = list(current_materials)
                best_thicknesses = list(current_thicknesses)
    
    # Final refinement: Nelder-Mead then adaptive step-size
    if time.time() - start_time < deadline - 10 and len(best_materials) > 0:
        try:
            # Nelder-Mead optimization
            nm_materials, nm_thicknesses = _nelder_mead(
                list(best_materials), list(best_thicknesses), problem, wls, materials_dict,
                n_amb, n_sub, band_nm, deadline - 10, max_iter=2000)
            
            if _is_valid(nm_materials, nm_thicknesses, problem):
                nm_score = _evaluate(nm_materials, nm_thicknesses, wls, n_amb, n_sub, materials_dict)
                if nm_score < best_score:
                    best_score = nm_score
                    best_materials = nm_materials
                    best_thicknesses = nm_thicknesses
        except Exception:
            pass
    
    # Adaptive step-size polish
    if time.time() - start_time < deadline - 5 and len(best_materials) > 0:
        try:
            refined_materials, refined_thicknesses = _local_refine(
                list(best_materials), list(best_thicknesses), problem, wls, materials_dict,
                n_amb, n_sub, band_nm, deadline - 5, max_iter=5000)
            
            if _is_valid(refined_materials, refined_thicknesses, problem):
                refined_score = _evaluate(refined_materials, refined_thicknesses, wls, n_amb, n_sub, materials_dict)
                if refined_score < best_score:
                    best_score = refined_score
                    best_materials = refined_materials
                    best_thicknesses = refined_thicknesses
        except Exception:
            pass
    
    # Ensure validity
    if not _is_valid(best_materials, best_thicknesses, problem):
        best_materials = ["L"]
        best_thicknesses = [wl_center / (4.0 * nL)]
    
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
