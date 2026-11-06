"""Broadband antireflection coating design: multi-phase SA + L-BFGS-B + Nelder-Mead + CD polish."""
import time
import numpy as np
import random
import math
import sys

# EVOLVE-BLOCK-START

def _reflectance_vectorized(materials, thicknesses, wls, ambient_index=1.0, substrate_index=1.52, materials_dict=None):
    """Compute reflectance at all wavelengths using characteristic matrix method (vectorized)."""
    if materials_dict is None:
        materials_dict = {'L': 1.38, 'H': 2.35}
    n_layers = len(materials)
    if n_layers == 0:
        r = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return np.full_like(wls, r * r, dtype=np.float64)
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
    return np.real(R)

def _evaluate(materials, thicknesses, wls, ambient_index, substrate_index, materials_dict):
    """Compute mean reflectance in percent."""
    refls = _reflectance_vectorized(materials, thicknesses, wls, ambient_index, substrate_index, materials_dict)
    return 100.0 * np.mean(refls)

def _is_valid(materials, thicknesses, problem):
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

def _clamp_thick(t, max_layer_nm):
    return max(0.0, min(t, max_layer_nm))

def _nelder_mead(materials, thicknesses, problem, wls, materials_dict,
                 ambient_index, substrate_index, deadline, max_iter=2000):
    """Nelder-Mead simplex optimization for thicknesses with fixed material sequence."""
    n_layers = len(materials)
    if n_layers == 0:
        return materials, thicknesses
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    simplex = [list(thicknesses)]
    rng = random.Random(42)
    for i in range(n_layers):
        perturbed = list(thicknesses)
        perturbed[i] = _clamp_thick(perturbed[i] + 15.0 * (1 if rng.random() < 0.5 else -1), max_layer_nm)
        simplex.append(perturbed)
    
    scores = []
    for sp in simplex:
        if _is_valid(materials, sp, problem):
            scores.append(_evaluate(materials, sp, wls, ambient_index, substrate_index, materials_dict))
        else:
            scores.append(1e6)
    
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        order = np.argsort(scores)
        simplex = [simplex[i] for i in order]
        scores = [scores[i] for i in order]
        
        if abs(scores[-1] - scores[0]) < 1e-14:
            break
        
        centroid = np.mean([np.array(simplex[i]) for i in range(len(simplex) - 1)], axis=0)
        worst = np.array(simplex[-1])
        
        reflected = centroid + alpha * (centroid - worst)
        reflected = np.array([_clamp_thick(v, max_layer_nm) for v in reflected])
        if _is_valid(materials, reflected, problem):
            ref_score = _evaluate(materials, reflected, wls, ambient_index, substrate_index, materials_dict)
        else:
            ref_score = 1e6
        
        if ref_score < scores[0]:
            expanded = centroid + gamma * (reflected - centroid)
            expanded = np.array([_clamp_thick(v, max_layer_nm) for v in expanded])
            if _is_valid(materials, expanded, problem):
                exp_score = _evaluate(materials, expanded, wls, ambient_index, substrate_index, materials_dict)
            else:
                exp_score = 1e6
            if exp_score < ref_score:
                simplex[-1] = expanded.tolist()
                scores[-1] = exp_score
            else:
                simplex[-1] = reflected.tolist()
                scores[-1] = ref_score
        elif ref_score < scores[-2]:
            simplex[-1] = reflected.tolist()
            scores[-1] = ref_score
        else:
            contracted = centroid + rho * (worst - centroid)
            contracted = np.array([_clamp_thick(v, max_layer_nm) for v in contracted])
            if _is_valid(materials, contracted, problem):
                con_score = _evaluate(materials, contracted, wls, ambient_index, substrate_index, materials_dict)
            else:
                con_score = 1e6
            if con_score < scores[-1]:
                simplex[-1] = contracted.tolist()
                scores[-1] = con_score
            else:
                for i in range(1, len(simplex)):
                    simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n_layers)]
                    if _is_valid(materials, simplex[i], problem):
                        scores[i] = _evaluate(materials, simplex[i], wls, ambient_index, substrate_index, materials_dict)
                    else:
                        scores[i] = 1e6
    
    best_idx = int(np.argmin(scores))
    return materials, simplex[best_idx]

def _coordinate_descent(materials, thicknesses, problem, wls, materials_dict,
                        ambient_index, substrate_index, deadline, max_rounds=100):
    """Coordinate descent with adaptive step for final polish."""
    n_layers = len(materials)
    if n_layers == 0:
        return materials, thicknesses
    max_layer_nm = problem.get("max_layer_nm", 400)
    current_thicks = list(thicknesses)
    current_score = _evaluate(materials, current_thicks, wls, ambient_index, substrate_index, materials_dict)
    step_sizes = [5.0] * n_layers
    min_step = 0.005
    max_step = 30.0
    
    for rnd in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n_layers):
            best_local_score = current_score
            best_local_thick = current_thicks[i]
            for factor in [1.0, 0.5, 0.2, 0.1, 0.05, 2.0, 4.0]:
                step = step_sizes[i] * factor
                for sign in [1, -1]:
                    new_thick = _clamp_thick(current_thicks[i] + sign * step, max_layer_nm)
                    new_thicks = list(current_thicks)
                    new_thicks[i] = new_thick
                    if _is_valid(materials, new_thicks, problem):
                        new_score = _evaluate(materials, new_thicks, wls, ambient_index, substrate_index, materials_dict)
                        if new_score < best_local_score:
                            best_local_score = new_score
                            best_local_thick = new_thick
            if best_local_score < current_score - 1e-13:
                current_thicks[i] = best_local_thick
                current_score = best_local_score
                improved = True
        if not improved:
            step_sizes = [s * 0.5 for s in step_sizes]
            if min(step_sizes) < min_step:
                break
        else:
            step_sizes = [min(s * 1.2, max_step) for s in step_sizes]
    return materials, current_thicks

def _sa_optimize(materials, thicknesses, problem, wls, materials_dict,
                 ambient_index, substrate_index, deadline, rng,
                 T0=5.0, T1=0.001, n_steps=50000):
    """Simulated annealing with variable-length layer sequences."""
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    nL = materials_dict['L']
    nH = materials_dict['H']
    
    cur_mats = list(materials)
    cur_thicks = list(thicknesses)
    if not _is_valid(cur_mats, cur_thicks, problem):
        return materials, thicknesses
    cur_score = _evaluate(cur_mats, cur_thicks, wls, ambient_index, substrate_index, materials_dict)
    
    best_mats = list(cur_mats)
    best_thicks = list(cur_thicks)
    best_score = cur_score
    
    T = T0
    cooling = (T1 / T0) ** (1.0 / max(n_steps, 1))
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        T = max(T * cooling, T1)
        beta = 1.0 / max(T, 1e-10)
        
        move_type = rng.random()
        new_mats = list(cur_mats)
        new_thicks = list(cur_thicks)
        
        if move_type < 0.45:
            if len(cur_mats) == 0:
                continue
            i = rng.randint(0, len(cur_mats) - 1)
            magnitude = rng.uniform(5, 40)
            new_thicks[i] = _clamp_thick(cur_thicks[i] + rng.uniform(-magnitude, magnitude), max_layer_nm)
        elif move_type < 0.60:
            if len(cur_mats) < 2:
                continue
            i, j = rng.sample(range(len(cur_mats)), 2)
            new_thicks[i] = _clamp_thick(cur_thicks[i] + rng.uniform(-30, 30), max_layer_nm)
            new_thicks[j] = _clamp_thick(cur_thicks[j] + rng.uniform(-30, 30), max_layer_nm)
        elif move_type < 0.72:
            if len(cur_mats) == 0:
                continue
            i = rng.randint(0, len(cur_mats) - 1)
            new_mats[i] = 'H' if cur_mats[i] == 'L' else 'L'
        elif move_type < 0.82:
            if len(cur_mats) >= max_layers:
                continue
            pos = rng.randint(0, len(cur_mats))
            mat = rng.choice(['L', 'H'])
            n = nL if mat == 'L' else nH
            d = rng.uniform(20, min(100, max_layer_nm))
            if sum(cur_thicks) + d > max_total_nm:
                d = max(0.0, max_total_nm - sum(cur_thicks))
            if d < 5:
                continue
            new_mats.insert(pos, mat)
            new_thicks.insert(pos, d)
        elif move_type < 0.92:
            if len(cur_mats) <= 1:
                continue
            i = rng.randint(0, len(cur_mats) - 1)
            new_mats.pop(i)
            new_thicks.pop(i)
        else:
            if len(cur_mats) < 2:
                continue
            i, j = rng.sample(range(len(cur_mats)), 2)
            new_mats[i], new_mats[j] = new_mats[j], new_mats[i]
            new_thicks[i], new_thicks[j] = new_thicks[j], new_thicks[i]
        
        if not _is_valid(new_mats, new_thicks, problem):
            continue
        
        new_score = _evaluate(new_mats, new_thicks, wls, ambient_index, substrate_index, materials_dict)
        delta = new_score - cur_score
        
        if delta < 0 or rng.random() < math.exp(-delta * beta):
            cur_mats = new_mats
            cur_thicks = new_thicks
            cur_score = new_score
            if cur_score < best_score:
                best_score = cur_score
                best_mats = list(cur_mats)
                best_thicks = list(cur_thicks)
    
    return best_mats, best_thicks

def _differential_evolution(materials_init, thicknesses_init, problem, wls, materials_dict,
                            ambient_index, substrate_index, deadline, rng,
                            pop_size=30, n_iter=200, F=0.8, CR=0.9):
    """Differential evolution for fixed material sequence."""
    n_layers = len(materials_init)
    if n_layers == 0:
        return materials_init, thicknesses_init
    max_layer_nm = problem.get("max_layer_nm", 400)
    max_total_nm = problem.get("max_total_nm", 1000)
    
    # Initialize population
    population = []
    for _ in range(pop_size):
        thicks = [_clamp_thick(t + rng.gauss(0, 20), max_layer_nm) for t in thicknesses_init]
        if _is_valid(materials_init, thicks, problem):
            population.append(thicks)
        else:
            population.append(list(thicknesses_init))
    
    scores = [_evaluate(materials_init, p, wls, ambient_index, substrate_index, materials_dict) for p in population]
    
    best_idx = int(np.argmin(scores))
    best_thicks = list(population[best_idx])
    best_score = scores[best_idx]
    
    for iteration in range(n_iter):
        if time.time() > deadline:
            break
        
        for i in range(pop_size):
            if time.time() > deadline:
                break
            
            # Select three distinct random individuals
            a, b, c = rng.sample([j for j in range(pop_size) if j != i], 3)
            
            # Create trial vector
            trial = [population[a][k] + F * (population[b][k] - population[c][k]) for k in range(n_layers)]
            trial = [_clamp_thick(t, max_layer_nm) for t in trial]
            
            # Enforce total thickness constraint by scaling
            total = sum(trial)
            if total > max_total_nm:
                scale = max_total_nm / total
                trial = [t * scale for t in trial]
            
            if not _is_valid(materials_init, trial, problem):
                continue
            
            trial_score = _evaluate(materials_init, trial, wls, ambient_index, substrate_index, materials_dict)
            
            # Selection
            if trial_score <= scores[i]:
                population[i] = trial
                scores[i] = trial_score
                if trial_score < best_score:
                    best_score = trial_score
                    best_thicks = list(trial)
        
        # Maintain diversity: occasionally replace worst with random
        if iteration % 10 == 0 and iteration > 0:
            worst_idx = int(np.argmax(scores))
            if scores[worst_idx] > best_score * 1.5:
                new_ind = [_clamp_thick(rng.uniform(20, 150), max_layer_nm) for _ in range(n_layers)]
                total = sum(new_ind)
                if total > max_total_nm:
                    scale = max_total_nm / total
                    new_ind = [t * scale for t in new_ind]
                if _is_valid(materials_init, new_ind, problem):
                    population[worst_idx] = new_ind
                    scores[worst_idx] = _evaluate(materials_init, new_ind, wls, ambient_index, substrate_index, materials_dict)
    
    return materials_init, best_thicks

def _lbfgsb_optimize(materials, thicknesses, problem, wls, materials_dict,
                     ambient_index, substrate_index, deadline, max_iter=300):
    """L-BFGS-B optimization for thicknesses with fixed material sequence."""
    from scipy.optimize import minimize
    n_layers = len(materials)
    if n_layers == 0:
        return materials, thicknesses
    max_layer_nm = problem.get("max_layer_nm", 400)
    max_total_nm = problem.get("max_total_nm", 1000)
    
    def objective(x):
        x = np.clip(x, 0.0, max_layer_nm)
        if np.sum(x) > max_total_nm:
            x = x * (max_total_nm / np.sum(x))
        x = np.clip(x, 0.0, max_layer_nm)
        refls = _reflectance_vectorized(materials, x, wls, ambient_index, substrate_index, materials_dict)
        return 100.0 * np.mean(refls)
    
    x0 = np.array(thicknesses, dtype=np.float64)
    bounds = [(0.0, max_layer_nm)] * n_layers
    
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
    
    best_x = np.clip(result.x, 0.0, max_layer_nm)
    if np.sum(best_x) > max_total_nm:
        best_x = best_x * (max_total_nm / np.sum(best_x))
        best_x = np.clip(best_x, 0.0, max_layer_nm)
    
    return materials, best_x.tolist()

def _generate_initial_solutions(problem, wls, materials_dict, band_nm, n_solutions=30, seed=42):
    """Generate diverse initial solutions."""
    rng = random.Random(seed)
    solutions = []
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    nL = materials_dict['L']
    nH = materials_dict['H']
    wl_center = 0.5 * (band_nm[0] + band_nm[1])
    
    # Quarter-wave stacks of various lengths
    for n in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]:
        layers = ['L' if (i % 2 == 0) else 'H' for i in range(n)]
        thicks = [wl_center / (4.0 * (nL if m == 'L' else nH)) for m in layers]
        if sum(thicks) <= max_total_nm:
            solutions.append((layers, thicks))
    
    # H-start quarter-wave
    for n in [3, 5, 7, 9, 11, 13]:
        layers = ['H' if (i % 2 == 0) else 'L' for i in range(n)]
        thicks = [wl_center / (4.0 * (nH if m == 'H' else nL)) for m in layers]
        if sum(thicks) <= max_total_nm:
            solutions.append((layers, thicks))
    
    # Double-quarter-wave designs (L-H-L-H or H-L-H-L with matched pairs)
    for n in [4, 6, 8, 10, 12]:
        for start in ['L', 'H']:
            layers = [start if (i % 2 == 0) else ('H' if start == 'L' else 'L') for i in range(n)]
            thicks = []
            for m in layers:
                n_val = nL if m == 'L' else nH
                d = wl_center / (4.0 * n_val)
                thicks.append(d)
            if sum(thicks) <= max_total_nm:
                solutions.append((layers, thicks))
    
    # Random designs with various layer counts
    for _ in range(20):
        n_layers = rng.randint(3, 15)
        materials = []
        thicknesses = []
        total = 0.0
        for _ in range(n_layers):
            mat = rng.choice(['L', 'H'])
            n = nL if mat == 'L' else nH
            d = rng.uniform(30.0, min(120.0, max_layer_nm))
            if total + d > max_total_nm:
                d = max(0.0, max_total_nm - total)
            if d < 5.0:
                continue
            materials.append(mat)
            thicknesses.append(d)
            total += d
        if len(materials) >= 1:
            solutions.append((materials, thicknesses))
    
    # Optimal-thickness designs: n_layer*d = lambda/4 at multiple wavelengths
    # Using the concept of "equal-reflectance" designs
    for n in [5, 7, 9]:
        # Alternate L and H with thicknesses chosen for broadband
        layers = ['L' if (i % 2 == 0) else 'H' for i in range(n)]
        thicks = []
        for m in layers:
            n_val = nL if m == 'L' else nH
            # Use geometric mean of band for thickness
            d = 0.55 * wl_center / (4.0 * n_val)
            thicks.append(d)
        if sum(thicks) <= max_total_nm:
            solutions.append((layers, thicks))
    
    valid = []
    for mats, thicks in solutions:
        if _is_valid(mats, thicks, problem):
            valid.append((mats, thicks))
    return valid

def design(problem):
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
    max_layers = problem.get("max_layers", 20)
    
    wls = np.linspace(band_nm[0], band_nm[1], 401)
    
    # Generate initial solutions
    initial_solutions = _generate_initial_solutions(problem, wls, materials_dict, band_nm, n_solutions=30, seed=42)
    
    if not initial_solutions:
        d_q = wl_center / (4.0 * nL)
        return ['L'], [d_q]
    
    # Evaluate and sort
    scored = []
    for mats, thicks in initial_solutions:
        score = _evaluate(mats, thicks, wls, n_amb, n_sub, materials_dict)
        scored.append((score, mats, thicks))
    scored.sort(key=lambda x: x[0])
    
    best_score = scored[0][0]
    best_materials = list(scored[0][1])
    best_thicknesses = list(scored[0][2])
    
    # Phase 1: Differential Evolution on top initial solutions
    rng_de = random.Random(123)
    n_de_runs = min(5, len(scored))
    for idx in range(n_de_runs):
        if time.time() > deadline - 55:
            break
        score, mats, thicks = scored[idx]
        de_mats, de_thicks = _differential_evolution(
            list(mats), list(thicks), problem, wls, materials_dict,
            n_amb, n_sub, deadline - 50, rng_de,
            pop_size=25, n_iter=150, F=0.8, CR=0.9)
        if _is_valid(de_mats, de_thicks, problem):
            de_score = _evaluate(de_mats, de_thicks, wls, n_amb, n_sub, materials_dict)
            if de_score < best_score:
                best_score = de_score
                best_materials = list(de_mats)
                best_thicknesses = list(de_thicks)
    
    # Phase 2: L-BFGS-B on top solutions (fast, smooth optimizer)
    n_lbfgs_runs = min(6, len(scored))
    for idx in range(n_lbfgs_runs):
        if time.time() > deadline - 40:
            break
        score, mats, thicks = scored[idx]
        try:
            lb_mats, lb_thicks = _lbfgsb_optimize(
                list(mats), list(thicks), problem, wls, materials_dict,
                n_amb, n_sub, deadline - 35, max_iter=200)
            if _is_valid(lb_mats, lb_thicks, problem):
                lb_score = _evaluate(lb_mats, lb_thicks, wls, n_amb, n_sub, materials_dict)
                if lb_score < best_score:
                    best_score = lb_score
                    best_materials = list(lb_mats)
                    best_thicknesses = list(lb_thicks)
        except Exception:
            pass
    
    # Phase 3: Simulated Annealing on best solutions
    rng_sa = random.Random(999)
    n_sa_runs = min(4, len(scored))
    for idx in range(n_sa_runs):
        if time.time() > deadline - 25:
            break
        score, mats, thicks = scored[idx]
        if idx == 0:
            sa_start_mats, sa_start_thicks = list(best_materials), list(best_thicknesses)
        else:
            sa_start_mats, sa_start_thicks = list(mats), list(thicks)
        
        sa_mats, sa_thicks = _sa_optimize(
            sa_start_mats, sa_start_thicks, problem, wls, materials_dict,
            n_amb, n_sub, deadline - 20, rng_sa,
            T0=3.0, T1=0.0005, n_steps=35000)
        
        if _is_valid(sa_mats, sa_thicks, problem):
            sa_score = _evaluate(sa_mats, sa_thicks, wls, n_amb, n_sub, materials_dict)
            if sa_score < best_score:
                best_score = sa_score
                best_materials = list(sa_mats)
                best_thicknesses = list(sa_thicks)
    
    # Phase 4: L-BFGS-B polish on best
    if time.time() < deadline - 15 and len(best_materials) > 0:
        try:
            lb2_mats, lb2_thicks = _lbfgsb_optimize(
                list(best_materials), list(best_thicknesses), problem, wls, materials_dict,
                n_amb, n_sub, deadline - 10, max_iter=300)
            if _is_valid(lb2_mats, lb2_thicks, problem):
                lb2_score = _evaluate(lb2_mats, lb2_thicks, wls, n_amb, n_sub, materials_dict)
                if lb2_score < best_score:
                    best_score = lb2_score
                    best_materials = list(lb2_mats)
                    best_thicknesses = list(lb2_thicks)
        except Exception:
            pass
    
    # Phase 5: Nelder-Mead refinement on best
    if time.time() < deadline - 8 and len(best_materials) > 0:
        nm_mats, nm_thicks = _nelder_mead(
            list(best_materials), list(best_thicknesses), problem, wls, materials_dict,
            n_amb, n_sub, deadline - 4, max_iter=1500)
        if _is_valid(nm_mats, nm_thicks, problem):
            nm_score = _evaluate(nm_mats, nm_thicks, wls, n_amb, n_sub, materials_dict)
            if nm_score < best_score:
                best_score = nm_score
                best_materials = list(nm_mats)
                best_thicknesses = list(nm_thicks)
    
    # Phase 6: Coordinate descent polish
    if time.time() < deadline - 2 and len(best_materials) > 0:
        try:
            cd_mats, cd_thicks = _coordinate_descent(
                list(best_materials), list(best_thicknesses), problem, wls, materials_dict,
                n_amb, n_sub, deadline - 0.5, max_rounds=100)
            if _is_valid(cd_mats, cd_thicks, problem):
                cd_score = _evaluate(cd_mats, cd_thicks, wls, n_amb, n_sub, materials_dict)
                if cd_score < best_score:
                    best_score = cd_score
                    best_materials = list(cd_mats)
                    best_thicknesses = list(cd_thicks)
        except Exception:
            pass
    
    # Ensure validity
    if not _is_valid(best_materials, best_thicknesses, problem):
        d_q = wl_center / (4.0 * nL)
        best_materials = ["L"]
        best_thicknesses = [d_q]
    
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
