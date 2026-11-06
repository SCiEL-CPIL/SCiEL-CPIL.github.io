"""Broadband antireflection coating design: SIMULATED ANNEALING with variable-length layers + DE-inspired population search.

Strategy:
1. Differential-evolution-inspired population search with variable-length layer sequences
2. Simulated annealing on top solutions for escape from local minima
3. Nelder-Mead local refinement
4. Final coordinate-descent polish
"""
import time
import numpy as np
import random
import math

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
                 ambient_index, substrate_index, deadline, max_iter=3000):
    """Nelder-Mead simplex optimization for thicknesses with fixed material sequence."""
    n_layers = len(materials)
    if n_layers == 0:
        return materials, thicknesses
    max_layer_nm = problem.get("max_layer_nm", 400)
    max_total_nm = problem.get("max_total_nm", 1000)
    
    simplex = [list(thicknesses)]
    rng = random.Random(12345)
    for i in range(n_layers):
        perturbed = list(thicknesses)
        perturbed[i] = _clamp_thick(perturbed[i] + 10.0 * (1 if rng.random() < 0.5 else -1), max_layer_nm)
        simplex.append(perturbed)
    
    for _ in range(min(3, n_layers)):
        perturbed = list(thicknesses)
        for j in range(n_layers):
            perturbed[j] = _clamp_thick(perturbed[j] + rng.uniform(-20, 20), max_layer_nm)
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
                        ambient_index, substrate_index, deadline, max_rounds=150):
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
            step_sizes = [s * 0.6 for s in step_sizes]
            if min(step_sizes) < min_step:
                break
        else:
            step_sizes = [min(s * 1.3, max_step) for s in step_sizes]
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
    band_nm = problem["band_nm"]
    wl_center = 0.5 * (band_nm[0] + band_nm[1])
    
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
        
        if move_type < 0.55:
            if len(cur_mats) == 0:
                continue
            i = rng.randint(0, len(cur_mats) - 1)
            magnitude = rng.uniform(5, 40)
            new_thicks[i] = _clamp_thick(cur_thicks[i] + rng.uniform(-magnitude, magnitude), max_layer_nm)
        elif move_type < 0.70:
            if len(cur_mats) < 2:
                continue
            i, j = rng.sample(range(len(cur_mats)), 2)
            new_thicks[i] = _clamp_thick(cur_thicks[i] + rng.uniform(-30, 30), max_layer_nm)
            new_thicks[j] = _clamp_thick(cur_thicks[j] + rng.uniform(-30, 30), max_layer_nm)
        elif move_type < 0.80:
            if len(cur_mats) == 0:
                continue
            i = rng.randint(0, len(cur_mats) - 1)
            new_mats[i] = 'H' if cur_mats[i] == 'L' else 'L'
        elif move_type < 0.88:
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
        elif move_type < 0.94:
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

def _de_crossover(m1, t1, m2, t2, m3, t3, problem, rng, max_layer_nm, max_total_nm, max_layers):
    """DE-style crossover for variable-length designs.
    Returns (mats, thicks) - a new design combining parents.
    """
    # Determine target length
    target_len = rng.choice([len(m1), len(m2), len(m3)])
    target_len = max(1, min(target_len, max_layers))
    
    new_mats = []
    new_thicks = []
    
    # Choose a random position to insert extra layers if needed
    for i in range(target_len):
        # Choose which parent to take this layer from
        r = rng.random()
        if r < 0.4:
            src_mats, src_thicks = m1, t1
        elif r < 0.7:
            src_mats, src_thicks = m2, t2
        else:
            src_mats, src_thicks = m3, t3
        
        if i < len(src_mats):
            mat = src_mats[i]
            thick = src_thicks[i]
        else:
            # Need to generate a new layer
            mat = rng.choice(['L', 'H'])
            thick = rng.uniform(30, min(120, max_layer_nm))
        
        new_mats.append(mat)
        new_thicks.append(_clamp_thick(thick, max_layer_nm))
    
    # Ensure total thickness constraint
    total = sum(new_thicks)
    if total > max_total_nm:
        scale = max_total_nm / total
        new_thicks = [t * scale for t in new_thicks]
        new_thicks = [_clamp_thick(t, max_layer_nm) for t in new_thicks]
    
    # If still invalid, try to fix
    if not _is_valid(new_mats, new_thicks, problem):
        # Fall back to a random valid design
        n_layers = rng.randint(3, min(7, max_layers))
        new_mats = [rng.choice(['L', 'H']) for _ in range(n_layers)]
        new_thicks = [rng.uniform(30, min(120, max_layer_nm)) for _ in range(n_layers)]
        total = sum(new_thicks)
        if total > max_total_nm:
            scale = max_total_nm / total
            new_thicks = [t * scale for t in new_thicks]
    
    return new_mats, new_thicks

def _de_search(population, problem, wls, materials_dict, ambient_index, substrate_index,
               deadline, rng, n_generations=30, F=0.8, CR=0.7):
    """Differential evolution-inspired search with variable-length designs."""
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    # Evaluate initial population
    scored = []
    for mats, thicks in population:
        if _is_valid(mats, thicks, problem):
            score = _evaluate(mats, thicks, wls, ambient_index, substrate_index, materials_dict)
            scored.append((score, mats, thicks))
        else:
            scored.append((1e6, mats, thicks))
    
    scored.sort(key=lambda x: x[0])
    
    best_score = scored[0][0]
    best_mats = list(scored[0][1])
    best_thicks = list(scored[0][2])
    
    pop_size = len(scored)
    if pop_size < 3:
        return best_mats, best_thicks
    
    for gen in range(n_generations):
        if time.time() > deadline:
            break
        
        new_population = [scored[0]]  # Keep best
        
        for i in range(1, pop_size):
            # Select three distinct parents
            candidates = list(range(pop_size))
            candidates.remove(i)
            if len(candidates) < 2:
                candidates = list(range(pop_size))
            a, b, c = rng.sample(candidates, 3)
            
            # DE crossover
            trial_mats, trial_thicks = _de_crossover(
                scored[a][1], scored[a][2],
                scored[b][1], scored[b][2],
                scored[c][1], scored[c][2],
                problem, rng, max_layer_nm, max_total_nm, max_layers
            )
            
            if not _is_valid(trial_mats, trial_thicks, problem):
                # Fall back to mutation of parent i
                trial_mats, trial_thicks = list(scored[i][1]), list(scored[i][2])
                if len(trial_mats) > 0:
                    idx = rng.randint(0, len(trial_mats) - 1)
                    trial_thicks[idx] = _clamp_thick(
                        trial_thicks[idx] + rng.uniform(-20, 20), max_layer_nm
                    )
            
            trial_score = _evaluate(trial_mats, trial_thicks, wls, ambient_index, substrate_index, materials_dict)
            
            # Tournament selection
            if trial_score < scored[i][0]:
                new_population.append((trial_score, trial_mats, trial_thicks))
                if trial_score < best_score:
                    best_score = trial_score
                    best_mats = list(trial_mats)
                    best_thicks = list(trial_thicks)
            else:
                new_population.append(scored[i])
        
        scored = new_population
        scored.sort(key=lambda x: x[0])
    
    return best_mats, best_thicks

def _generate_initial_solutions(problem, wls, materials_dict, band_nm, n_solutions=20, seed=42):
    """Generate diverse initial solutions."""
    rng = random.Random(seed)
    solutions = []
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    nL = materials_dict['L']
    nH = materials_dict['H']
    wl_center = 0.5 * (band_nm[0] + band_nm[1])
    
    # Quarter-wave stacks of various lengths
    for n in [1, 3, 5, 7, 9, 11, 13]:
        layers = ['L' if (i % 2 == 0) else 'H' for i in range(n)]
        thicks = [wl_center / (4.0 * (nL if m == 'L' else nH)) for m in layers]
        if sum(thicks) <= max_total_nm:
            solutions.append((layers, thicks))
    
    # H-start quarter-wave
    for n in [3, 5, 7]:
        layers = ['H' if (i % 2 == 0) else 'L' for i in range(n)]
        thicks = [wl_center / (4.0 * (nH if m == 'H' else nL)) for m in layers]
        if sum(thicks) <= max_total_nm:
            solutions.append((layers, thicks))
    
    # Random designs
    for _ in range(10):
        n_layers = rng.randint(2, 10)
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
    initial_solutions = _generate_initial_solutions(problem, wls, materials_dict, band_nm, n_solutions=20, seed=42)
    
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
    
    # DE-inspired population search on top solutions
    if time.time() < deadline - 30:
        # Use top 12 solutions as initial population
        pop_size = min(12, len(scored))
        population = [(list(mats), list(thicks)) for _, mats, thicks in scored[:pop_size]]
        
        de_rng = random.Random(777)
        de_mats, de_thicks = _de_search(
            population, problem, wls, materials_dict,
            n_amb, n_sub, deadline - 25, de_rng,
            n_generations=25, F=0.8, CR=0.7
        )
        
        if _is_valid(de_mats, de_thicks, problem):
            de_score = _evaluate(de_mats, de_thicks, wls, n_amb, n_sub, materials_dict)
            if de_score < best_score:
                best_score = de_score
                best_materials = list(de_mats)
                best_thicknesses = list(de_thicks)
    
    # SA on top solutions with different seeds
    rng = random.Random(999)
    n_sa_runs = min(4, len(scored))
    for idx in range(n_sa_runs):
        if time.time() > deadline - 20:
            break
        score, mats, thicks = scored[idx]
        sa_seed = idx * 7 + 13
        sa_rng = random.Random(sa_seed)
        
        sa_mats, sa_thicks = _sa_optimize(
            list(mats), list(thicks), problem, wls, materials_dict,
            n_amb, n_sub, deadline - 15, sa_rng,
            T0=3.0, T1=0.0005, n_steps=30000)
        
        if _is_valid(sa_mats, sa_thicks, problem):
            sa_score = _evaluate(sa_mats, sa_thicks, wls, n_amb, n_sub, materials_dict)
            if sa_score < best_score:
                best_score = sa_score
                best_materials = list(sa_mats)
                best_thicknesses = list(sa_thicks)
    
    # Nelder-Mead refinement on best
    if time.time() < deadline - 10 and len(best_materials) > 0:
        nm_mats, nm_thicks = _nelder_mead(
            list(best_materials), list(best_thicknesses), problem, wls, materials_dict,
            n_amb, n_sub, deadline - 5, max_iter=2000)
        if _is_valid(nm_mats, nm_thicks, problem):
            nm_score = _evaluate(nm_mats, nm_thicks, wls, n_amb, n_sub, materials_dict)
            if nm_score < best_score:
                best_score = nm_score
                best_materials = list(nm_mats)
                best_thicknesses = list(nm_thicks)
    
    # Coordinate descent polish
    if time.time() < deadline - 3 and len(best_materials) > 0:
        try:
            cd_mats, cd_thicks = _coordinate_descent(
                list(best_materials), list(best_thicknesses), problem, wls, materials_dict,
                n_amb, n_sub, deadline - 1, max_rounds=80)
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
