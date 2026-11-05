"""Broadband antireflection coating design via vectorized Nelder-Mead optimization."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_DEADLINE = 150.0
_MARGIN = 5.0  # seconds of safety margin

def _compute_reflectance_vectorized(materials, thicknesses, wavelengths, n0, ns, nL, nH):
    """
    Compute reflectance at all wavelengths simultaneously using vectorized ops.
    Returns array of reflectance fractions.
    """
    n_layers = len(materials)
    if n_layers == 0:
        r = (n0 - ns) / (n0 + ns)
        return np.full(len(wavelengths), abs(r)**2)
    
    # M11, M12, M21, M22 as complex arrays of length len(wavelengths)
    M11 = np.ones(len(wavelengths), dtype=complex)
    M12 = np.zeros(len(wavelengths), dtype=complex)
    M21 = np.zeros(len(wavelengths), dtype=complex)
    M22 = np.ones(len(wavelengths), dtype=complex)
    
    for i, (mat, d) in enumerate(zip(materials, thicknesses)):
        n = nL if mat == 'L' else nH
        beta = 2.0 * np.pi * n * d / wavelengths
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        L11 = cos_b
        L12 = 1j * sin_b / n
        L21 = 1j * n * sin_b
        L22 = cos_b
        
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        
        M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
    
    B = M11 + M12 * ns
    C = M21 + M22 * ns
    Y = C / B
    r = (n0 - Y) / (n0 + Y)
    R = np.abs(r)**2
    return np.real(R)

def _mean_reflectance_pct(materials, thicknesses, problem, n_wl=401):
    """Mean reflectance in percent over the band."""
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    band = problem["band_nm"]
    
    wls = np.linspace(band[0], band[1], n_wl)
    R_array = _compute_reflectance_vectorized(materials, thicknesses, wls, n0, ns, nL, nH)
    return float(np.mean(R_array) * 100.0)

def _is_valid(materials, thicknesses, max_layers, max_total, max_layer):
    if len(materials) > max_layers:
        return False
    if len(materials) != len(thicknesses):
        return False
    for t in thicknesses:
        if t < -1e-9 or t > max_layer + 1e-9:
            return False
    if sum(thicknesses) > max_total + 1e-6:
        return False
    return True

def _evaluate(materials, thicknesses, problem, n_wl=401):
    """Return mean reflectance in percent, or inf if invalid."""
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    if not _is_valid(materials, thicknesses, max_layers, max_total, max_layer):
        return float('inf')
    return _mean_reflectance_pct(materials, thicknesses, problem, n_wl)

def _nelder_mead(materials, thicknesses, problem, deadline, max_iter=500):
    """
    Nelder-Mead simplex optimization on thicknesses.
    Materials are fixed; only thicknesses are optimized.
    """
    n_layers = len(thicknesses)
    if n_layers == 0:
        return materials, thicknesses
    
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    # Objective function
    def objective(thick_arr):
        # Check validity quickly
        if len(thick_arr) != len(materials):
            return float('inf')
        for t in thick_arr:
            if t < -1e-9 or t > max_layer_nm + 1e-9:
                return float('inf')
        if sum(thick_arr) > max_total_nm + 1e-6:
            return float('inf')
        
        # Clip tiny negatives for evaluation
        thks = [max(0.0, t) for t in thick_arr]
        return _mean_reflectance_pct(materials, thks, problem)

    # Initialize simplex
    # Start with initial point and perturbations
    simplex = []
    scores = []
    
    init = np.array(thicknesses, dtype=float)
    simplex.append(init)
    scores.append(objective(init))
    
    for i in range(n_layers):
        p = init.copy()
        # Perturb dimension i
        step = 20.0
        p[i] += step
        if p[i] > max_layer_nm:
             p[i] = max_layer_nm
        if p[i] < 0:
             p[i] = 0.0
        if sum(p) <= max_total_nm:
             simplex.append(p)
             scores.append(objective(p))
        else:
             # Try smaller step
             p[i] = init[i] + 5.0
             if p[i] > max_layer_nm: p[i] = max_layer_nm
             if p[i] < 0: p[i] = 0.0
             if sum(p) <= max_total_nm:
                 simplex.append(p)
                 scores.append(objective(p))
             else:
                 # Fallback: duplicate init
                 simplex.append(init.copy())
                 scores.append(scores[0])
    
    # Parameters
    alpha = 0.5
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for _ in range(max_iter):
        if time.time() > deadline - _MARGIN:
            break
        
        # Sort by score
        sorted_indices = np.argsort(scores)
        simplex = [simplex[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]
        
        # Convergence check
        if abs(scores[0] - scores[-1]) < 1e-10:
            break
        
        worst = simplex[-1]
        worst_score = scores[-1]
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        reflected = centroid + alpha * (centroid - worst)
        reflected = np.clip(reflected, 0, max_layer_nm)
        if sum(reflected) <= max_total_nm + 1e-6:
            ref_score = objective(reflected)
            
            if ref_score < scores[0]:
                # Expansion
                expanded = centroid + gamma * (reflected - centroid)
                expanded = np.clip(expanded, 0, max_layer_nm)
                if sum(expanded) <= max_total_nm + 1e-6:
                    exp_score = objective(expanded)
                    if exp_score < ref_score:
                        simplex[-1] = expanded
                        scores[-1] = exp_score
                    else:
                        simplex[-1] = reflected
                        scores[-1] = ref_score
                else:
                    simplex[-1] = reflected
                    scores[-1] = ref_score
            elif ref_score < worst_score:
                simplex[-1] = reflected
                scores[-1] = ref_score
            else:
                # Contraction
                contracted = centroid + rho * (worst - centroid)
                contracted = np.clip(contracted, 0, max_layer_nm)
                if sum(contracted) <= max_total_nm + 1e-6:
                    con_score = objective(contracted)
                    if con_score < worst_score:
                        simplex[-1] = contracted
                        scores[-1] = con_score
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            scores[i] = objective(simplex[i])
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        scores[i] = objective(simplex[i])
        
        # If we hit time limit during objective eval, break
        if time.time() > deadline - _MARGIN:
            break
            
    # Return best point
    best_idx = np.argmin(scores)
    best_thks = simplex[best_idx]
    # Ensure strict validity for return
    best_thks = [max(0.0, min(max_layer_nm, t)) for t in best_thks]
    if sum(best_thks) > max_total_nm:
        # Scale down if necessary (should be rare with clipping)
        scale = max_total_nm / sum(best_thks)
        best_thks = [t * scale for t in best_thks]
        
    return materials, best_thks

def _coordinate_descent(materials, thicknesses, problem, deadline, max_iter=1000):
    """
    Coordinate descent with adaptive steps for local refinement.
    """
    n_layers = len(thicknesses)
    if n_layers == 0:
        return materials, thicknesses
    
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    current_thick = list(thicknesses)
    current_score = _mean_reflectance_pct(materials, current_thick, problem)
    
    step_sizes = [50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.1, 0.05]
    
    for step in step_sizes:
        if time.time() > deadline - _MARGIN:
            break
        
        improved = True
        while improved and time.time() < deadline - _MARGIN:
            improved = False
            
            for i in range(n_layers):
                if time.time() > deadline - _MARGIN:
                    break
                
                orig = current_thick[i]
                # Try both directions
                for sign in [1, -1]:
                    new_thick = current_thick[:]
                    new_thick[i] = orig + sign * step
                    
                    if new_thick[i] < 0 or new_thick[i] > max_layer_nm:
                        continue
                    if sum(new_thick) > max_total_nm:
                        continue
                    
                    trial_score = _mean_reflectance_pct(materials, new_thick, problem)
                    if trial_score < current_score - 1e-12:
                        current_thick = new_thick
                        current_score = trial_score
                        improved = True
                        break
    
    return materials, current_thick

def _generate_candidates(problem):
    """Generate diverse initial candidates."""
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    band = problem["band_nm"]
    centre = 0.5 * (band[0] + band[1])
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    candidates = []
    
    # Quarter-wave L
    qw_L = centre / (4.0 * nL)
    candidates.append((["L"], [qw_L]))
    
    # Quarter-wave H
    qw_H = centre / (4.0 * nH)
    candidates.append((["H"], [qw_H]))
    
    # L-H quarter-wave pair
    if qw_L + qw_H <= max_total:
        candidates.append((["L", "H"], [qw_L, qw_H]))
        candidates.append((["H", "L"], [qw_H, qw_L]))
    
    # Multi-layer quarter-wave stacks
    for n in range(3, 9):
        total = 0.0
        mats = []
        thks = []
        for i in range(n):
            if i % 2 == 0:
                mat = "L"
                thk = centre / (4.0 * nL)
            else:
                mat = "H"
                thk = centre / (4.0 * nH)
            if total + thk > max_total:
                break
            mats.append(mat)
            thks.append(thk)
            total += thk
        if len(mats) >= 2:
            candidates.append((mats, thks))
            
    # Randomized structures
    rng = np.random.default_rng(42)
    for _ in range(20):
        n_layers = rng.integers(2, 15)
        mats = []
        thks = []
        total = 0.0
        for _ in range(n_layers):
            mat = 'L' if rng.random() > 0.5 else 'H'
            # Random thickness between 20 and 150 nm
            thk = rng.uniform(20, 150)
            if total + thk > max_total:
                break
            mats.append(mat)
            thks.append(thk)
            total += thk
        if len(mats) >= 2:
             candidates.append((mats, thks))
             
    return candidates

def design(problem):
    """Multi-start Nelder-Mead optimization for broadband antireflection coating."""
    start_time = time.time()
    deadline = start_time + _DEADLINE - _MARGIN
    
    candidates = _generate_candidates(problem)
    
    best_score = float('inf')
    best_mats = None
    best_thks = None
    
    # Evaluate and optimize each candidate
    for mats, thks in candidates:
        if time.time() > deadline:
            break
        
        # Quick pre-evaluation
        pre_score = _evaluate(mats, thks, problem)
        if pre_score < best_score:
            best_score = pre_score
            best_mats = list(mats)
            best_thks = list(thks)
        
        # Local optimization with Nelder-Mead
        opt_mats, opt_thks = _nelder_mead(
            list(mats), list(thks), problem, deadline
        )
        
        opt_score = _evaluate(opt_mats, opt_thks, problem)
        if opt_score < best_score:
            best_score = opt_score
            best_mats = opt_mats
            best_thks = opt_thks
        
        # If we have good time remaining, do more optimization on best
        if time.time() < deadline - 20:
            opt_mats2, opt_thks2 = _nelder_mead(
                list(best_mats), list(best_thks), problem, deadline
            )
            opt_score2 = _evaluate(opt_mats2, opt_thks2, problem)
            if opt_score2 < best_score:
                best_score = opt_score2
                best_mats = opt_mats2
                best_thks = opt_thks2
    
    # Final refinement with coordinate descent for extra precision
    if best_mats is not None and time.time() < deadline - 10:
        opt_mats, opt_thks = _coordinate_descent(
            list(best_mats), list(best_thks), problem, deadline
        )
        opt_score = _evaluate(opt_mats, opt_thks, problem)
        if opt_score < best_score:
            best_mats = opt_mats
            best_thks = opt_thks
            best_score = opt_score
    
    # One more Nelder-Mead pass if time allows
    if best_mats is not None and time.time() < deadline - 5:
        opt_mats, opt_thks = _nelder_mead(
            list(best_mats), list(best_thks), problem, deadline
        )
        opt_score = _evaluate(opt_mats, opt_thks, problem)
        if opt_score < best_score:
            best_mats = opt_mats
            best_thks = opt_thks
    
    # Ensure validity
    if best_mats is None or not _is_valid(best_mats, best_thks, 
                                           problem["max_layers"],
                                           problem["max_total_nm"],
                                           problem["max_layer_nm"]):
        # Fallback: single quarter-wave L
        centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
        best_mats = ["L"]
        best_thks = [centre / (4.0 * problem["materials"]["L"])]
    
    # Clamp thicknesses
    best_thks = [max(0.0, min(problem["max_layer_nm"], t)) for t in best_thks]
    
    return best_mats, best_thks
# EVOLVE-BLOCK-END
