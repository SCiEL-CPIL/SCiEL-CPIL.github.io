# EVOLVE-BLOCK-START
import time
import numpy as np

def _mean_reflectance_vec(materials, thicknesses, problem):
    """Compute mean reflectance in percent over the band using vectorized numpy."""
    if len(materials) == 0:
        n0 = problem["ambient_index"]
        ns = problem["substrate_index"]
        r = (n0 - ns) / (n0 + ns)
        return 100.0 * r * r
    
    band = problem["band_nm"]
    wavelengths = np.linspace(band[0], band[1], 401)
    mat_dict = problem["materials"]
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    
    # Initialize total matrix for all wavelengths
    M = np.ones((len(wavelengths), 2, 2), dtype=complex)
    
    for mat, thick in zip(materials, thicknesses):
        n = mat_dict[mat]
        d = thick
        if d == 0:
            continue
        beta = 2.0 * np.pi * n * d / wavelengths
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        m00 = cos_b
        m01 = 1.0j * sin_b / n
        m10 = n * 1.0j * sin_b
        m11 = cos_b
        
        M_new = np.empty_like(M)
        M_new[:, 0, 0] = M[:, 0, 0] * m00 + M[:, 0, 1] * m10
        M_new[:, 0, 1] = M[:, 0, 0] * m01 + M[:, 0, 1] * m11
        M_new[:, 1, 0] = M[:, 1, 0] * m00 + M[:, 1, 1] * m10
        M_new[:, 1, 1] = M[:, 1, 0] * m01 + M[:, 1, 1] * m11
        M = M_new
        
    Ys = ns
    Y = M[:, 1, 1] / M[:, 0, 1]
    r = (n0 - Y) / (n0 + Y)
    R = np.abs(r) ** 2 * 100.0
    
    return np.mean(R)

def _is_valid(materials, thicknesses, problem):
    """Check if the design is valid."""
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    if len(materials) > max_layers:
        return False
    if len(materials) != len(thicknesses):
        return False
    if sum(thicknesses) > max_total_nm + 1e-9:
        return False
    for t in thicknesses:
        if t < -1e-9 or t > max_layer_nm + 1e-9:
            return False
    return True

def _project_to_feasible(x, problem):
    """Project x onto the feasible region."""
    x = np.clip(x, 0.0, problem["max_layer_nm"])
    total = np.sum(x)
    if total > problem["max_total_nm"]:
        x = x * (problem["max_total_nm"] / total)
    return x

def _coordinate_descent(materials, thicknesses, problem, time_limit, step=10.0):
    """Perform coordinate descent on layer thicknesses."""
    x = np.array(thicknesses, dtype=float)
    x = _project_to_feasible(x, problem)
    n_params = len(x)
    if n_params == 0:
        return x, _mean_reflectance_vec(materials, thicknesses, problem)
    
    f = _mean_reflectance_vec(materials, x.tolist(), problem)
    best_f = f
    best_x = x.copy()
    
    # Adaptive step sizes
    steps = np.full(n_params, step)
    no_improve = 0
    
    while time.time() < time_limit:
        improved = False
        for i in range(n_params):
            if time.time() > time_limit:
                break
            # Try positive step
            x_trial = x.copy()
            x_trial[i] = min(x[i] + steps[i], problem["max_layer_nm"])
            x_trial = _project_to_feasible(x_trial, problem)
            f_trial = _mean_reflectance_vec(materials, x_trial.tolist(), problem)
            
            if f_trial < f:
                x = x_trial
                f = f_trial
                steps[i] *= 1.2
                improved = True
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            else:
                # Try negative step
                x_trial = x.copy()
                x_trial[i] = max(x[i] - steps[i], 0.0)
                x_trial = _project_to_feasible(x_trial, problem)
                f_trial = _mean_reflectance_vec(materials, x_trial.tolist(), problem)
                
                if f_trial < f:
                    x = x_trial
                    f = f_trial
                    steps[i] *= 1.2
                    improved = True
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                else:
                    steps[i] *= 0.7
                    # Try small positive
                    x_trial = x.copy()
                    x_trial[i] = min(x[i] + steps[i], problem["max_layer_nm"])
                    x_trial = _project_to_feasible(x_trial, problem)
                    f_trial = _mean_reflectance_vec(materials, x_trial.tolist(), problem)
                    if f_trial < f:
                        x = x_trial
                        f = f_trial
                        improved = True
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                    else:
                        # Try small negative
                        x_trial = x.copy()
                        x_trial[i] = max(x[i] - steps[i], 0.0)
                        x_trial = _project_to_feasible(x_trial, problem)
                        f_trial = _mean_reflectance_vec(materials, x_trial.tolist(), problem)
                        if f_trial < f:
                            x = x_trial
                            f = f_trial
                            improved = True
                            if f < best_f:
                                best_f = f
                                best_x = x.copy()
        
        if not improved:
            no_improve += 1
            if no_improve > 2:
                break
        
        # Check convergence
        if np.std(x) < 1e-8:
            break
    
    return best_x, best_f

def _gradient_step(materials, x, problem, lr=5.0):
    """One gradient descent step using finite differences."""
    n_params = len(x)
    if n_params == 0:
        return x
    
    f = _mean_reflectance_vec(materials, x.tolist(), problem)
    grad = np.zeros(n_params)
    h = 1.0
    
    for i in range(n_params):
        x_plus = x.copy()
        x_plus[i] = min(x[i] + h, problem["max_layer_nm"])
        x_plus = _project_to_feasible(x_plus, problem)
        f_plus = _mean_reflectance_vec(materials, x_plus.tolist(), problem)
        
        x_minus = x.copy()
        x_minus[i] = max(x[i] - h, 0.0)
        x_minus = _project_to_feasible(x_minus, problem)
        f_minus = _mean_reflectance_vec(materials, x_minus.tolist(), problem)
        
        if x_plus[i] != x_minus[i]:
            grad[i] = (f_plus - f_minus) / (x_plus[i] - x_minus[i])
        else:
            grad[i] = 0
    
    x_new = x - lr * grad
    x_new = _project_to_feasible(x_new, problem)
    return x_new

def _local_search(materials, initial_thicks, problem, time_limit):
    """
    Multi-stage local search: coordinate descent + gradient refinement.
    Returns (best_thicks, best_score).
    """
    x = np.array(initial_thicks, dtype=float)
    x = _project_to_feasible(x, problem)
    
    f = _mean_reflectance_vec(materials, x.tolist(), problem)
    best_f = f
    best_x = x.copy()
    
    # Stage 1: Coordinate descent with larger steps
    stage1_limit = min(time.time() + (time_limit - time.time()) * 0.6, time_limit - 5.0)
    x, f = _coordinate_descent(materials, x.tolist(), problem, stage1_limit, step=15.0)
    if f < best_f:
        best_f = f
        best_x = x.copy()
    
    # Stage 2: Coordinate descent with smaller steps for refinement
    stage2_limit = min(time.time() + (time_limit - time.time()) * 0.3, time_limit - 3.0)
    x, f = _coordinate_descent(materials, x.tolist(), problem, stage2_limit, step=3.0)
    if f < best_f:
        best_f = f
        best_x = x.copy()
    
    # Stage 3: Gradient-based refinement
    if time.time() < time_limit - 2.0:
        lr = 2.0
        for _ in range(200):
            if time.time() > time_limit - 1.0:
                break
            x_new = _gradient_step(materials, x, problem, lr)
            f_new = _mean_reflectance_vec(materials, x_new.tolist(), problem)
            if f_new < f:
                x = x_new
                f = f_new
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                lr *= 0.95
            else:
                lr *= 0.7
                if lr < 0.1:
                    break
    
    return best_x.tolist(), best_f

def _try_layer_reordering(materials, thicknesses, problem, time_limit):
    """
    Try swapping adjacent layers to see if reordering improves the result.
    Only effective for structures with repeating patterns.
    """
    n = len(materials)
    if n < 2:
        return materials, thicknesses, _mean_reflectance_vec(materials, thicknesses, problem)
    
    best_f = _mean_reflectance_vec(materials, thicknesses, problem)
    best_mats = materials[:]
    best_thicks = thicknesses[:]
    
    # Try all adjacent swaps
    for i in range(n - 1):
        if time.time() > time_limit:
            break
        new_mats = materials[:]
        new_thicks = thicknesses[:]
        new_mats[i], new_mats[i+1] = new_mats[i+1], new_mats[i]
        new_thicks[i], new_thicks[i+1] = new_thicks[i+1], new_thicks[i]
        
        if not _is_valid(new_mats, new_thicks, problem):
            continue
        
        f = _mean_reflectance_vec(new_mats, new_thicks, problem)
        if f < best_f:
            best_f = f
            best_mats = new_mats
            best_thicks = new_thicks
    
    return best_mats, best_thicks, best_f

def _generate_candidates(problem):
    """Generate diverse candidate structures with analytical initial thicknesses."""
    centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    
    dL = centre / (4.0 * nL)
    dH = centre / (4.0 * nH)
    
    candidates = []
    
    # Quarter-wave stacks with increasing layer count
    for n_layers in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]:
        mats = []
        thicks = []
        for i in range(n_layers):
            if i % 2 == 0:
                mats.append("L")
                thicks.append(dL)
            else:
                mats.append("H")
                thicks.append(dH)
        
        if _is_valid(mats, thicks, problem):
            candidates.append((mats, thicks))
    
    # Also try half-wave L only (for broadband)
    dL_half = centre / (2.0 * nL)
    candidates.append((["L"], [dL_half]))
    
    # Try asymmetric starting patterns
    # H-L-H-L-...
    for n_layers in [3, 5, 7, 9, 11, 13, 15, 17, 19]:
        mats = []
        thicks = []
        for i in range(n_layers):
            if i % 2 == 0:
                mats.append("H")
                thicks.append(dH)
            else:
                mats.append("L")
                thicks.append(dL)
        
        if _is_valid(mats, thicks, problem):
            candidates.append((mats, thicks))
    
    # Try some non-quarter-wave initializations
    # Scale factor variations
    for scale in [0.8, 1.2]:
        dL_s = dL * scale
        dH_s = dH * scale
        for n_layers in [5, 7, 9, 11]:
            mats = []
            thicks = []
            for i in range(n_layers):
                if i % 2 == 0:
                    mats.append("L")
                    thicks.append(dL_s)
                else:
                    mats.append("H")
                    thicks.append(dH_s)
            
            if _is_valid(mats, thicks, problem):
                candidates.append((mats, thicks))
    
    return candidates

def design(problem):
    """Design an antireflection coating stack."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    candidates = _generate_candidates(problem)
    
    # Evaluate all candidates
    scored_candidates = []
    for mats, thicks in candidates:
        if not _is_valid(mats, thicks, problem):
            continue
        score = _mean_reflectance_vec(mats, thicks, problem)
        scored_candidates.append((score, mats, thicks))
    
    if not scored_candidates:
        centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
        nL = problem["materials"]["L"]
        return (["L"], [centre / (4.0 * nL)])
    
    # Sort by score
    scored_candidates.sort(key=lambda x: x[0])
    
    best_score = scored_candidates[0][0]
    best_result = (scored_candidates[0][1], scored_candidates[0][2])
    
    # Optimize top candidates with local search
    # Allocate time budget based on remaining time
    num_to_optimize = min(8, len(scored_candidates))
    
    for i in range(num_to_optimize):
        if time.time() > deadline - 10.0:
            break
            
        score, mats, thicks = scored_candidates[i]
        
        # Skip if much worse than current best
        if score > best_score * 2.0 and i > 0:
            continue
            
        # Allocate time: give more time to better initial candidates
        remaining_candidates = num_to_optimize - i
        time_alloc = max(3.0, (deadline - time.time() - 10.0) / remaining_candidates)
        
        opt_thicks, opt_score = _local_search(mats, thicks, problem, time.time() + time_alloc)
        
        if opt_score < best_score:
            best_score = opt_score
            best_result = (mats, opt_thicks)
            
            # Try layer reordering if we have time
            if time.time() < deadline - 15.0 and len(mats) >= 2:
                reorder_limit = min(time.time() + 5.0, deadline - 10.0)
                new_mats, new_thicks, new_score = _try_layer_reordering(
                    mats, opt_thicks, problem, reorder_limit
                )
                if new_score < best_score:
                    best_score = new_score
                    best_result = (new_mats, new_thicks)
    
    # Final validation
    if not _is_valid(best_result[0], best_result[1], problem):
        # Fallback to a valid candidate
        for mats, thicks in scored_candidates:
            if _is_valid(mats, thicks, problem):
                return (mats, thicks)
        centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
        nL = problem["materials"]["L"]
        return (["L"], [centre / (4.0 * nL)])
    
    return best_result
# EVOLVE-BLOCK-END
