"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _mean_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance (percent) over the band using characteristic matrix method."""
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    wls = np.linspace(band[0], band[1], 401)
    n_mat = np.array([n_L if m == 'L' else n_H for m in materials])
    d = np.array(thicknesses, dtype=float)
    total_R = 0.0
    for wl in wls:
        # Build characteristic matrix product
        M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        for i in range(len(materials)):
            n_i = n_mat[i]
            phase = 2.0 * np.pi * n_i * d[i] / wl
            cos_p = np.cos(phase)
            sin_p = np.sin(phase)
            B = np.array([[cos_p, 1j * sin_p / n_i],
                          [1j * n_i * sin_p, cos_p]], dtype=complex)
            M = M @ B
        # Reflectance at normal incidence
        Y = M[0, 1] / M[1, 1]
        r = (n_amb - Y) / (n_amb + Y)
        R = np.abs(r) ** 2
        total_R += R * 100.0
    return total_R / len(wls)

def _optimize_thicknesses(materials, thicknesses, problem, deadline):
    """Coordinate descent optimization of thicknesses for a fixed material sequence."""
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    best_thick = list(thicknesses)
    best_score = _mean_reflectance(materials, best_thick, problem)
    n_layers = len(materials)
    
    # Multiple rounds of coordinate descent
    for round_num in range(5):
        if time.time() > deadline - 30:
            break
        improved = False
        for i in range(n_layers):
            if time.time() > deadline - 30:
                break
            # Try perturbations around current value
            current = best_thick[i]
            for delta in [0.05, -0.05, 0.1, -0.1, 0.2, -0.2, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0]:
                new_val = current + delta
                if new_val < 0 or new_val > max_layer_nm:
                    continue
                trial = best_thick[:]
                trial[i] = new_val
                if sum(trial) > max_total_nm:
                    continue
                score = _mean_reflectance(materials, trial, problem)
                if score < best_score:
                    best_score = score
                    best_thick = trial
                    current = new_val
                    improved = True
        if not improved:
            break
    
    return best_mats_check, best_thick, best_score

def design(problem):
    """Multi-layer quarter-wave stack with local optimization for broadband AR coating."""
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin before 150s kill

    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    centre = 0.5 * (band[0] + band[1])
    
    random.seed(42)

    # Fallback: single quarter-wave layer
    best_mats = ["L"]
    best_thick = [centre / (4.0 * n_L)]
    best_score = _mean_reflectance(best_mats, best_thick, problem)

    # Try multi-layer quarter-wave stacks with alternating pattern
    for n_layers in range(2, min(20, max_layers + 1)):
        if time.time() > deadline - 40:
            break
        mats = []
        thick = []
        for i in range(n_layers):
            if i % 2 == 0:
                mats.append('L')
                thick.append(centre / (4.0 * n_L))
            else:
                mats.append('H')
                thick.append(centre / (4.0 * n_H))
        
        if sum(thick) > max_total_nm or any(t > max_layer_nm for t in thick):
            continue
        
        score = _mean_reflectance(mats, thick, problem)
        if score < best_score:
            best_score = score
            best_mats = mats[:]
            best_thick = thick[:]
    
    # Simulated annealing on discrete layer sequences with continuous thickness optimization
    if time.time() < deadline - 40:
        best_mats_sa = best_mats[:]
        best_thick_sa = best_thick[:]
        best_score_sa = best_score
        current_mats = best_mats[:]
        current_thick = best_thick[:]
        current_score = best_score
        
        T = 2.0  # initial temperature
        cooling = 0.98
        max_sa_iters = 5000
        
        for sa_iter in range(max_sa_iters):
            if time.time() > deadline - 40:
                break
            
            # Random mutation of material sequence
            new_mats = current_mats[:]
            if len(new_mats) < max_layers:
                # Add a layer at random position
                pos = random.randint(0, len(new_mats))
                new_mats.insert(pos, random.choice(['L', 'H']))
            elif len(new_mats) > 1:
                # Remove a layer at random position
                pos = random.randint(0, len(new_mats) - 1)
                new_mats.pop(pos)
            else:
                # Swap or change a layer
                if random.random() < 0.5:
                    pos = random.randint(0, len(new_mats) - 1)
                    new_mats[pos] = 'L' if new_mats[pos] == 'H' else 'H'
                else:
                    # Swap two layers
                    if len(new_mats) >= 2:
                        p1, p2 = random.sample(range(len(new_mats)), 2)
                        new_mats[p1], new_mats[p2] = new_mats[p2], new_mats[p1]
            
            # Initialize thicknesses based on quarter-wave values
            new_thick = []
            for m in new_mats:
                if m == 'L':
                    new_thick.append(centre / (4.0 * n_L))
                else:
                    new_thick.append(centre / (4.0 * n_H))
            
            if sum(new_thick) > max_total_nm or any(t > max_layer_nm for t in new_thick):
                continue
            
            # Quick evaluation first
            quick_score = _mean_reflectance(new_mats, new_thick, problem)
            
            # Accept or reject based on Metropolis criterion
            delta = quick_score - current_score
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Accept: do local thickness optimization
                if time.time() < deadline - 40:
                    # Coordinate descent on thicknesses
                    local_thick = new_thick[:]
                    local_score = quick_score
                    n_layers_local = len(new_mats)
                    
                    for cd_round in range(3):
                        if time.time() > deadline - 40:
                            break
                        for i in range(n_layers_local):
                            if time.time() > deadline - 40:
                                break
                            current_t = local_thick[i]
                            for d_val in [0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0]:
                                new_t = current_t + d_val
                                if new_t < 0 or new_t > max_layer_nm:
                                    continue
                                trial = local_thick[:]
                                trial[i] = new_t
                                if sum(trial) > max_total_nm:
                                    continue
                                s = _mean_reflectance(new_mats, trial, problem)
                                if s < local_score:
                                    local_score = s
                                    local_thick = trial
                                    current_t = new_t
                    
                    current_mats = new_mats
                    current_thick = local_thick
                    current_score = local_score
                    
                    if current_score < best_score_sa:
                        best_score_sa = current_score
                        best_mats_sa = current_mats[:]
                        best_thick_sa = current_thick[:]
            
            T *= cooling
        
        if best_score_sa < best_score:
            best_score = best_score_sa
            best_mats = best_mats_sa
            best_thick = best_thick_sa
    
    # Final local optimization with Nelder-Mead if time permits
    if time.time() < deadline - 10 and len(best_mats) >= 2:
        try:
            from scipy.optimize import minimize
            n_params = len(best_thick)
            x0 = np.array(best_thick, dtype=float)
            
            def obj(x):
                if np.any(x < 0) or np.any(x > max_layer_nm) or np.sum(x) > max_total_nm:
                    return 100.0
                return _mean_reflectance(best_mats, list(x), problem)
            
            res = minimize(obj, x0, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-8})
            
            if res.success and res.fun < best_score:
                x_opt = np.clip(res.x, 0, max_layer_nm)
                if np.sum(x_opt) <= max_total_nm:
                    best_score = float(res.fun)
                    best_thick = list(x_opt)
        except Exception:
            pass
    
    return best_mats, best_thick
# EVOLVE-BLOCK-END
