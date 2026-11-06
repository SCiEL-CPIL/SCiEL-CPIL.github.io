"""Broadband antireflection coating design using genetic algorithm + differential evolution."""
import time
import numpy as np
from scipy.optimize import differential_evolution

def _compute_R(labels, thicknesses, n_amb, n_sub, n_L, n_H, band_nm):
    """Compute mean reflectance in percent over 401 equally spaced wavelengths."""
    lam_min, lam_max = band_nm
    wavelengths = np.linspace(lam_min, lam_max, 401)
    
    if len(labels) == 0:
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return np.abs(r) ** 2 * 100.0
    
    n_mat = np.array([n_L if l == 'L' else n_H for l in labels], dtype=float)
    d = np.array(thicknesses, dtype=float)
    
    total_r = 0.0
    for start in range(0, len(wavelengths), 50):
        end = min(start + 50, len(wavelengths))
        lams = wavelengths[start:end]
        
        M = np.ones((len(lams), 2, 2), dtype=complex)
        for i in range(len(labels)):
            delta = 2.0 * np.pi * n_mat[i] * d[i] / lams
            c = np.cos(delta)
            s = np.sin(delta)
            
            Mi_00 = c
            Mi_01 = 1j * s / n_mat[i]
            Mi_10 = 1j * n_mat[i] * s
            Mi_11 = c
            
            M_new = np.empty_like(M)
            M_new[:, 0, 0] = M[:, 0, 0] * Mi_00 + M[:, 0, 1] * Mi_10
            M_new[:, 0, 1] = M[:, 0, 0] * Mi_01 + M[:, 0, 1] * Mi_11
            M_new[:, 1, 0] = M[:, 1, 0] * Mi_00 + M[:, 1, 1] * Mi_10
            M_new[:, 1, 1] = M[:, 1, 0] * Mi_01 + M[:, 1, 1] * Mi_11
            M = M_new
        
        Y = M[:, 1, 0] / M[:, 0, 0]
        r = (n_amb - Y) / (n_amb + Y)
        R = np.abs(r) ** 2 * 100.0
        total_r += np.sum(R)
    
    return total_r / len(wavelengths)


def design(problem):
    """Design a broadband antireflection coating stack using DE + NM polish."""
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s margin
    
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    centre = 0.5 * (band_nm[0] + band_nm[1])
    
    best_labels = None
    best_thicknesses = None
    best_score = float('inf')
    
    # Generate candidate sequences: alternating L/H patterns of various lengths
    candidate_sequences = []
    for n_layers in range(2, min(max_layers + 1, 12)):
        seq_L = ["L" if i % 2 == 0 else "H" for i in range(n_layers)]
        seq_H = ["H" if i % 2 == 0 else "L" for i in range(n_layers)]
        candidate_sequences.append(seq_L)
        if seq_H != seq_L:
            candidate_sequences.append(seq_H)
    
    # Add some non-alternating patterns
    extra_patterns = [
        ["L", "H", "H", "L", "L", "H"],
        ["H", "L", "L", "H", "H", "L"],
        ["L", "H", "L", "L", "H", "L"],
        ["H", "L", "H", "H", "L", "H"],
        ["L", "H", "H", "L", "H", "L"],
        ["H", "L", "L", "H", "L", "H"],
    ]
    for p in extra_patterns:
        if len(p) <= max_layers and p not in candidate_sequences:
            candidate_sequences.append(p)
    
    def objective(thicknesses, labels):
        """Objective function for DE: returns mean reflectance, or penalty if infeasible."""
        if np.any(thicknesses < 0) or np.any(thicknesses > max_layer_nm):
            return 100.0
        if np.sum(thicknesses) > max_total_nm:
            return 100.0
        return _compute_R(labels, list(thicknesses), n_amb, n_sub, n_L, n_H, band_nm)
    
    # For each candidate sequence, run differential evolution
    for labels in candidate_sequences:
        if time.time() > deadline - 20.0:
            break
        
        n = len(labels)
        if n > max_layers:
            continue
        
        # Initial guess: quarter-wave at center wavelength
        centre = 0.5 * (band_nm[0] + band_nm[1])
        x0 = np.array([centre / (4.0 * (n_L if l == 'L' else n_H)) for l in labels])
        
        # Check feasibility of initial guess
        if np.sum(x0) > max_total_nm:
            continue
        
        # Define bounds for DE
        bounds = [(0.0, max_layer_nm) for _ in range(n)]
        
        # Use DE with small population for speed
        try:
            result = differential_evolution(
                lambda x: objective(x, labels),
                bounds,
                maxiter=80,
                popsize=6,
                tol=1e-4,
                seed=42,
                polish=False,  # We'll do our own polish
                init='latinhypercube',
                update_interval=1,
                maxfevals=2000,
            )
        except Exception:
            continue
        
        if result.success or result.fun < best_score:
            opt_thick = result.x
            opt_score = result.fun
            if opt_score < best_score and all(0 <= t <= max_layer_nm for t in opt_thick) and sum(opt_thick) <= max_total_nm:
                best_score = opt_score
                best_labels = labels[:]
                best_thicknesses = list(opt_thick)
    
    # Polish the best result with a local optimization (simple coordinate descent)
    if best_labels is not None and time.time() < deadline - 15.0:
        labels = best_labels
        thicknesses = np.array(best_thicknesses, dtype=float)
        n = len(labels)
        
        # Simple coordinate descent with adaptive step
        step_sizes = [0.5 * (max_layer_nm / max(1, n)) for _ in range(n)]
        improved = True
        max_cd_iter = 200
        
        for cd_iter in range(max_cd_iter):
            if time.time() > deadline - 10.0:
                break
            if not improved:
                break
            improved = False
            
            current_score = _compute_R(labels, list(thicknesses), n_amb, n_sub, n_L, n_H, band_nm)
            
            for i in range(n):
                # Try positive and negative steps
                for direction in [1, -1]:
                    new_t = thicknesses.copy()
                    new_t[i] += direction * step_sizes[i]
                    
                    if new_t[i] < 0 or new_t[i] > max_layer_nm:
                        continue
                    if np.sum(new_t) > max_total_nm:
                        continue
                    
                    new_score = _compute_R(labels, list(new_t), n_amb, n_sub, n_L, n_H, band_nm)
                    
                    if new_score < current_score - 1e-6:
                        thicknesses = new_t
                        current_score = new_score
                        improved = True
                        # Shrink step size for this coordinate
                        step_sizes[i] *= 0.7
                    else:
                        # If no improvement, shrink step
                        step_sizes[i] *= 0.5
                
                # If step size too small, reset to a reasonable value
                if step_sizes[i] < 1.0:
                    step_sizes[i] = max(1.0, 0.1 * max_layer_nm / max(1, n))
        
        # Final check
        final_score = _compute_R(labels, list(thicknesses), n_amb, n_sub, n_L, n_H, band_nm)
        if final_score < best_score:
            best_score = final_score
            best_thicknesses = list(thicknesses)
    
    # If we have time, try a few random sequences with DE
    if time.time() < deadline - 25.0:
        rng = np.random.RandomState(42)
        for trial in range(10):
            if time.time() > deadline - 15.0:
                break
            
            n_layers = rng.randint(3, min(8, max_layers))
            labels = [rng.choice(['L', 'H']) for _ in range(n_layers)]
            
            # Skip if already tried
            if labels in candidate_sequences:
                continue
            
            centre = 0.5 * (band_nm[0] + band_nm[1])
            x0 = np.array([centre / (4.0 * (n_L if l == 'L' else n_H)) for l in labels])
            if np.sum(x0) > max_total_nm:
                continue
            
            bounds = [(0.0, max_layer_nm) for _ in range(n_layers)]
            
            try:
                result = differential_evolution(
                    lambda x: objective(x, labels),
                    bounds,
                    maxiter=50,
                    popsize=5,
                    tol=1e-4,
                    seed=int(rng.randint(0, 10000)),
                    polish=False,
                    init='latinhypercube',
                    maxfevals=1500,
                )
            except Exception:
                continue
            
            if result.fun < best_score:
                opt_thick = result.x
                if all(0 <= t <= max_layer_nm for t in opt_thick) and sum(opt_thick) <= max_total_nm:
                    best_score = result.fun
                    best_labels = labels[:]
                    best_thicknesses = list(opt_thick)
    
    # Fallback if nothing found
    if best_labels is None:
        best_labels = ["L"]
        best_thicknesses = [centre / (4.0 * n_L)]
    
    # Final constraint enforcement
    best_thicknesses = [max(0.0, min(t, max_layer_nm)) for t in best_thicknesses]
    if sum(best_thicknesses) > max_total_nm:
        scale = max_total_nm / sum(best_thicknesses)
        best_thicknesses = [t * scale for t in best_thicknesses]
    
    return best_labels, best_thicknesses
