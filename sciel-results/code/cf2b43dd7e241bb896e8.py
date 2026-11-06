"""Rewritten program: QWOT-based combinatorial design with simulated annealing."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _reflectance_spectrum(materials, thicknesses, ambient_index, substrate_index, wavelengths):
    """
    Compute normal-incidence reflectance (fraction) for a stack of layers.
    materials: list of refractive indices (real, lossless)
    thicknesses: list of physical thicknesses in nm
    wavelengths: array of wavelengths in nm
    Returns: array of reflectance fractions.
    """
    n0 = ambient_index
    ns = substrate_index
    n_layers = len(materials)
    
    if n_layers == 0:
        r = (n0 - ns) / (n0 + ns)
        return np.full(len(wavelengths), r**2)
    
    num_waves = len(wavelengths)
    R = np.zeros(num_waves)
    
    # Vectorized computation over all wavelengths
    # delta_j = n_j * d_j * 2*pi/lambda  for each layer j and wavelength iw
    inv_lambda = 2.0 * np.pi / wavelengths  # shape (num_waves,)
    
    # Precompute cos and sin for all layers and wavelengths
    # M00, M01, M10, M11 are complex arrays of shape (num_waves,)
    M00 = np.ones(num_waves, dtype=complex)
    M01 = np.zeros(num_waves, dtype=complex)
    M10 = np.zeros(num_waves, dtype=complex)
    M11 = np.ones(num_waves, dtype=complex)
    
    for j in range(n_layers):
        n_j = materials[j]
        d_j = thicknesses[j]
        delta_j = n_j * d_j * inv_lambda  # shape (num_waves,)
        c = np.cos(delta_j)
        s = np.sin(delta_j)
        
        # Characteristic matrix multiplication: M = M @ Mj
        # Mj = [[c, i*s/n_j], [i*n_j*s, c]]
        new_M00 = M00 * c + M01 * (1j * s / n_j)
        new_M01 = M00 * (1j * s / n_j) + M01 * c
        new_M10 = M10 * c + M11 * (1j * n_j * s)
        new_M11 = M10 * (1j * n_j * s) + M11 * c
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    B = M00 + M01 / ns
    C = M10 + M11 / ns
    
    Y = C / B
    r = (n0 - Y) / (n0 + Y)
    R = (r.real**2 + r.imag**2)
    
    return R

def _evaluate(materials_seq, scale, problem):
    """
    Evaluate a design given a sequence of 'L'/'H' and a QWOT scale factor.
    Each layer has optical thickness = scale * lambda_0/4 at the center wavelength.
    Physical thickness = scale * lambda_0 / (4 * n).
    """
    band = problem["band_nm"]
    lambda_0 = 0.5 * (band[0] + band[1])
    mat_dict = problem["materials"]
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    max_layers = problem["max_layers"]
    
    if len(materials_seq) > max_layers:
        return 100.0
    
    # Fast constraint check using vectorized sum
    n_vals = np.array([mat_dict[label] for label in materials_seq])
    thicknesses = scale * lambda_0 / (4.0 * n_vals)
    
    if np.any(thicknesses < 0) or np.any(thicknesses > max_layer_nm):
        return 100.0
    if np.sum(thicknesses) > max_total_nm:
        return 100.0
    
    # Compute reflectance
    wavelengths = problem["wavelengths"]
    R = _reflectance_spectrum(n_vals.tolist(), thicknesses.tolist(), problem["ambient_index"], 
                               problem["substrate_index"], wavelengths)
    mean_R = np.mean(R) * 100.0  # in percent
    return mean_R

def _optimize_scale_for_seq(materials_seq, problem):
    """
    For a fixed sequence, find the optimal scale factor by minimizing mean reflectance.
    Uses coarse-to-fine grid search.
    """
    band = problem["band_nm"]
    lambda_0 = 0.5 * (band[0] + band[1])
    mat_dict = problem["materials"]
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    if len(materials_seq) == 0:
        return 1.0, 100.0
    
    # Determine valid range for scale
    n_vals = np.array([mat_dict[label] for label in materials_seq])
    
    # scale * lambda_0 / (4*n) <= max_layer_nm for all n
    min_n = np.min(n_vals)
    max_scale = max_layer_nm * 4.0 * min_n / lambda_0
    
    # Total thickness: scale * lambda_0/4 * sum(1/n_j) <= max_total_nm
    sum_inv_n = np.sum(1.0 / n_vals)
    if sum_inv_n > 0:
        s_max_total = max_total_nm * 4.0 / (lambda_0 * sum_inv_n)
        max_scale = min(max_scale, s_max_total)
    
    min_scale = 0.01
    max_scale = max(max_scale, min_scale)
    
    # Coarse grid search
    best_s = 1.0
    best_score = 100.0
    n_coarse = 25
    for s in np.linspace(min_scale, max_scale, n_coarse):
        score = _evaluate(materials_seq, s, problem)
        if score < best_score:
            best_score = score
            best_s = s
    
    # Fine grid search around best
    s_range = max(0.05, (max_scale - min_scale) / n_coarse * 2)
    n_fine = 40
    for s in np.linspace(best_s - s_range, best_s + s_range, n_fine):
        if s < min_scale or s > max_scale:
            continue
        score = _evaluate(materials_seq, s, problem)
        if score < best_score:
            best_score = score
            best_s = s
    
    # Even finer
    s_range = max(0.01, s_range / 5)
    n_finer = 40
    for s in np.linspace(best_s - s_range, best_s + s_range, n_finer):
        if s < min_scale or s > max_scale:
            continue
        score = _evaluate(materials_seq, s, problem)
        if score < best_score:
            best_score = score
            best_s = s
    
    return best_s, best_score

def design(problem):
    """
    Design an antireflection coating using QWOT-based simulated annealing.
    """
    start_time = time.time()
    time_budget = 55.0  # Leave margin for safety
    
    band = problem["band_nm"]
    lambda_0 = 0.5 * (band[0] + band[1])
    mat_dict = problem["materials"]
    n_L = mat_dict["L"]
    n_H = mat_dict["H"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Precompute wavelengths
    wavelengths = np.linspace(band[0], band[1], 401)
    problem["wavelengths"] = wavelengths
    
    # Generate diverse initial sequences
    initial_seqs = []
    
    # Alternating patterns of various lengths
    for n in range(1, 21):
        seq_L = ['L' if i % 2 == 0 else 'H' for i in range(n)]
        seq_H = ['H' if i % 2 == 0 else 'L' for i in range(n)]
        initial_seqs.append(seq_L)
        initial_seqs.append(seq_H)
    
    # Some non-alternating patterns
    initial_seqs.append(['L', 'L', 'H', 'H', 'L', 'L'])
    initial_seqs.append(['H', 'H', 'L', 'L', 'H', 'H'])
    initial_seqs.append(['L', 'H', 'H', 'L', 'L', 'H'])
    initial_seqs.append(['H', 'L', 'L', 'H', 'H', 'L'])
    
    # Evaluate all initial sequences
    best_seq = initial_seqs[0]
    best_scale = 1.0
    best_score = 100.0
    
    for seq in initial_seqs:
        s, score = _optimize_scale_for_seq(seq, problem)
        if score < best_score:
            best_score = score
            best_seq = list(seq)
            best_scale = s
    
    # Simulated annealing with multiple restarts
    np.random.seed(42)
    
    num_restarts = 8
    iters_per_restart = 6000
    T_init = 0.3
    T_final = 0.0005
    
    for restart in range(num_restarts):
        # Check time
        if time.time() - start_time > time_budget - 12:
            break
        
        # Initialize current state
        if restart == 0:
            current_seq = list(best_seq)
            current_scale = best_scale
            current_score = best_score
        else:
            # Random restart with bias toward good structures
            n_layers = np.random.randint(2, max_layers + 1)
            if np.random.rand() < 0.5:
                # Alternating pattern with random start
                start_char = 'L' if np.random.rand() < 0.5 else 'H'
                current_seq = [start_char if i % 2 == 0 else ('H' if start_char == 'L' else 'L') for i in range(n_layers)]
            else:
                # Random
                current_seq = ['L' if np.random.rand() < 0.5 else 'H' for _ in range(n_layers)]
            current_scale = np.random.uniform(0.5, 2.0)
            current_score = _evaluate(current_seq, current_scale, problem)
            if current_score < best_score:
                best_score = current_score
                best_seq = list(current_seq)
                best_scale = current_scale
        
        cooling = (T_final / T_init) ** (1.0 / iters_per_restart)
        
        for i in range(iters_per_restart):
            # Check time
            if time.time() - start_time > time_budget - 5:
                break
            
            T = T_init * (cooling ** i)
            
            # Generate a neighbor
            r = np.random.rand()
            if r < 0.35:
                # Change a random layer
                new_seq = list(current_seq)
                if len(new_seq) > 0:
                    idx = np.random.randint(len(new_seq))
                    new_seq[idx] = 'H' if new_seq[idx] == 'L' else 'L'
                else:
                    new_seq = ['L']
                new_scale = current_scale
            elif r < 0.55:
                # Add a layer
                if len(current_seq) < max_layers:
                    new_seq = list(current_seq)
                    pos = np.random.randint(len(new_seq) + 1)
                    new_layer = 'L' if np.random.rand() < 0.5 else 'H'
                    new_seq.insert(pos, new_layer)
                else:
                    new_seq = list(current_seq)
                new_scale = current_scale
            elif r < 0.75:
                # Remove a layer
                if len(current_seq) > 1:
                    new_seq = list(current_seq)
                    idx = np.random.randint(len(new_seq))
                    new_seq.pop(idx)
                else:
                    new_seq = list(current_seq)
                new_scale = current_scale
            elif r < 0.90:
                # Perturb the scale
                new_seq = list(current_seq)
                new_scale = current_scale * (1.0 + 0.15 * np.random.randn())
                new_scale = max(0.3, min(3.0, new_scale))
            else:
                # Swap two adjacent layers
                new_seq = list(current_seq)
                if len(new_seq) >= 2:
                    idx = np.random.randint(len(new_seq) - 1)
                    new_seq[idx], new_seq[idx+1] = new_seq[idx+1], new_seq[idx]
                new_scale = current_scale
            
            # Evaluate
            new_score = _evaluate(new_seq, new_scale, problem)
            
            # Accept if better or with probability exp(-delta/T)
            if new_score < current_score:
                current_seq = new_seq
                current_scale = new_scale
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_seq = list(current_seq)
                    best_scale = current_scale
            else:
                delta = new_score - current_score
                if delta > 0 and T > 0:
                    accept_prob = np.exp(-delta / T)
                    if np.random.rand() < accept_prob:
                        current_seq = new_seq
                        current_scale = new_scale
                        current_score = new_score
    
    # Final refinement: optimize scale for best sequence
    best_scale, best_score = _optimize_scale_for_seq(best_seq, problem)
    
    # Try small modifications to best sequence (limited to stay within time)
    elapsed = time.time() - start_time
    if elapsed < time_budget - 5:
        max_trials = int((time_budget - 5 - elapsed) / 0.15)  # estimate ~0.15s per trial
        max_trials = min(max_trials, 100)
        for _ in range(max_trials):
            if time.time() - start_time > time_budget - 3:
                break
            trial_seq = list(best_seq)
            r = np.random.rand()
            if r < 0.4 and len(trial_seq) < max_layers:
                pos = np.random.randint(len(trial_seq) + 1)
                trial_seq.insert(pos, 'L' if np.random.rand() < 0.5 else 'H')
            elif r < 0.7 and len(trial_seq) > 1:
                idx = np.random.randint(len(trial_seq))
                trial_seq.pop(idx)
            else:
                if len(trial_seq) > 0:
                    idx = np.random.randint(len(trial_seq))
                    trial_seq[idx] = 'H' if trial_seq[idx] == 'L' else 'L'
            
            s, score = _optimize_scale_for_seq(trial_seq, problem)
            if score < best_score:
                best_score = score
                best_seq = list(trial_seq)
                best_scale = s
    
    # Convert back to labels and thicknesses
    materials_labels = best_seq
    thicknesses = []
    for label in best_seq:
        n = mat_dict[label]
        d = best_scale * lambda_0 / (4.0 * n)
        thicknesses.append(d)
    
    return materials_labels, thicknesses
# EVOLVE-BLOCK-END
