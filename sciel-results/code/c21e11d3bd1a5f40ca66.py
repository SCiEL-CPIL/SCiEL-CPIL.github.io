"""
Broadband antireflection coating design via Differential Evolution.

Strategy:
- Use Differential Evolution (DE) for global optimization.
- Explore variable layer counts (10-20) and non-alternating index sequences.
- Allocate time budget to the most promising configurations.
- Ensure final output satisfies all constraints.
"""
import numpy as np
import time
from scipy.optimize import differential_evolution

def _compute_reflectance(wavelengths_nm, indices, thicknesses_nm, ambient_idx, substrate_idx):
    """
    Compute reflectance (fraction, 0-1) at each wavelength using the characteristic matrix method.
    """
    n_layers = len(indices)
    if n_layers == 0:
        r0 = (ambient_idx - substrate_idx) / (ambient_idx + substrate_idx)
        return np.full(len(wavelengths_nm), r0 * r0)
    
    k = 2.0 * np.pi / wavelengths_nm
    
    M00 = np.ones(len(wavelengths_nm), dtype=complex)
    M01 = np.zeros(len(wavelengths_nm), dtype=complex)
    M10 = np.zeros(len(wavelengths_nm), dtype=complex)
    M11 = np.ones(len(wavelengths_nm), dtype=complex)
    
    for i in range(n_layers):
        n_i = indices[i]
        d_i = thicknesses_nm[i]
        delta = k * n_i * d_i
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        B00 = cos_d
        B01 = 1j * sin_d / n_i
        B10 = 1j * n_i * sin_d
        B11 = cos_d
        
        new_M00 = M00 * B00 + M01 * B10
        new_M01 = M00 * B01 + M01 * B11
        new_M10 = M10 * B00 + M11 * B10
        new_M11 = M10 * B01 + M11 * B11
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    Y_out = substrate_idx
    Y_in = (M10 + M11 * Y_out) / (M00 + M01 * Y_out)
    
    r = (ambient_idx - Y_in) / (ambient_idx + Y_in)
    R = np.abs(r) ** 2
    
    return R

def _objective_de(thicknesses, indices, wavelengths_nm, ambient_idx, substrate_idx, 
                  max_layer_nm, max_total_nm):
    """
    Objective for DE: mean reflectance in percent. Returns large penalty if constraints violated.
    """
    thicknesses = np.asarray(thicknesses, dtype=float)
    if np.any(thicknesses < -1e-6) or np.any(thicknesses > max_layer_nm + 1e-6):
        return 100.0
    if np.sum(thicknesses) > max_total_nm + 1e-6:
        return 100.0
    
    thicknesses_clipped = np.clip(thicknesses, 0.0, max_layer_nm)
    if np.sum(thicknesses_clipped) > max_total_nm:
        return 100.0
    
    R = _compute_reflectance(wavelengths_nm, indices, thicknesses_clipped, 
                             ambient_idx, substrate_idx)
    return np.mean(R) * 100.0

def _generate_index_sequences(n_layers, max_layers):
    """
    Generate diverse index sequences for a given number of layers.
    We prioritize sequences that start with L (lower index) for better impedance matching with air (n=1).
    We also include some random sequences to explore non-alternating patterns.
    """
    sequences = []
    
    # 1. Alternating starting with L
    seq_l = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
    sequences.append(seq_l)
    
    # 2. Alternating starting with H
    seq_h = ['H' if i % 2 == 0 else 'L' for i in range(n_layers)]
    sequences.append(seq_h)
    
    # 3. Some random sequences (deterministic for reproducibility)
    rng = np.random.RandomState(123)
    for _ in range(5):
        # Ensure at least some L and H
        rand_seq = rng.choice(['L', 'H'], size=n_layers)
        # Force first layer to be L for better air matching
        rand_seq[0] = 'L'
        sequences.append(list(rand_seq))
        
    return sequences

def design(problem):
    """
    Design a broadband antireflection coating.
    Returns: (labels, thicknesses)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wavelengths_nm = np.linspace(band_nm[0], band_nm[1], 401)
    center_wl = 0.5 * (band_nm[0] + band_nm[1])
    
    best_labels = None
    best_thicknesses = None
    best_score = np.inf
    
    # Define a set of layer counts to try.
    # Higher layer counts generally allow for better broadband performance.
    # We will try a few representative counts.
    layer_counts_to_try = [12, 14, 16, 18, 20]
    
    # Filter out counts exceeding max_layers
    layer_counts_to_try = [lc for lc in layer_counts_to_try if lc <= max_layers]
    
    # If max_layers is small, adjust
    if not layer_counts_to_try:
        layer_counts_to_try = [min(10, max_layers)]
        
    # We will iterate through these counts.
    # For each count, we will try a few index sequences.
    # To manage time, we will allocate a portion of the remaining time to each (count, sequence) pair.
    
    # Total time budget for optimization
    total_opt_time = deadline - start_time - 10.0 # Leave 10s for final checks
    
    # Number of (count, sequence) combinations
    n_combos = 0
    for lc in layer_counts_to_try:
        n_combos += len(_generate_index_sequences(lc, max_layers))
        
    # Time per combination
    time_per_combo = max(1.0, total_opt_time / n_combos)
    
    for lc in layer_counts_to_try:
        if time.time() > deadline - 5.0:
            break
            
        sequences = _generate_index_sequences(lc, max_layers)
        
        for seq in sequences:
            if time.time() > deadline - 5.0:
                break
                
            indices = [materials[label] for label in seq]
            n_layers = len(indices)
            bounds = [(0.0, float(max_layer_nm))] * n_layers
            
            def f_de(x):
                return _objective_de(x, indices, wavelengths_nm, ambient_idx, substrate_idx,
                                     max_layer_nm, max_total_nm)
            
            # Run DE
            # maxiter is limited by time. We estimate iterations based on time_per_combo.
            # A rough estimate: 1 function eval ~ 0.1ms? No, it's vectorized over 401 wavelengths.
            # Let's just set a reasonable maxiter and rely on the timeout in scipy if available, 
            # or just check time after.
            # SciPy's DE doesn't have a built-in time limit, so we must be careful.
            # We will run DE with a moderate maxiter and check time afterwards.
            
            try:
                # Use a smaller population for speed? DE default is 15*N.
                # For N=20, pop=300. 300 evals * ~1ms = 0.3s.
                # Let's try maxiter=50. 50 * 300 = 15000 evals.
                # If 1 eval is 0.5ms, that's 7.5s. Too slow.
                # Let's estimate eval time.
                t0 = time.time()
                _ = f_de(np.array([100.0] * n_layers))
                t1 = time.time()
                eval_time = t1 - t0
                if eval_time < 1e-5: eval_time = 1e-5
                
                # Target time for this run
                target_time = min(time_per_combo, deadline - time.time() - 2.0)
                if target_time <= 0:
                    continue
                
                # Estimate max iterations
                # pop_size = 15 * n_layers
                pop_size = 15 * n_layers
                # evals per iter = pop_size
                max_iter_est = int(target_time / (pop_size * eval_time))
                max_iter_est = max(10, min(max_iter_est, 200)) # Cap to avoid runaway
                
                result = differential_evolution(
                    f_de, bounds, maxiter=max_iter_est, tol=1e-8, 
                    seed=42, polish=False # We can polish later if time permits
                )
                
                if result.success or result.fun < best_score:
                    # Check constraints on result
                    x_res = result.x
                    x_res = np.clip(x_res, 0, max_layer_nm)
                    if np.sum(x_res) <= max_total_nm:
                        # Calculate true score
                        score = _objective_de(x_res, indices, wavelengths_nm, ambient_idx, substrate_idx,
                                              max_layer_nm, max_total_nm)
                        if score < best_score:
                            best_score = score
                            best_labels = seq.copy()
                            best_thicknesses = x_res.copy()
            except Exception:
                pass
    
    # If no valid result found, fallback to a simple quarter-wave stack
    if best_labels is None:
        best_labels = ['L']
        best_thicknesses = np.array([center_wl / (4.0 * materials['L'])])
        best_score = _objective_de(best_thicknesses, [materials['L']], wavelengths_nm, ambient_idx, substrate_idx,
                                   max_layer_nm, max_total_nm)
        
    # Final constraint check and clipping
    thicknesses = np.array(best_thicknesses)
    thicknesses = np.clip(thicknesses, 0, max_layer_nm)
    if np.sum(thicknesses) > max_total_nm:
        # Scale down to fit
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
        
    return best_labels, thicknesses.tolist()
