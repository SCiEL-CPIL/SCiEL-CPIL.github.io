"""Broadband antireflection coating design via Differential Evolution and local polish."""
import numpy as np
import time
import scipy.optimize as so

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

def _objective_de(x, indices, wavelengths_nm, ambient_idx, substrate_idx, max_total_nm, max_layer_nm):
    """
    Objective function for Differential Evolution.
    Returns mean reflectance in percent + penalty for constraint violation.
    """
    # Clip to bounds to ensure validity during search
    x_clipped = np.clip(x, 0.0, max_layer_nm)
    
    # Check total thickness constraint
    total_thick = np.sum(x_clipped)
    if total_thick > max_total_nm:
        # Penalty proportional to violation
        penalty = (total_thick - max_total_nm) * 0.1
        # Still compute reflectance to provide gradient-like info, but add penalty
        R = _compute_reflectance(wavelengths_nm, indices, x_clipped, ambient_idx, substrate_idx)
        return np.mean(R) * 100.0 + penalty
    
    R = _compute_reflectance(wavelengths_nm, indices, x_clipped, ambient_idx, substrate_idx)
    return np.mean(R) * 100.0

def _nelder_mead(f, x0, bounds, max_iter=2000, tol=1e-10, time_deadline=None):
    """
    Nelder-Mead simplex for local polish.
    """
    n = len(x0)
    lo_bounds = np.array([b[0] for b in bounds])
    hi_bounds = np.array([b[1] for b in bounds])
    x0 = np.clip(np.asarray(x0, dtype=float), lo_bounds, hi_bounds)
    
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    
    for i in range(n):
        simplex[i + 1] = x0.copy()
        lo, hi = bounds[i]
        step = 0.05 * (hi - lo) if hi > lo else 0.05
        simplex[i + 1][i] += step
    
    fvals = np.array([f(simplex[i]) for i in range(n + 1)])
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for it in range(max_iter):
        if time_deadline is not None and time.time() > time_deadline:
            break
        
        idx = np.argsort(fvals)
        simplex = simplex[idx]
        fvals = fvals[idx]
        
        if np.max(fvals) - np.min(fvals) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        x_r = centroid + alpha * (centroid - simplex[-1])
        x_r = np.clip(x_r, lo_bounds, hi_bounds)
        f_r = f(x_r)
        
        if f_r < fvals[0]:
            x_e = centroid + gamma * (x_r - centroid)
            x_e = np.clip(x_e, lo_bounds, hi_bounds)
            f_e = f(x_e)
            
            if f_e < f_r:
                simplex[-1] = x_e
                fvals[-1] = f_e
            else:
                simplex[-1] = x_r
                fvals[-1] = f_r
        elif f_r < fvals[-2]:
            simplex[-1] = x_r
            fvals[-1] = f_r
        else:
            x_c = centroid + rho * (simplex[-1] - centroid)
            x_c = np.clip(x_c, lo_bounds, hi_bounds)
            f_c = f(x_c)
            
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], lo_bounds, hi_bounds)
                    fvals[i] = f(simplex[i])
    
    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]

def design(problem):
    """
    Design a broadband antireflection coating.
    Returns: (labels, thicknesses)
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wavelengths_nm = np.linspace(band_nm[0], band_nm[1], 401)
    
    # Best result tracking
    best_labels = None
    best_thicknesses = None
    best_score = np.inf
    
    # Candidate sequences: Alternating L-H starting with L, ending with L
    # This is optimal for Air(1.0)-L(1.38)-...-L(1.38)-Glass(1.52)
    # Because L is closer to both Air and Glass than H is to Glass.
    # Sequence: L, H, L, H, ..., L
    # For N layers:
    # N=1: L
    # N=3: L H L
    # N=5: L H L H L
    # ...
    
    sequences = []
    # Try odd numbers of layers to end with L
    for n in [3, 5, 7, 9, 11, 13, 15, 17, 19]:
        if n > max_layers:
            continue
        labels = ['L' if i % 2 == 0 else 'H' for i in range(n)]
        sequences.append((n, labels))
        
        # Also try even numbers? L H L H ... H
        # Ends with H. H is 2.35, Glass is 1.52. Mismatch is large.
        # So odd numbers ending in L are better.
        
    # Also try a few even ones just in case, maybe L H L H ... L H? No, that ends in H.
    # What about starting with H? H L H L ...
    # Air(1.0) to H(2.35) is a big jump. L is better.
    
    # Sort sequences by preference: Fewer layers are simpler, but more layers give more freedom.
    # Usually 5-9 layers are good for broadband.
    # We will process them in order of increasing complexity, but allocate time based on potential.
    
    # To be efficient, we'll run DE on the most promising sequences.
    # Let's prioritize N=5, 7, 9, 11.
    
    # We will split the time budget among these sequences.
    # Let's say we spend 15s on each of 4 sequences = 60s. Plus polish.
    
    # Prepare DE parameters
    # DE is stochastic, so we use a fixed seed for reproducibility in this sandbox? 
    # No, we want to explore. But for debugging, fixed seed helps.
    # We'll use a different seed for each sequence to diversify.
    
    de_params = {
        'popsize': 15,
        'maxiter': 100,
        'mutation': (0.3, 0.9),
        'recombination': 0.7,
        'tol': 1e-6,
        'init': 'latinhypercube',
        'seed': 42
    }
    
    for n_layers, labels in sequences:
        if time.time() > deadline - 20:
            break
            
        indices = [materials[label] for label in labels]
        bounds = [(0.0, float(max_layer_nm))] * n_layers
        
        # Define objective for DE
        def f_de(x):
            return _objective_de(x, indices, wavelengths_nm, ambient_idx, substrate_idx, 
                                 max_total_nm, max_layer_nm)
        
        # Time budget for this sequence
        time_per_seq = 12.0
        seq_deadline = time.time() + time_per_seq
        
        try:
            # Run Differential Evolution
            # Note: DE might take longer if it doesn't converge.
            # We use a wrapper to check time? DE doesn't support time limits directly.
            # We can limit maxiter.
            # Let's estimate: 15 pop * 100 iter * 11 vars = 16500 evals.
            # Each eval is ~1ms? 16s. Too slow.
            # Reduce popsize and maxiter.
            # Popsize 10, Maxiter 50. 5000 evals. ~5s.
            
            de_params_seq = de_params.copy()
            de_params_seq['popsize'] = 10
            de_params_seq['maxiter'] = 50
            de_params_seq['seed'] = n_layers # Different seed per sequence
            
            result = so.differential_evolution(f_de, bounds, **de_params_seq)
            
            x_opt = result.x
            f_opt = result.fun
            
            # Check if valid
            if np.sum(x_opt) <= max_total_nm + 1e-6:
                # Local polish
                if time.time() < deadline - 10:
                    x_polished, f_polished = _nelder_mead(f_de, x_opt, bounds, 
                                                           max_iter=500, 
                                                           time_deadline=deadline - 5)
                    if f_polished < f_opt:
                        x_opt = x_polished
                        f_opt = f_polished
                
                if f_opt < best_score:
                    best_score = f_opt
                    best_labels = labels.copy()
                    best_thicknesses = x_opt.copy()
                    
        except Exception as e:
            pass # Continue to next sequence
    
    # If no valid result found, fallback to a simple 1-layer L
    if best_labels is None:
        best_labels = ["L"]
        best_thicknesses = [center_wl / (4.0 * materials["L"]) if 'center_wl' in locals() else 600 / (4.0 * materials["L"])]
        # Calculate center wl properly if needed
        center_wl = 0.5 * (band_nm[0] + band_nm[1])
        best_thicknesses = [center_wl / (4.0 * materials["L"])]
        
    # Final constraint check and clipping
    thicknesses = np.clip(best_thicknesses, 0, max_layer_nm)
    if np.sum(thicknesses) > max_total_nm:
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
        
    return best_labels, thicknesses.tolist()
