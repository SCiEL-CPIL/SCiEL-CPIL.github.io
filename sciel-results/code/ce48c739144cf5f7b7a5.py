"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _compute_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance in percent over the band using characteristic matrix method."""
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    band = problem["band_nm"]
    wavelengths = np.linspace(band[0], band[1], 401)  # 401 equally spaced
    
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    
    n_layers = len(materials)
    if n_layers == 0:
        R = ((ns - n0) / (ns + n0)) ** 2
        return 100.0 * R
    
    # Precompute indices for each layer
    n_layers_arr = np.array([n_L if m == 'L' else n_H for m in materials])
    d_layers_arr = np.array(thicknesses)
    
    # Vectorize over wavelengths
    # delta_i = 2*pi*n_i*d_i / lambda
    # For each wavelength, compute the product of 2x2 matrices
    
    # We'll compute this in a loop over layers but vectorized over wavelengths
    # M_total starts as identity for each wavelength
    M11 = np.ones(len(wavelengths))
    M12 = np.zeros(len(wavelengths))
    M21 = np.zeros(len(wavelengths))
    M22 = np.ones(len(wavelengths))
    
    two_pi = 2.0 * np.pi
    for i in range(n_layers):
        delta = two_pi * n_layers_arr[i] * d_layers_arr[i] / wavelengths
        c = np.cos(delta)
        s = np.sin(delta)
        
        # Layer matrix: [[c, i*s/n], [i*n*s, c]]
        # But we're tracking real and imaginary parts... 
        # Actually at normal incidence with lossless materials, the matrices are complex.
        # Let me use complex arithmetic.
        pass
    
    # Actually, let me just use complex arrays - it's simpler and NumPy handles it
    M11 = np.ones(len(wavelengths), dtype=complex)
    M12 = np.zeros(len(wavelengths), dtype=complex)
    M21 = np.zeros(len(wavelengths), dtype=complex)
    M22 = np.ones(len(wavelengths), dtype=complex)
    
    for i in range(n_layers):
        delta = two_pi * n_layers_arr[i] * d_layers_arr[i] / wavelengths
        c = np.cos(delta)
        s = np.sin(delta)
        n_i = n_layers_arr[i]
        
        # Layer matrix elements
        L11 = c
        L12 = 1j * s / n_i
        L21 = 1j * n_i * s
        L22 = c
        
        # M_total = M_total @ L
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        
        M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
    
    # Admittance: Y = (M11 + M12 * ns) / (M21 + M22 * ns)
    Y = (M11 + M12 * ns) / (M21 + M22 * ns)
    
    # Reflectance: R = ((n0 - Y) / (n0 + Y))^2, take real part (should be real for lossless)
    R = np.abs((n0 - Y) / (n0 + Y)) ** 2
    
    return 100.0 * np.mean(R)


def _validate(materials, thicknesses, problem):
    """Check if design is valid."""
    max_layers = problem.get("max_layers", 20)
    max_total = problem.get("max_total_nm", 1000)
    max_layer = problem.get("max_layer_nm", 400)
    
    if len(materials) > max_layers:
        return False
    if len(materials) != len(thicknesses):
        return False
    for t in thicknesses:
        if t < 0 or t > max_layer:
            return False
    if sum(thicknesses) > max_total:
        return False
    return True


def _clip_thicknesses(thicknesses, problem):
    """Clip thicknesses to valid range."""
    max_layer = problem.get("max_layer_nm", 400)
    max_total = problem.get("max_total_nm", 1000)
    n = len(thicknesses)
    
    clipped = np.clip(thicknesses, 0.0, max_layer)
    total = np.sum(clipped)
    if total > max_total:
        # Scale down proportionally
        if total > 0:
            clipped = clipped * (max_total / total)
        clipped = np.clip(clipped, 0.0, max_layer)
    return clipped


def _coordinate_descent(materials, thicknesses, problem, max_iter=500, tol=1e-6):
    """Coordinate descent optimization on thicknesses."""
    thicknesses = np.array(thicknesses, dtype=float)
    n_layers = len(thicknesses)
    max_layer = problem.get("max_layer_nm", 400)
    max_total = problem.get("max_total_nm", 1000)
    
    current_val = _compute_reflectance(materials, thicknesses.tolist(), problem)
    
    for it in range(max_iter):
        improved = False
        for i in range(n_layers):
            # Try to optimize thickness[i] via line search
            best_t = thicknesses[i]
            best_v = current_val
            
            # Sample a few candidate values
            candidates = [
                best_t + 1.0, best_t - 1.0,
                best_t + 5.0, best_t - 5.0,
                best_t + 10.0, best_t - 10.0,
                best_t + 20.0, best_t - 20.0,
                0.0, max_layer,
            ]
            # Also try proportional
            if best_t > 0:
                candidates += [best_t * 0.8, best_t * 1.2, best_t * 0.5, best_t * 1.5]
            
            for cand in candidates:
                cand = max(0.0, min(max_layer, cand))
                new_th = thicknesses.copy()
                new_th[i] = cand
                
                # Check total constraint
                if np.sum(new_th) > max_total:
                    # Scale down
                    if np.sum(new_th) > 0:
                        new_th = new_th * (max_total / np.sum(new_th))
                    new_th = np.clip(new_th, 0.0, max_layer)
                
                v = _compute_reflectance(materials, new_th.tolist(), problem)
                if v < best_v - 1e-8:
                    best_v = v
                    best_t = cand
                    improved = True
            
            thicknesses[i] = best_t
        
        if not improved:
            break
        current_val = _compute_reflectance(materials, thicknesses.tolist(), problem)
    
    return materials, thicknesses.tolist(), current_val


def _nelder_mead(materials, thicknesses, problem, max_iter=200, tol=1e-6):
    """Nelder-Mead optimization on thicknesses with constraint handling."""
    n = len(thicknesses)
    max_layer = problem.get("max_layer_nm", 400)
    max_total = problem.get("max_total_nm", 1000)
    
    x0 = np.array(thicknesses, dtype=float)
    
    # Initialize simplex
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        step = 10.0 if x0[i] == 0 else x0[i] * 0.1
        xi[i] += step
        simplex.append(xi)
    
    def objective(x):
        x = np.clip(x, 0.0, max_layer)
        total = np.sum(x)
        if total > max_total:
            if total > 0:
                x = x * (max_total / total)
            x = np.clip(x, 0.0, max_layer)
        return _compute_reflectance(materials, x.tolist(), problem)
    
    # Evaluate initial simplex
    fvals = [objective(s) for s in simplex]
    
    for it in range(max_iter):
        # Sort by function value
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Check convergence
        if abs(fvals[-1] - fvals[0]) < tol:
            break
        
        x_worst = simplex[-1]
        x_centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        alpha = 1.0
        x_r = x_centroid + alpha * (x_centroid - x_worst)
        f_r = objective(x_r)
        
        if fvals[0] <= f_r < fvals[-2]:
            simplex[-1] = x_r
            fvals[-1] = f_r
        elif f_r < fvals[0]:
            # Expansion
            gamma = 2.0
            x_e = x_centroid + gamma * (x_r - x_worst)
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                fvals[-1] = f_e
            else:
                simplex[-1] = x_r
                fvals[-1] = f_r
        else:
            # Contraction
            rho = 0.5
            x_c = x_centroid + rho * (x_worst - x_centroid)
            f_c = objective(x_c)
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = objective(simplex[i])
    
    # Return best point
    order = np.argsort(fvals)
    best_idx = order[0]
    x_best = simplex[best_idx]
    x_best = np.clip(x_best, 0.0, max_layer)
    total = np.sum(x_best)
    if total > max_total:
        if total > 0:
            x_best = x_best * (max_total / total)
        x_best = np.clip(x_best, 0.0, max_layer)
    
    return materials, x_best.tolist(), fvals[best_idx]


def _generate_initial_guesses(problem, n_guesses=20):
    """Generate diverse initial guesses for the optimizer."""
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    center = 0.5 * (band[0] + band[1])
    max_layers = problem.get("max_layers", 20)
    max_total = problem.get("max_total_nm", 1000)
    
    guesses = []
    
    # Quarter-wave based designs with varying center wavelengths
    for n_layers in [1, 2, 3, 4, 5, 6, 7, 8]:
        if n_layers > max_layers:
            break
        for center_shift in [-100, -50, 0, 50, 100]:
            c = center + center_shift
            c = max(band[0] + 50, min(band[1] - 50, c))
            
            materials = []
            thicknesses = []
            for i in range(n_layers):
                if i % 2 == 0:
                    materials.append('L')
                    d = c / (4.0 * n_L)
                else:
                    materials.append('H')
                    d = c / (4.0 * n_H)
                thicknesses.append(d)
            
            # Check total
            total = sum(thicknesses)
            if total > max_total:
                scale = max_total / total
                thicknesses = [t * scale for t in thicknesses]
            
            if all(0 <= t <= 400 for t in thicknesses) and sum(thicknesses) <= max_total:
                guesses.append((materials, thicknesses))
    
    # Some random-ish variations
    np.random.seed(42)
    for _ in range(10):
        n_layers = np.random.randint(2, min(8, max_layers + 1))
        materials = []
        thicknesses = []
        for i in range(n_layers):
            mat = 'L' if i % 2 == 0 else 'H'
            materials.append(mat)
            n_mat = n_L if mat == 'L' else n_H
            d = np.random.uniform(50, 250)
            thicknesses.append(d)
        
        total = sum(thicknesses)
        if total > max_total:
            scale = max_total / total
            thicknesses = [t * scale for t in thicknesses]
        
        if all(0 <= t <= 400 for t in thicknesses) and sum(thicknesses) <= max_total:
            guesses.append((materials, thicknesses))
    
    # Limit total guesses
    return guesses[:n_guesses]


def design(problem):
    """Design an antireflection coating stack."""
    start_time = time.time()
    deadline = start_time + 120.0  # 120s budget, leave margin
    
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    center = 0.5 * (band[0] + band[1])
    
    # Best result so far
    best_materials = None
    best_thicknesses = None
    best_value = float('inf')
    
    # Baseline: single L quarter-wave layer
    d0 = center / (4.0 * n_L)
    baseline_materials = ['L']
    baseline_thicknesses = [d0]
    if _validate(baseline_materials, baseline_thicknesses, problem):
        best_materials = baseline_materials[:]
        best_thicknesses = baseline_thicknesses[:]
        best_value = _compute_reflectance(baseline_materials, baseline_thicknesses, problem)
    
    # Generate initial guesses
    guesses = _generate_initial_guesses(problem, n_guesses=30)
    
    # Optimize each guess
    for idx, (materials, thicknesses) in enumerate(guesses):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10.0:  # 10s safety margin
            break
        
        try:
            # Coordinate descent first
            mats_cd, th_cd, val_cd = _coordinate_descent(materials, thicknesses, problem, max_iter=300)
            
            if val_cd < best_value:
                best_materials = mats_cd[:]
                best_thicknesses = th_cd[:]
                best_value = val_cd
            
            # Nelder-Mead polish if time permits
            elapsed = time.time() - start_time
            if elapsed < deadline - 15.0:
                mats_nm, th_nm, val_nm = _nelder_mead(mats_cd, th_cd, problem, max_iter=150)
                
                if val_nm < best_value:
                    best_materials = mats_nm[:]
                    best_thicknesses = th_nm[:]
                    best_value = val_nm
                    
        except Exception:
            continue
    
    # Final polish on the best result if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 20.0 and best_materials is not None:
        try:
            mats_final, th_final, val_final = _nelder_mead(best_materials, best_thicknesses, problem, max_iter=300, tol=1e-8)
            if val_final < best_value:
                best_materials = mats_final[:]
                best_thicknesses = th_final[:]
                best_value = val_final
        except Exception:
            pass
    
    # Ensure validity
    if best_materials is None:
        return ['L'], [center / (4.0 * n_L)]
    
    # Final validation and clipping
    max_layer = problem.get("max_layer_nm", 400)
    max_total = problem.get("max_total_nm", 1000)
    best_thicknesses = [max(0.0, min(max_layer, t)) for t in best_thicknesses]
    total = sum(best_thicknesses)
    if total > max_total:
        scale = max_total / total
        best_thicknesses = [t * scale for t in best_thicknesses]
        best_thicknesses = [max(0.0, min(max_layer, t)) for t in best_thicknesses]
    
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
