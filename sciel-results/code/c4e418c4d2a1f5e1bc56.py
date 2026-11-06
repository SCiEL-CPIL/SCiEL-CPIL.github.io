"""Broadband antireflection coating design with Simulated Annealing optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _reflectance_vectorized(wavelengths, thicknesses, indices, ambient_n, substrate_n):
    """Compute reflectance spectrum using vectorized characteristic matrix method."""
    n_layers = len(thicknesses)
    if n_layers == 0:
        r0 = (ambient_n - substrate_n) / (ambient_n + substrate_n)
        return np.full(len(wavelengths), np.abs(r0) ** 2)
    
    # Precompute phase factors
    # delta_j = 2*pi*n_j*d_j/lambda
    M00 = np.ones(len(wavelengths), dtype=complex)
    M01 = np.zeros(len(wavelengths), dtype=complex)
    M10 = np.zeros(len(wavelengths), dtype=complex)
    M11 = np.ones(len(wavelengths), dtype=complex)
    
    for j in range(n_layers):
        n_j = indices[j]
        d_j = thicknesses[j]
        
        if d_j < 1e-10:
            continue
        
        delta_j = 2.0 * np.pi * n_j * d_j / wavelengths
        cos_d = np.cos(delta_j)
        sin_d = np.sin(delta_j)
        
        # M_j elements
        m00 = cos_d
        m01 = 1j * sin_d / n_j
        m10 = 1j * n_j * sin_d
        m11 = cos_d
        
        # M = M @ M_j
        new_M00 = M00 * m00 + M01 * m10
        new_M01 = M00 * m01 + M01 * m11
        new_M10 = M10 * m00 + M11 * m10
        new_M11 = M10 * m01 + M11 * m11
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    # B = M[0,0] + M[0,1] * n_s
    # C = M[1,0] * n_0 + M[1,1] * n_s
    # R = |(C - B) / (C + B)|^2
    B = M00 + M01 * substrate_n
    C = M10 * ambient_n + M11 * substrate_n
    
    denom = C + B
    # Avoid division by zero
    denom_safe = np.where(np.abs(denom) < 1e-15, 1e-15, denom)
    
    r = (C - B) / denom_safe
    reflectance = np.abs(r) ** 2
    
    return reflectance

def _evaluate(materials, thicknesses, problem):
    """Compute mean reflectance in percent over the band."""
    wavelengths = np.linspace(problem["band_nm"][0], problem["band_nm"][1], 401)
    indices = np.array([problem["materials"][m] for m in materials], dtype=float)
    thicknesses_arr = np.array(thicknesses, dtype=float)
    refl = _reflectance_vectorized(wavelengths, thicknesses_arr, indices,
                                   problem["ambient_index"], problem["substrate_index"])
    return 100.0 * np.mean(refl)

def _is_valid(materials, thicknesses, problem):
    """Check if the design is valid."""
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > problem["max_layers"]:
        return False
    if len(materials) < 0:
        return False
    if any(t < -1e-9 or t > problem["max_layer_nm"] + 1e-9 for t in thicknesses):
        return False
    if sum(thicknesses) > problem["max_total_nm"] + 1e-9:
        return False
    if any(m not in problem["materials"] for m in materials):
        return False
    return True

def _nelder_mead(f, x0, bounds=None, maxiter=500, tol=1e-6):
    """
    Custom Nelder-Mead implementation to replace scipy.optimize.minimize.
    """
    x0 = np.array(x0, dtype=float)
    n = len(x0)
    
    # Initialize simplex
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        if xi[i] != 0:
            xi[i] *= 1.05
        else:
            xi[i] += 0.05
        simplex.append(xi)
        
    # Evaluate initial simplex
    fvals = [f(x) for x in simplex]
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for it in range(maxiter):
        # Sort simplex by function value
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Check convergence
        if abs(fvals[-1] - fvals[0]) < tol:
            break
        
        # Centroid of all points except the worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = centroid + alpha * (centroid - simplex[-1])
        f_r = f(x_r)
        
        if f_r < fvals[0]:
            # Expansion
            x_e = centroid + gamma * (x_r - centroid)
            f_e = f(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                fvals[-1] = f_e
            else:
                simplex[-1] = x_r
                fvals[-1] = f_r
        elif f_r < fvals[-2]:
            # Accept reflection
            simplex[-1] = x_r
            fvals[-1] = f_r
        else:
            # Contraction
            x_c = centroid + rho * (simplex[-1] - centroid)
            f_c = f(x_c)
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = f(simplex[i])
                    
    # Return best point
    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]

def _project_to_feasible(x, problem):
    """Project a candidate thickness vector to the feasible region."""
    x = np.array(x, dtype=float)
    # Clip to individual layer bounds
    x = np.clip(x, 0, problem["max_layer_nm"])
    # Check total thickness
    total = np.sum(x)
    if total > problem["max_total_nm"]:
        if total > 0:
            x = x * (problem["max_total_nm"] / total)
        else:
            x = np.zeros_like(x)
    return x

def design(problem):
    """Design antireflection coating using Random Restart Nelder-Mead."""
    start_time = time.time()
    time_limit = 145.0  # Leave 5s safety margin
    
    L_index = problem["materials"]["L"]
    H_index = problem["materials"]["H"]
    center_wl = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    best_score = float('inf')
    best_materials = ["L"]
    best_thicknesses = [center_wl / (4.0 * L_index)]
    
    # Precompute wavelengths for speed
    wavelengths = np.linspace(problem["band_nm"][0], problem["band_nm"][1], 401)
    
    # Generate a set of diverse material sequences to try
    sequences_to_try = []
    
    # 1. Quarter-wave stack of alternating L/H (good baseline)
    for n in range(2, max_layers + 1, 2):
        seq = ["L" if i % 2 == 0 else "H" for i in range(n)]
        sequences_to_try.append(seq)
        seq2 = ["H" if i % 2 == 0 else "L" for i in range(n)]
        sequences_to_try.append(seq2)
        
    # 2. Random sequences
    random.seed(42)
    for _ in range(50):
        n_layers = random.randint(1, max_layers)
        seq = [random.choice(["L", "H"]) for _ in range(n_layers)]
        sequences_to_try.append(seq)
        
    # Remove duplicates
    unique_sequences = []
    seen = set()
    for seq in sequences_to_try:
        key = tuple(seq)
        if key not in seen:
            seen.add(key)
            unique_sequences.append(seq)
            
    for materials in unique_sequences:
        if time.time() - start_time > time_limit:
            break
            
        n_layers = len(materials)
        indices = np.array([problem["materials"][m] for m in materials], dtype=float)
        
        # Initial guess: Quarter wave thicknesses
        x0 = []
        for m in materials:
            n_idx = problem["materials"][m]
            t_qw = center_wl / (4.0 * n_idx)
            x0.append(t_qw)
            
        x0 = np.array(x0, dtype=float)
        
        # Check if x0 is feasible, if not project it
        x0 = _project_to_feasible(x0, problem)
        
        # Define objective function for Nelder-Mead
        def objective(x):
            # Project to feasible to ensure we are evaluating a valid physical design
            x_feasible = _project_to_feasible(x, problem)
            
            # Evaluate reflectance
            refl = _reflectance_vectorized(wavelengths, x_feasible, indices,
                                           problem["ambient_index"], problem["substrate_index"])
            mean_refl = 100.0 * np.mean(refl)
            
            # Add penalty if the projected x is significantly different from x?
            # Or just rely on the projection.
            penalty = 0.0
            for xi in x:
                if xi < 0:
                    penalty += -xi * 10.0
                if xi > max_layer_nm:
                    penalty += (xi - max_layer_nm) * 10.0
            total_x = np.sum(x)
            if total_x > max_total_nm:
                penalty += (total_x - max_total_nm) * 10.0
                
            return mean_refl + penalty
        
        # Run Custom Nelder-Mead
        try:
            # Increase maxiter to compensate for lack of scipy's robustness, but keep it bounded
            # 500 iterations is usually enough for <20 dimensions
            x_opt, f_opt = _nelder_mead(objective, x0, maxiter=300, tol=1e-5)
            
            x_opt = _project_to_feasible(x_opt, problem)
            
            # Final validation
            if _is_valid(materials, x_opt.tolist(), problem):
                score = _evaluate(materials, x_opt.tolist(), problem)
                if score < best_score:
                    best_score = score
                    best_materials = list(materials)
                    best_thicknesses = [float(t) for t in x_opt]
        except Exception:
            pass
            
    # If no valid result found or all failed, return a simple 1-layer design
    if not _is_valid(best_materials, best_thicknesses, problem):
        best_materials = ["L"]
        best_thicknesses = [center_wl / (4.0 * L_index)]
    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
