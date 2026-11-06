"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

# EVOLVE-BLOCK-START
def _reflectance_spectrum(materials, thicknesses, problem, wavelengths=None):
    """Compute mean reflectance (%) over the band using characteristic matrix method."""
    if wavelengths is None:
        band = problem["band_nm"]
        wavelengths = np.linspace(band[0], band[1], 401)
    
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    n_mat = problem["materials"]
    
    # Build the stack: ambient | layers | substrate
    # Each layer has index n_i and optical thickness d_i (physical thickness)
    # Characteristic matrix for a layer at normal incidence:
    # M_i = [[cos(delta_i), i*sin(delta_i)/n_i],
    #        [i*n_i*sin(delta_i), cos(delta_i)]]
    # where delta_i = 2*pi*n_i*d_i/lambda
    
    # Total matrix M = M_1 * M_2 * ... * M_N
    # Y = (M11 + M12*n_sub) / (M21 + M22*n_sub)  -- admittance
    # R = |(Y - n_ambient) / (Y + n_ambient)|^2
    
    N = len(thicknesses)
    if N == 0:
        # Bare substrate
        r = (n_substrate - n_ambient) / (n_substrate + n_ambient)
        return 100.0 * r * r
    
    # Precompute for speed
    two_pi = 2.0 * np.pi
    n_layer_indices = np.array([n_mat[m] for m in materials])
    d_layer = np.array(thicknesses)
    
    # For each wavelength, compute the product of matrices
    # M_total = M_1 * M_2 * ... * M_N
    # We can vectorize over wavelengths
    
    # delta[i, j] = 2*pi*n_i*d_j / lambda_j
    # delta shape: (N, len(wavelengths))
    delta = two_pi * np.outer(n_layer_indices, 1.0 / wavelengths) * d_layer[:, np.newaxis]
    
    cos_d = np.cos(delta)  # (N, W)
    sin_d = np.sin(delta)  # (N, W)
    
    # Matrix elements for each layer and wavelength
    # M11 = cos(d), M12 = i*sin(d)/n, M21 = i*n*sin(d), M22 = cos(d)
    # We'll propagate the 2x2 matrix product
    
    # Initialize total matrix as identity
    # M_total = [[1, 0], [0, 1]]
    # We multiply M_total = M_total * M_i for i in 0..N-1
    
    # For efficiency, we store the 4 elements as arrays of shape (W,)
    # Start with identity
    M11 = np.ones(len(wavelengths), dtype=complex)
    M12 = np.zeros(len(wavelengths), dtype=complex)
    M21 = np.zeros(len(wavelengths), dtype=complex)
    M22 = np.ones(len(wavelengths), dtype=complex)
    
    for i in range(N):
        c = cos_d[i]
        s = sin_d[i]
        n_i = n_layer_indices[i]
        
        # M_i = [[c, i*s/n_i], [i*n_i*s, c]]
        m11 = c
        m12 = 1j * s / n_i
        m21 = 1j * n_i * s
        m22 = c
        
        # New M = M_old * M_i
        new_M11 = M11 * m11 + M12 * m21
        new_M12 = M11 * m12 + M12 * m22
        new_M21 = M21 * m11 + M22 * m21
        new_M22 = M21 * m12 + M22 * m22
        
        M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
    
    # Admittance Y = (M11 + M12 * n_sub) / (M21 + M22 * n_sub)
    Y = (M11 + M12 * n_substrate) / (M21 + M22 * n_substrate)
    
    # Reflectance R = |(Y - n_amb) / (Y + n_amb)|^2
    r = (Y - n_ambient) / (Y + n_ambient)
    R = np.abs(r) ** 2
    
    return 100.0 * np.mean(R)


def _evaluate_solution(materials, thicknesses, problem):
    """Evaluate a solution, returning (valid, mean_reflectance)."""
    # Check constraints
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    if len(materials) > max_layers:
        return False, float('inf')
    if len(materials) != len(thicknesses):
        return False, float('inf')
    for t in thicknesses:
        if t < 0 or t > max_layer_nm:
            return False, float('inf')
    if sum(thicknesses) > max_total_nm:
        return False, float('inf')
    for m in materials:
        if m not in problem["materials"]:
            return False, float('inf')
    
    try:
        r = _reflectance_spectrum(materials, thicknesses, problem)
        return True, r
    except Exception:
        return False, float('inf')


def design(problem):
    """Design an antireflection coating using global optimization."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    band = problem["band_nm"]
    center_wl = 0.5 * (band[0] + band[1])
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_sub = problem["substrate_index"]
    
    # Start with a good initial guess: quarter-wave stack
    # For broadband AR, use multiple quarter-wave layers
    # A common approach: H-L-H-L-H or L-H-L-H-L etc.
    
    # Let's try several initial designs and pick the best
    initial_designs = []
    
    # Design 1: Single quarter-wave low-index
    d_L = center_wl / (4.0 * n_L)
    initial_designs.append((["L"], [d_L]))
    
    # Design 2: Single quarter-wave high-index
    d_H = center_wl / (4.0 * n_H)
    initial_designs.append((["H"], [d_H]))
    
    # Design 3: Two-layer quarter-wave (L-H)
    initial_designs.append((["L", "H"], [d_L, d_H]))
    initial_designs.append((["H", "L"], [d_H, d_L]))
    
    # Design 4: Three-layer quarter-wave (H-L-H)
    initial_designs.append((["H", "L", "H"], [d_H, d_L, d_H]))
    initial_designs.append((["L", "H", "L"], [d_L, d_H, d_L]))
    
    # Design 5: Four-layer
    initial_designs.append((["H", "L", "H", "L"], [d_H, d_L, d_H, d_L]))
    initial_designs.append((["L", "H", "L", "H"], [d_L, d_H, d_L, d_H]))
    
    # Design 6: Five-layer
    initial_designs.append((["H", "L", "H", "L", "H"], [d_H, d_L, d_H, d_L, d_H]))
    initial_designs.append((["L", "H", "L", "H", "L"], [d_L, d_H, d_L, d_H, d_L]))
    
    # Evaluate initial designs
    best_r = float('inf')
    best_materials = None
    best_thicknesses = None
    for mats, thicks in initial_designs:
        valid, r = _evaluate_solution(mats, thicks, problem)
        if valid and r < best_r:
            best_r = r
            best_materials = list(mats)
            best_thicknesses = list(thicks)
    
    # Now use differential evolution to optimize thicknesses for a fixed layer count
    # We'll try several layer counts and patterns
    
    def _optimize_thicknesses(fixed_materials, max_iter=50, tol=1e-4):
        """Optimize thicknesses for a fixed material sequence."""
        n_layers = len(fixed_materials)
        if n_layers == 0:
            return None, float('inf')
        
        max_layer_nm = problem.get("max_layer_nm", 400)
        max_total_nm = problem.get("max_total_nm", 1000)
        
        # Bounds: each thickness in [0, max_layer_nm], but sum <= max_total_nm
        # DE doesn't handle sum constraint directly, so we'll use a penalty
        bounds = [(0.0, max_layer_nm)] * n_layers
        
        def objective(x):
            # Penalty for exceeding total thickness
            if sum(x) > max_total_nm:
                return 100.0 + (sum(x) - max_total_nm) / max_total_nm
            mats = list(fixed_materials)
            thicks = list(x)
            valid, r = _evaluate_solution(mats, thicks, problem)
            if not valid:
                return 100.0
            return r
        
        # Use differential evolution
        try:
            result = differential_evolution(
                objective, bounds,
                maxiter=max_iter, tol=tol,
                popsize=10, seed=42,
                polish=True
            )
            if result.success or result.fun < 100.0:
                thicks = list(result.x)
                # Ensure constraints
                for i in range(n_layers):
                    thicks[i] = max(0.0, min(thicks[i], max_layer_nm))
                if sum(thicks) > max_total_nm:
                    # Scale down
                    scale = max_total_nm / sum(thicks)
                    thicks = [t * scale for t in thicks]
                valid, r = _evaluate_solution(fixed_materials, thicks, problem)
                if valid:
                    return thicks, r
        except Exception:
            pass
        return None, float('inf')
    
    # Try optimizing for various layer counts
    layer_counts_to_try = [1, 2, 3, 4, 5, 6, 8, 10]
    
    for n_layers in layer_counts_to_try:
        if time.time() > deadline:
            break
        
        # Try alternating patterns starting with L and H
        for start_char in ["L", "H"]:
            if time.time() > deadline:
                break
            
            mats = []
            for i in range(n_layers):
                if i % 2 == 0:
                    mats.append(start_char)
                else:
                    mats.append("H" if start_char == "L" else "L")
            
            thicks, r = _optimize_thicknesses(mats, max_iter=30, tol=1e-3)
            if thicks is not None and r < best_r:
                best_r = r
                best_materials = list(mats)
                best_thicknesses = list(thicks)
    
    # Also try some non-alternating patterns
    if time.time() < deadline:
        special_patterns = [
            ["H", "L", "L", "H"],
            ["L", "H", "H", "L"],
            ["H", "L", "H", "L", "H"],
            ["L", "H", "L", "H", "L"],
            ["H", "L", "L", "L", "H"],
            ["L", "H", "H", "H", "L"],
            ["H", "L", "H", "L", "H", "L"],
            ["L", "H", "L", "H", "L", "H"],
        ]
        for mats in special_patterns:
            if time.time() > deadline:
                break
            thicks, r = _optimize_thicknesses(mats, max_iter=25, tol=1e-3)
            if thicks is not None and r < best_r:
                best_r = r
                best_materials = list(mats)
                best_thicknesses = list(thicks)
    
    # If we still have time, do a local refinement on the best solution
    if time.time() < deadline and best_materials is not None and len(best_materials) > 0:
        try:
            def _local_obj(x):
                if sum(x) > problem.get("max_total_nm", 1000):
                    return 100.0
                return _reflectance_spectrum(best_materials, list(x), problem)
            
            x0 = np.array(best_thicknesses)
            bounds = [(0.0, problem.get("max_layer_nm", 400))] * len(x0)
            
            result = minimize(
                _local_obj, x0, method="L-BFGS-B",
                bounds=bounds, options={"maxiter": 100}
            )
            if result.fun < best_r:
                thicks = [max(0.0, min(float(t), problem.get("max_layer_nm", 400))) for t in result.x]
                if sum(thicks) <= problem.get("max_total_nm", 1000):
                    valid, r = _evaluate_solution(best_materials, thicks, problem)
                    if valid and r < best_r:
                        best_r = r
                        best_thicknesses = thicks
        except Exception:
            pass
    
    # Final check
    if best_materials is None or best_thicknesses is None:
        # Fallback: single quarter-wave layer
        d_L = center_wl / (4.0 * n_L)
        best_materials = ["L"]
        best_thicknesses = [d_L]
    
    # Ensure all constraints are met
    max_layer_nm = problem.get("max_layer_nm", 400)
    max_total_nm = problem.get("max_total_nm", 1000)
    
    for i in range(len(best_thicknesses)):
        best_thicknesses[i] = max(0.0, min(best_thicknesses[i], max_layer_nm))
    
    if sum(best_thicknesses) > max_total_nm:
        scale = max_total_nm / sum(best_thicknesses)
        best_thicknesses = [t * scale for t in best_thicknesses]
    
    # Remove any zero-thickness layers
    final_materials = []
    final_thicknesses = []
    for m, t in zip(best_materials, best_thicknesses):
        if t > 0.1:  # Keep layers with at least 0.1 nm
            final_materials.append(m)
            final_thicknesses.append(t)
    
    if len(final_materials) == 0:
        d_L = center_wl / (4.0 * n_L)
        final_materials = ["L"]
        final_thicknesses = [d_L]
    
    return final_materials, final_thicknesses
# EVOLVE-BLOCK-END
