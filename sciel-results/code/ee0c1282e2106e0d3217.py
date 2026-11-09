"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _compute_reflectance(wavelengths, materials_list, thicknesses, n_ambient, n_substrate, n_materials):
    """Compute reflectance spectrum using characteristic matrix method (vectorized)."""
    # wavelengths: (N,) array in nm
    # materials_list: list of 'L'/'H'
    # thicknesses: list of floats in nm
    # Returns: reflectance (N,) as fraction (0-1)
    
    N = len(wavelengths)
    n_layers = len(materials_list)
    
    # Complex refractive indices (real, lossless)
    # k = 2*pi/lambda
    k = 2.0 * np.pi / wavelengths  # (N,)
    
    # Build transfer matrix: M = M_1 * M_2 * ... * M_N * M_substrate
    # Each layer matrix: [[cos(delta), i*sin(delta)/n], [i*n*sin(delta), cos(delta)]]
    # where delta = k * n * d (in vacuum wavelength terms)
    
    # Start with substrate matrix
    # Actually, the standard approach:
    # The total matrix M = product of layer matrices
    # Then Y = M[0,1]/M[0,0] (admittance at front)
    # R = |(n_ambient - Y)/(n_ambient + Y)|^2
    
    # Let's compute the product of all layer matrices
    # Initialize as identity
    M0 = np.ones((N, 2, 2), dtype=np.complex128)
    M1 = np.zeros((N, 2, 2), dtype=np.complex128)
    
    # Actually, let me be more careful. The characteristic matrix for a stack is:
    # M_total = M_1 * M_2 * ... * M_N
    # where M_j = [[cos(delta_j), i*sin(delta_j)/n_j], [i*n_j*sin(delta_j), cos(delta_j)]]
    
    # Then the optical admittance at the front surface (air side) is:
    # Y = M_total[0,1] / M_total[0,0]
    # R = |(n_0 - Y)/(n_0 + Y)|^2
    
    # Let's compute step by step
    # Start with identity matrix
    M00 = np.ones(N, dtype=np.complex128)
    M01 = np.zeros(N, dtype=np.complex128)
    M10 = np.zeros(N, dtype=np.complex128)
    M11 = np.ones(N, dtype=np.complex128)
    
    for j in range(n_layers):
        n_j = n_materials[materials_list[j]]
        d_j = thicknesses[j]
        delta_j = k * n_j * d_j  # (N,)
        
        cos_d = np.cos(delta_j)
        sin_d = np.sin(delta_j)
        
        # Layer matrix
        L00 = cos_d
        L01 = 1j * sin_d / n_j
        L10 = 1j * n_j * sin_d
        L11 = cos_d
        
        # M_new = M_old * L
        # M_new[0,0] = M00*L00 + M01*L10
        # M_new[0,1] = M00*L01 + M01*L11
        # M_new[1,0] = M10*L00 + M11*L10
        # M_new[1,1] = M10*L01 + M11*L11
        
        new_M00 = M00 * L00 + M01 * L10
        new_M01 = M00 * L01 + M01 * L11
        new_M10 = M10 * L00 + M11 * L10
        new_M11 = M10 * L01 + M11 * L11
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    # Admittance at front
    Y = M01 / M00
    R = np.abs((n_ambient - Y) / (n_ambient + Y)) ** 2
    
    return R

def _mean_reflectance_pct(materials_list, thicknesses, problem):
    """Compute mean reflectance in percent over the band."""
    band = problem["band_nm"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    n_materials = problem["materials"]
    
    wavelengths = np.linspace(band[0], band[1], 401)
    
    R = _compute_reflectance(wavelengths, materials_list, thicknesses, n_ambient, n_substrate, n_materials)
    
    return float(np.mean(R) * 100.0)

def _is_valid(materials_list, thicknesses, problem):
    """Check if the design is valid."""
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    if len(materials_list) != len(thicknesses):
        return False
    if len(materials_list) > max_layers:
        return False
    if len(materials_list) == 0:
        return False
    
    for t in thicknesses:
        if t < 0 or t > max_layer_nm:
            return False
    
    if sum(thicknesses) > max_total_nm:
        return False
    
    for m in materials_list:
        if m not in ('L', 'H'):
            return False
    
    return True

def _initial_design(problem):
    """Create an initial quarter-wave stack design."""
    band = problem["band_nm"]
    centre = 0.5 * (band[0] + band[1])
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    
    # Quarter-wave thicknesses at center wavelength
    d_L = centre / (4.0 * n_L)
    d_H = centre / (4.0 * n_H)
    
    # Try a few initial patterns
    candidates = []
    
    # Simple: L only
    candidates.append((["L"], [d_L]))
    
    # L-H-L
    candidates.append((["L", "H", "L"], [d_L, d_H, d_L]))
    
    # H-L-H
    candidates.append((["H", "L", "H"], [d_H, d_L, d_H]))
    
    # L-H-L-H-L
    candidates.append((["L", "H", "L", "H", "L"], [d_L, d_H, d_L, d_H, d_L]))
    
    # H-L-H-L-H
    candidates.append((["H", "L", "H", "L", "H"], [d_H, d_L, d_H, d_L, d_H]))
    
    # L-H-L-H-L-H-L
    candidates.append((["L", "H", "L", "H", "L", "H", "L"], [d_L, d_H, d_L, d_H, d_L, d_H, d_L]))
    
    # H-L-H-L-H-L-H
    candidates.append((["H", "L", "H", "L", "H", "L", "H"], [d_H, d_L, d_H, d_L, d_H, d_L, d_H]))
    
    # Find best candidate
    best = None
    best_score = float('inf')
    for mats, thicks in candidates:
        if _is_valid(mats, thicks, problem):
            score = _mean_reflectance_pct(mats, thicks, problem)
            if score < best_score:
                best_score = score
                best = (mats, thicks)
    
    if best is None:
        return ["L"], [d_L]
    
    return best[0], best[1]

def _nelder_mead(objective, x0, bounds, max_iter=500, tol=1e-8):
    """Simple Nelder-Mead optimizer."""
    n = len(x0)
    x = np.array(x0, dtype=float)
    
    # Create simplex
    simplex = [x.copy()]
    for i in range(n):
        xi = x.copy()
        step = 0.1 * abs(xi[i]) if xi[i] != 0 else 0.1
        xi[i] += step
        simplex.append(xi)
    
    # Evaluate
    fvals = [objective(s) for s in simplex]
    
    for it in range(max_iter):
        # Sort by function value
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Check convergence
        if abs(fvals[-1] - fvals[0]) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + (centroid - simplex[-1])
        fr = objective(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + 2.0 * (centroid - simplex[-1])
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = objective(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = objective(simplex[i])
    
    return simplex[0], fvals[0]

def _local_search(materials_list, thicknesses, problem, time_limit=30.0):
    """Perform local search to optimize thicknesses."""
    start_time = time.time()
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    n_layers = len(thicknesses)
    current_thicks = list(thicknesses)
    current_score = _mean_reflectance_pct(materials_list, current_thicks, problem)
    
    best_thicks = list(current_thicks)
    best_score = current_score
    
    # Step size
    step = 10.0  # nm
    
    while time.time() - start_time < time_limit:
        improved = False
        
        for i in range(n_layers):
            # Try increasing
            new_t = current_thicks[i] + step
            if 0 <= new_t <= max_layer_nm:
                new_thicks = list(current_thicks)
                new_thicks[i] = new_t
                if sum(new_thicks) <= max_total_nm:
                    score = _mean_reflectance_pct(materials_list, new_thicks, problem)
                    if score < best_score - 1e-10:
                        best_thicks = new_thicks
                        best_score = score
                        current_thicks = new_thicks
                        improved = True
            # Try decreasing
            new_t = current_thicks[i] - step
            if 0 <= new_t <= max_layer_nm:
                new_thicks = list(current_thicks)
                new_thicks[i] = new_t
                if sum(new_thicks) <= max_total_nm:
                    score = _mean_reflectance_pct(materials_list, new_thicks, problem)
                    if score < best_score - 1e-10:
                        best_thicks = new_thicks
                        best_score = score
                        current_thicks = new_thicks
                        improved = True
        
        if not improved:
            step *= 0.5
            if step < 0.1:
                break
    
    return best_thicks, best_score

def design(problem):
    """Design a broadband antireflection coating."""
    deadline = time.time() + 140.0  # Safety margin
    
    # Get initial design
    materials, thicknesses = _initial_design(problem)
    current_score = _mean_reflectance_pct(materials, thicknesses, problem)
    
    best_materials = list(materials)
    best_thicknesses = list(thicknesses)
    best_score = current_score
    
    # Try different layer counts and patterns
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    centre = 0.5 * (band[0] + band[1])
    d_L = centre / (4.0 * n_L)
    d_H = centre / (4.0 * n_H)
    
    # Generate candidate patterns
    patterns = []
    
    # For each number of layers, try alternating patterns
    for n_layers in range(1, 21):
        # Start with L
        mats_L = []
        thicks_L = []
        for i in range(n_layers):
            if i % 2 == 0:
                mats_L.append("L")
                thicks_L.append(d_L)
            else:
                mats_L.append("H")
                thicks_L.append(d_H)
        if sum(thicks_L) <= problem["max_total_nm"]:
            patterns.append((mats_L, thicks_L))
        
        # Start with H
        mats_H = []
        thicks_H = []
        for i in range(n_layers):
            if i % 2 == 0:
                mats_H.append("H")
                thicks_H.append(d_H)
            else:
                mats_H.append("L")
                thicks_H.append(d_L)
        if sum(thicks_H) <= problem["max_total_nm"]:
            patterns.append((mats_H, thicks_H))
    
    # Also try some non-alternating patterns
    # L-L-H-H-L-L
    for pattern in [["L", "L", "H", "H", "L", "L"], 
                    ["H", "H", "L", "L", "H", "H"],
                    ["L", "H", "H", "L", "L", "H"],
                    ["H", "L", "L", "H", "H", "L"]]:
        thicks = []
        for m in pattern:
            if m == "L":
                thicks.append(d_L)
            else:
                thicks.append(d_H)
        if sum(thicks) <= problem["max_total_nm"]:
            patterns.append((list(pattern), thicks))
    
    # Evaluate all patterns and pick top few
    scored_patterns = []
    for mats, thicks in patterns:
        if _is_valid(mats, thicks, problem):
            score = _mean_reflectance_pct(mats, thicks, problem)
            scored_patterns.append((score, mats, thicks))
    
    scored_patterns.sort(key=lambda x: x[0])
    
    # Optimize top patterns
    top_k = min(5, len(scored_patterns))
    
    for i in range(top_k):
        if time.time() > deadline - 20.0:
            break
        
        score, mats, thicks = scored_patterns[i]
        
        # Local search on thicknesses
        opt_thicks, opt_score = _local_search(mats, thicks, problem, time_limit=20.0)
        
        if opt_score < best_score:
            best_score = opt_score
            best_materials = list(mats)
            best_thicknesses = list(opt_thicks)
    
    # If we have time, try Nelder-Mead on the best design
    if time.time() < deadline - 10.0 and len(best_thicknesses) > 0:
        n_params = len(best_thicknesses)
        if n_params <= 10:  # Only for small problems
            x0 = np.array(best_thicknesses)
            bounds_lower = np.zeros(n_params)
            bounds_upper = np.full(n_params, problem["max_layer_nm"])
            
            def objective(x):
                thicks = list(x)
                if any(t < 0 or t > problem["max_layer_nm"] for t in thicks):
                    return 100.0
                if sum(thicks) > problem["max_total_nm"]:
                    return 100.0
                return _mean_reflectance_pct(best_materials, thicks, problem)
            
            x_opt, f_opt = _nelder_mead(objective, x0, (bounds_lower, bounds_upper), max_iter=200)
            
            if f_opt < best_score and _is_valid(best_materials, list(x_opt), problem):
                best_score = f_opt
                best_thicknesses = list(x_opt)
    
    # Final check
    if not _is_valid(best_materials, best_thicknesses, problem):
        # Fallback to simple design
        best_materials = ["L"]
        best_thicknesses = [centre / (4.0 * n_L)]
    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
