"""Broadband antireflection coating design via discrete-continuous hybrid search."""
import time
import math
import numpy as np

def _reflectance_array(materials, thicknesses, wls, n_ambient, n_substrate, n_dict):
    """
    Compute reflectance at all wavelengths in wls using characteristic matrix method.
    Returns array of reflectance values (0 to 1).
    """
    n_layers = len(materials)
    n_wl = len(wls)
    
    if n_layers == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return np.full(n_wl, abs(r) ** 2)
    
    # Precompute beta for each layer and wavelength
    # beta[i, j] = 2*pi * n_i * d_i / wl_j
    betas = np.zeros((n_layers, n_wl))
    for i in range(n_layers):
        n_mat = n_dict[materials[i]]
        d_mat = thicknesses[i]
        betas[i, :] = 2.0 * np.pi * n_mat * d_mat / wls
    
    cos_betas = np.cos(betas)
    sin_betas = np.sin(betas)
    
    # Build total characteristic matrix for each wavelength
    # M_total[j] = M_0 @ M_1 @ ... @ M_{n-1} for wavelength j
    M_total = np.zeros((n_wl, 2, 2), dtype=complex)
    M_total[:, 0, 0] = 1.0
    M_total[:, 1, 1] = 1.0
    
    for i in range(n_layers):
        n_mat = n_dict[materials[i]]
        # M_i[j] = [[cos_b, i*sin_b/n], [i*n*sin_b, cos_b]]
        M_i = np.zeros((n_wl, 2, 2), dtype=complex)
        M_i[:, 0, 0] = cos_betas[i, :]
        M_i[:, 0, 1] = 1j * sin_betas[i, :] / n_mat
        M_i[:, 1, 0] = 1j * n_mat * sin_betas[i, :]
        M_i[:, 1, 1] = cos_betas[i, :]
        
        # M_total = M_total @ M_i (element-wise matrix multiplication over wavelengths)
        # (M_total @ M_i)[j] = M_total[j] @ M_i[j]
        new_M = np.zeros_like(M_total)
        new_M[:, 0, 0] = M_total[:, 0, 0] * M_i[:, 0, 0] + M_total[:, 0, 1] * M_i[:, 1, 0]
        new_M[:, 0, 1] = M_total[:, 0, 0] * M_i[:, 0, 1] + M_total[:, 0, 1] * M_i[:, 1, 1]
        new_M[:, 1, 0] = M_total[:, 1, 0] * M_i[:, 0, 0] + M_total[:, 1, 1] * M_i[:, 1, 0]
        new_M[:, 1, 1] = M_total[:, 1, 0] * M_i[:, 0, 1] + M_total[:, 1, 1] * M_i[:, 1, 1]
        M_total = new_M
    
    # Compute admittance Y = (M11 + M12 * n_s) / (M21 + M22 * n_s)
    Y = (M_total[:, 0, 0] + M_total[:, 0, 1] * n_substrate) / (M_total[:, 1, 0] + M_total[:, 1, 1] * n_substrate)
    
    # Reflectance
    r = (n_ambient - Y) / (n_ambient + Y)
    return np.abs(r) ** 2

def _mean_reflectance_percent(materials, thicknesses, problem):
    """Compute mean reflectance in percent over the band."""
    band = problem["band_nm"]
    wl_start, wl_end = band[0], band[1]
    n_wl = 401
    wls = np.linspace(wl_start, wl_end, n_wl)
    
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    n_dict = problem["materials"]
    
    R = _reflectance_array(materials, thicknesses, wls, n_ambient, n_substrate, n_dict)
    return np.mean(R) * 100.0

def _project_to_feasible(thicknesses, max_layer, max_total):
    """Project thicknesses to feasible region."""
    thicknesses = [max(0.0, min(max_layer, float(t))) for t in thicknesses]
    total = sum(thicknesses)
    if total > max_total:
        scale = max_total / total
        thicknesses = [t * scale for t in thicknesses]
    return thicknesses

def _nelder_mead(objective, x0, max_iter=100, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5, tol=1e-7, deadline=None):
    """Nelder-Mead simplex optimization with deadline guard."""
    n = len(x0)
    simplex = [list(x0)]
    for i in range(n):
        xi = list(x0)
        xi[i] = xi[i] + 0.1 if xi[i] == 0 else xi[i] * 1.05
        simplex.append(xi)
    
    fvals = [objective(np.array(s)) for s in simplex]
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Sort simplex
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[0] - fvals[-1]) < tol:
            break
        
        # Centroid of all but worst point
        centroid = np.mean([np.array(simplex[i]) for i in range(n)], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - np.array(simplex[n]))
        fr = objective(xr)
        
        if fvals[0] < fr < fvals[n-1]:
            simplex[n] = list(xr)
            fvals[n] = fr
            continue
        
        # Expansion
        if fr < fvals[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = objective(xe)
            if fe < fr:
                simplex[n] = list(xe)
                fvals[n] = fe
            else:
                simplex[n] = list(xr)
                fvals[n] = fr
            continue
        
        # Contraction
        if fr < fvals[n-1]:
            xc = centroid + sigma * (xr - centroid)
            fc = objective(xc)
            if fc <= fr:
                simplex[n] = list(xc)
                fvals[n] = fc
            else:
                xi = centroid + sigma * (np.array(simplex[n]) - centroid)
                fi = objective(xi)
                if fi < fvals[n-1]:
                    simplex[n] = list(xi)
                    fvals[n] = fi
                else:
                    break
        else:
            xi = centroid + sigma * (np.array(simplex[n]) - centroid)
            fi = objective(xi)
            if fi < fvals[n-1]:
                simplex[n] = list(xi)
                fvals[n] = fi
            else:
                break
        
        # Shrink
        for i in range(1, n+1):
            simplex[i] = [simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]) for j in range(n)]
            fvals[i] = objective(np.array(simplex[i]))
    
    order = np.argsort(fvals)
    return np.array(simplex[order[0]]), fvals[order[0]]

def _coordinate_descent(objective, x0, max_iter=50, step=5.0, deadline=None):
    """Coordinate descent as a polishing step."""
    x = np.array(x0, dtype=float)
    n = len(x)
    current_f = objective(x)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            # Try decreasing
            for factor in [0.5, 0.7, 0.85, 0.95]:
                x_new = x.copy()
                x_new[i] = max(0.0, x[i] * factor)
                f_new = objective(x_new)
                if f_new < current_f:
                    x = x_new
                    current_f = f_new
                    improved = True
                    break
            
            # Try increasing
            for factor in [1.05, 1.15, 1.3, 1.5]:
                x_new = x.copy()
                x_new[i] = min(400.0, x[i] * factor)
                f_new = objective(x_new)
                if f_new < current_f:
                    x = x_new
                    current_f = f_new
                    improved = True
                    break
        
        if not improved:
            break
    
    return x, current_f

def design(problem):
    """Design a broadband antireflection coating using hybrid discrete-continuous search."""
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s limit
    
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    n_dict = problem["materials"]
    max_layers = problem.get("max_layers", 20)
    max_total = problem.get("max_total_nm", 1000)
    max_layer = problem.get("max_layer_nm", 400)
    band = problem["band_nm"]
    centre_wl = 0.5 * (band[0] + band[1])
    
    best_materials = None
    best_thicknesses = None
    best_score = float('inf')
    
    def evaluate(materials, thicknesses):
        """Evaluate a design and return score."""
        thicknesses = _project_to_feasible(thicknesses, max_layer, max_total)
        return _mean_reflectance_percent(materials, thicknesses, problem), thicknesses
    
    def make_objective(materials_pattern):
        """Create objective function for a fixed materials pattern."""
        def obj(x):
            thicknesses = _project_to_feasible(x, max_layer, max_total)
            return _mean_reflectance_percent(materials_pattern, thicknesses, problem)
        return obj
    
    # Generate candidate patterns
    patterns = []
    
    # Alternating patterns
    for n in range(1, min(max_layers, 10) + 1):
        patterns.append(['L' if i % 2 == 0 else 'H' for i in range(n)])
        patterns.append(['H' if i % 2 == 0 else 'L' for i in range(n)])
    
    # Remove duplicates
    seen = set()
    unique_patterns = []
    for p in patterns:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            unique_patterns.append(p)
    
    # Sort by number of layers (try smaller first, then larger)
    unique_patterns.sort(key=lambda p: len(p))
    
    # Phase 1: Quick scan of all patterns with quarter-wave initialization
    for materials_pattern in unique_patterns:
        if time.time() > deadline - 30.0:
            break
        
        n_layers = len(materials_pattern)
        
        # Quarter-wave initialization
        x0 = []
        for mat in materials_pattern:
            n_mat = n_dict[mat]
            x0.append(centre_wl / (4.0 * n_mat))
        
        score, thick = evaluate(materials_pattern, x0)
        if score < best_score:
            best_score = score
            best_materials = list(materials_pattern)
            best_thicknesses = thick
    
    # Phase 2: Local optimization on promising patterns
    # Try patterns with 3-8 layers (sweet spot for broadband AR)
    promising_patterns = [p for p in unique_patterns if 3 <= len(p) <= 8]
    
    for materials_pattern in promising_patterns:
        if time.time() > deadline - 10.0:
            break
        
        n_layers = len(materials_pattern)
        
        # Quarter-wave initialization
        x0 = []
        for mat in materials_pattern:
            n_mat = n_dict[mat]
            x0.append(centre_wl / (4.0 * n_mat))
        
        obj = make_objective(materials_pattern)
        
        try:
            x_opt, f_opt = _nelder_mead(obj, np.array(x0), max_iter=80, deadline=deadline - 5.0)
            
            if f_opt < best_score:
                best_score = f_opt
                best_materials = list(materials_pattern)
                best_thicknesses = _project_to_feasible(x_opt, max_layer, max_total)
        except Exception:
            continue
    
    # Phase 3: Polish the best solution with coordinate descent
    if best_materials is not None and time.time() < deadline - 5.0:
        obj = make_objective(best_materials)
        try:
            x_polished, f_polished = _coordinate_descent(obj, np.array(best_thicknesses), max_iter=30, deadline=deadline - 2.0)
            if f_polished < best_score:
                best_score = f_polished
                best_thicknesses = _project_to_feasible(x_polished, max_layer, max_total)
        except Exception:
            pass
    
    # Fallback
    if best_materials is None:
        best_materials = ['L']
        best_thicknesses = [centre_wl / (4.0 * n_dict['L'])]
    
    return best_materials, best_thicknesses
