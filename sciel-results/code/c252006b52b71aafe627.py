"""Broadband antireflection coating design via SA + Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _vectorized_reflectance(thicknesses, materials, n_ambient, n_substrate, n_L, n_H, wls):
    """
    Compute mean reflectance fraction for a single design across all wavelengths.
    Uses vectorized operations over the wavelength axis.
    """
    n_layers = len(thicknesses)
    if n_layers == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return float(r * r)
    
    n_layers_arr = np.array([n_H if m == 'H' else n_L for m in materials])
    
    M11 = np.ones_like(wls, dtype=complex)
    M12 = np.zeros_like(wls, dtype=complex)
    M21 = np.zeros_like(wls, dtype=complex)
    M22 = np.ones_like(wls, dtype=complex)
    
    for i in range(n_layers):
        n = n_layers_arr[i]
        d = thicknesses[i]
        beta = 2.0 * np.pi * n * d / wls
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        L11 = cos_b
        L12 = -1j * sin_b / n
        L21 = 1j * n * sin_b
        L22 = cos_b
        
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
    
    Y = (M11 * n_substrate + M12) / (M21 * n_substrate + M22)
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.real(r * r.conjugate())
    
    return float(np.mean(R))

def _evaluate_design(thicknesses, materials, n_ambient, n_substrate, n_L, n_H, wls, max_total_nm, max_layer_nm):
    """
    Evaluate a design. Returns reflectance percentage if valid, inf otherwise.
    """
    if len(thicknesses) == 0:
        return float('inf')
    
    if any(t < -1e-6 or t > max_layer_nm + 1e-6 for t in thicknesses):
        return float('inf')
    if sum(thicknesses) > max_total_nm + 1e-6:
        return float('inf')
        
    t_calc = np.clip(thicknesses, 0, max_layer_nm)
    
    R_frac = _vectorized_reflectance(t_calc, materials, n_ambient, n_substrate, n_L, n_H, wls)
    return R_frac * 100.0

def _nelder_mead(objective, x0, bounds, deadline, max_iter=500, tol=1e-8):
    """
    Nelder-Mead optimization with bounds handling and adaptive step sizing.
    """
    x0 = np.array(x0, dtype=float)
    n = len(x0)
    lo, hi = bounds
    
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        range_i = hi[i] - lo[i]
        step = 0.1 * range_i if range_i > 1e-6 else 1.0
        xi[i] += step
        simplex.append(np.clip(xi, lo, hi))
        
    f_vals = [objective(x) for x in simplex]
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    best_f = min(f_vals)
    stagnant_count = 0
    
    for iter_count in range(max_iter):
        if time.time() > deadline:
            break
            
        order = np.argsort(f_vals)
        simplex = [simplex[i] for i in order]
        f_vals = [f_vals[i] for i in order]
        
        spread = abs(f_vals[0] - f_vals[-1])
        if spread < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        x_r = np.clip(centroid + alpha * (centroid - simplex[-1]), lo, hi)
        f_r = objective(x_r)
        
        if f_vals[0] <= f_r < f_vals[-2]:
            simplex[-1] = x_r
            f_vals[-1] = f_r
            stagnant_count = 0
            continue
            
        if f_r < f_vals[0]:
            x_e = np.clip(centroid + gamma * (x_r - centroid), lo, hi)
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_vals[-1] = f_e
            else:
                simplex[-1] = x_r
                f_vals[-1] = f_r
            stagnant_count = 0
            continue
            
        x_c = np.clip(centroid + rho * (simplex[-1] - centroid), lo, hi)
        f_c = objective(x_c)
        if f_c < f_vals[-1]:
            simplex[-1] = x_c
            f_vals[-1] = f_c
            stagnant_count = 0
            continue
            
        for i in range(1, n + 1):
            simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), lo, hi)
            f_vals[i] = objective(simplex[i])
            
        if min(f_vals) > best_f - 1e-10:
            stagnant_count += 1
        else:
            stagnant_count = 0
            best_f = min(f_vals)
            
        if stagnant_count > 10:
            x_best = simplex[0].copy()
            simplex = [x_best.copy()]
            for i in range(n):
                xi = x_best.copy()
                range_i = hi[i] - lo[i]
                step = 0.3 * range_i if range_i > 1e-6 else 3.0
                xi[i] += step
                simplex.append(np.clip(xi, lo, hi))
            f_vals = [objective(x) for x in simplex]
            stagnant_count = 0
            
    order = np.argsort(f_vals)
    best_idx = order[0]
    return np.clip(simplex[best_idx], lo, hi), f_vals[best_idx]

def design(problem):
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    wl_min, wl_max = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    start_time = time.time()
    deadline = start_time + 145.0
    
    wls = np.linspace(wl_min, wl_max, 401)
    wl_center = 0.5 * (wl_min + wl_max)
    
    qw_L = wl_center / (4.0 * n_L)
    qw_H = wl_center / (4.0 * n_H)
    
    best_score = float('inf')
    best_thicknesses = [qw_L]
    best_materials = ['L']
    
    score = _evaluate_design(best_thicknesses, best_materials, n_ambient, n_substrate, n_L, n_H, wls, max_total_nm, max_layer_nm)
    if score < best_score:
        best_score = score

    # Phase 1: Simulated Annealing to find good structure and rough thicknesses
    init_n_layers = 5
    current_materials = []
    current_thicknesses = []
    for i in range(init_n_layers):
        if i % 2 == 0:
            current_materials.append('H')
            current_thicknesses.append(qw_H)
        else:
            current_materials.append('L')
            current_thicknesses.append(qw_L)
    current_thicknesses = np.array(current_thicknesses)
    
    current_score = _evaluate_design(current_thicknesses, current_materials, n_ambient, n_substrate, n_L, n_H, wls, max_total_nm, max_layer_nm)
    if current_score < best_score:
        best_score = current_score
        best_thicknesses = current_thicknesses.tolist()
        best_materials = current_materials[:]

    T = 100.0
    T_min = 0.01
    alpha = 0.999
    max_iterations = 30000
    
    sa_deadline = start_time + 100.0  # Reserve 45s for NM polish
    
    for step in range(max_iterations):
        if time.time() > sa_deadline:
            break
            
        new_materials = current_materials[:]
        new_thicknesses = current_thicknesses.copy()
        
        move_type = np.random.rand()
        
        if move_type < 0.7:
            n_perturb = np.random.randint(1, 4)
            if len(new_thicknesses) > 0:
                indices = np.random.choice(len(new_thicknesses), n_perturb, replace=False)
                for idx in indices:
                    delta = np.random.normal(0, 20)
                    new_thicknesses[idx] += delta
                    new_thicknesses[idx] = np.clip(new_thicknesses[idx], 0, max_layer_nm)
                    
        elif move_type < 0.9:
            if len(new_materials) > 0:
                idx = np.random.randint(len(new_materials))
                if new_materials[idx] == 'H':
                    new_materials[idx] = 'L'
                else:
                    new_materials[idx] = 'H'
        else:
            if len(new_materials) < max_layers and np.random.rand() < 0.5:
                pos = np.random.randint(0, len(new_materials) + 1)
                mat = 'H' if np.random.rand() < 0.5 else 'L'
                thick = np.random.uniform(50, 200)
                new_materials.insert(pos, mat)
                new_thicknesses = np.insert(new_thicknesses, pos, thick)
            elif len(new_materials) > 1 and np.random.rand() < 0.5:
                pos = np.random.randint(0, len(new_materials))
                new_materials.pop(pos)
                new_thicknesses = np.delete(new_thicknesses, pos)
            
        if len(new_thicknesses) == 0:
            continue
            
        if any(t < -1e-6 or t > max_layer_nm + 1e-6 for t in new_thicknesses):
            continue
        if sum(new_thicknesses) > max_total_nm + 1e-6:
            continue
            
        new_score = _evaluate_design(new_thicknesses, new_materials, n_ambient, n_substrate, n_L, n_H, wls, max_total_nm, max_layer_nm)
        
        if np.isnan(new_score) or np.isinf(new_score):
            continue
            
        if new_score < current_score:
            current_score = new_score
            current_materials = new_materials
            current_thicknesses = new_thicknesses
            
            if current_score < best_score:
                best_score = current_score
                best_thicknesses = current_thicknesses.tolist()
                best_materials = current_materials[:]
        else:
            delta = new_score - current_score
            if delta > 50 * T:
                prob = 0
            else:
                prob = np.exp(-delta / T)
            if np.random.rand() < prob:
                current_score = new_score
                current_materials = new_materials
                current_thicknesses = new_thicknesses
                
        T *= alpha
        if T < T_min:
            T = T_min

    # Phase 2: Nelder-Mead polish on the best structure found by SA
    if best_materials and len(best_materials) > 0:
        # Try to polish the best solution found
        nm_deadline = deadline - 5.0  # 5s safety margin
        
        # Build objective function for NM
        def nm_objective(x):
            x = np.clip(x, 0, max_layer_nm)
            # Check total thickness constraint
            if np.sum(x) > max_total_nm + 1e-6:
                return float('inf')
            return _evaluate_design(x, best_materials, n_ambient, n_substrate, n_L, n_H, wls, max_total_nm, max_layer_nm)
        
        x0 = np.array(best_thicknesses, dtype=float)
        bounds = (np.zeros_like(x0), np.full_like(x0, max_layer_nm))
        
        try:
            x_opt, f_opt = _nelder_mead(nm_objective, x0, bounds, nm_deadline, max_iter=300)
            
            if f_opt < best_score and not np.isinf(f_opt):
                # Validate the NM solution
                x_opt = np.clip(x_opt, 0, max_layer_nm)
                if np.sum(x_opt) <= max_total_nm + 1e-6:
                    best_score = f_opt
                    best_thicknesses = x_opt.tolist()
        except Exception:
            pass
    
    # Final validity check and cleanup
    if not best_thicknesses or any(t < 0 or t > max_layer_nm for t in best_thicknesses) or sum(best_thicknesses) > max_total_nm:
        return ["L"], [qw_L]
        
    cleaned_thick = [max(0.0, t) for t in best_thicknesses]
    if sum(cleaned_thick) > max_total_nm:
        scale = max_total_nm / sum(cleaned_thick)
        cleaned_thick = [t * scale for t in cleaned_thick]
        
    return best_materials, cleaned_thick

# EVOLVE-BLOCK-END
