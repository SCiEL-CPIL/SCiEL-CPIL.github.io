import numpy as np
import time
from scipy.optimize import minimize

def design(problem):
    n_0 = problem["ambient_index"]
    n_s = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    wavelengths = np.linspace(band[0], band[1], 401)
    center_lam = 0.5 * (band[0] + band[1])
    
    def compute_reflectance_from_params(params):
        """
        Compute mean reflectance from continuous parameters.
        params: [n_1, d_1, n_2, d_2, ..., n_N, d_N]
        """
        N = len(params) // 2
        if N == 0:
            r = (n_0 - n_s) / (n_0 + n_s)
            return (abs(r)**2) * 100.0
        
        total_ref = 0.0
        for lam in wavelengths:
            M = np.array([[1.0+0j, 0.0+0j], [0.0+0j, 1.0+0j]])
            for i in range(N):
                n_layer = params[2*i]
                d = params[2*i+1]
                delta = 2.0 * np.pi * n_layer * d / lam
                c = np.cos(delta)
                s = np.sin(delta)
                M_i = np.array([[c, 1j*s/n_layer], [1j*n_layer*s, c]])
                M = M @ M_i
            Y = (M[0,0] + M[0,1]*n_s) / (M[1,0] + M[1,1]*n_s)
            r = (Y - n_0) / (Y + n_0)
            R = (abs(r)**2) * 100.0
            total_ref += R
        return total_ref / len(wavelengths)
    
    def objective(params):
        """Objective function with penalty for constraint violations."""
        N = len(params) // 2
        # Check thickness constraints
        for i in range(N):
            d = params[2*i+1]
            if d < 0 or d > max_layer_nm:
                return 1e6
        total_d = sum(params[2*i+1] for i in range(N))
        if total_d > max_total_nm:
            return 1e6
        # Check refractive index range
        for i in range(N):
            n = params[2*i]
            if n < n_L - 0.01 or n > n_H + 0.01:
                return 1e6
        return compute_reflectance_from_params(params)
    
    # Start with a reasonable number of layers
    N_start = 8
    # Initial guess: alternating L and H with quarter-wave thicknesses
    t_qw = center_lam / (4.0 * 1.8)  # approximate
    params0 = []
    for i in range(N_start):
        n_i = n_L if i % 2 == 0 else n_H
        d_i = t_qw
        params0.extend([n_i, d_i])
    params0 = np.array(params0)
    
    # Bounds for L-BFGS-B
    bounds = []
    for i in range(N_start):
        bounds.append((n_L, n_H))  # refractive index
        bounds.append((0, max_layer_nm))  # thickness
    
    start_time = time.time()
    time_limit = 50.0
    
    # Optimize with L-BFGS-B
    result = minimize(objective, params0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 100, 'ftol': 1e-10, 'maxls': 50})
    
    if time.time() - start_time > time_limit:
        # Fallback to simple design
        t_L_1 = center_lam / (4.0 * n_L)
        return ['L'], [t_L_1]
    
    opt_params = result.x
    N_opt = len(opt_params) // 2
    
    # Quantize: map each continuous n to nearest discrete material
    materials = []
    thicknesses = []
    for i in range(N_opt):
        n_i = opt_params[2*i]
        d_i = opt_params[2*i+1]
        # Assign to L or H based on which is closer
        if abs(n_i - n_L) <= abs(n_i - n_H):
            materials.append('L')
        else:
            materials.append('H')
        thicknesses.append(d_i)
    
    # Remove zero-thickness layers
    final_materials = []
    final_thicknesses = []
    for m, t in zip(materials, thicknesses):
        if t > 1e-6:
            final_materials.append(m)
            final_thicknesses.append(t)
    
    # Ensure constraints are satisfied
    if not final_materials:
        t_L_1 = center_lam / (4.0 * n_L)
        return ['L'], [t_L_1]
    
    # Clip thicknesses
    for i in range(len(final_thicknesses)):
        final_thicknesses[i] = max(0, min(max_layer_nm, final_thicknesses[i]))
    
    # If total thickness exceeds limit, scale down
    total_t = sum(final_thicknesses)
    if total_t > max_total_nm:
        scale = max_total_nm / total_t
        final_thicknesses = [t * scale for t in final_thicknesses]
    
    # If too many layers, keep only the most significant ones
    if len(final_materials) > max_layers:
        # Sort by thickness and keep top max_layers
        indices = np.argsort(final_thicknesses)[::-1][:max_layers]
        # Re-sort by original position
        indices = sorted(indices)
        final_materials = [final_materials[i] for i in indices]
        final_thicknesses = [final_thicknesses[i] for i in indices]
    
    # Local refinement with discrete search
    best_materials = final_materials[:]
    best_thicknesses = final_thicknesses[:]
    best_score = compute_reflectance_from_params(
        np.array([n_L if m == 'L' else n_H for m in best_materials] + 
                 [0]*len(best_materials))  # placeholder, need proper function
    )
    
    # Actually, let's use the original compute_reflectance for discrete stacks
    def compute_reflectance_discrete(materials, thicknesses):
        if not materials:
            r = (n_0 - n_s) / (n_0 + n_s)
            return (abs(r)**2) * 100.0
        total_ref = 0.0
        N = len(materials)
        for lam in wavelengths:
            M = np.array([[1.0+0j, 0.0+0j], [0.0+0j, 1.0+0j]])
            for i in range(N):
                n_layer = n_L if materials[i] == 'L' else n_H
                d = thicknesses[i]
                delta = 2.0 * np.pi * n_layer * d / lam
                c = np.cos(delta)
                s = np.sin(delta)
                M_i = np.array([[c, 1j*s/n_layer], [1j*n_layer*s, c]])
                M = M @ M_i
            Y = (M[0,0] + M[0,1]*n_s) / (M[1,0] + M[1,1]*n_s)
            r = (Y - n_0) / (Y + n_0)
            R = (abs(r)**2) * 100.0
            total_ref += R
        return total_ref / len(wavelengths)
    
    best_score = compute_reflectance_discrete(best_materials, best_thicknesses)
    
    # Local search refinement
    start_time2 = time.time()
    time_limit2 = max(0, time_limit - (time.time() - start_time))
    import random
    random.seed(42)
    
    while time.time() - start_time2 < time_limit2 and time.time() - start_time < time_limit:
        move_type = random.choice(['thick', 'mat', 'add', 'rem'])
        new_materials = best_materials[:]
        new_thicknesses = best_thicknesses[:]
        valid_move = False
        
        if move_type == 'thick' and len(new_materials) > 0:
            idx = random.randint(0, len(new_materials)-1)
            delta_t = random.uniform(-10, 10)
            new_thicknesses[idx] = max(0, min(max_layer_nm, new_thicknesses[idx] + delta_t))
            valid_move = True
        elif move_type == 'mat' and len(new_materials) > 0:
            idx = random.randint(0, len(new_materials)-1)
            new_materials[idx] = 'H' if new_materials[idx] == 'L' else 'L'
            valid_move = True
        elif move_type == 'add' and len(new_materials) < max_layers:
            pos = random.randint(0, len(new_materials))
            new_materials.insert(pos, random.choice(['L', 'H']))
            new_thicknesses.insert(pos, random.uniform(20, 80))
            valid_move = True
        elif move_type == 'rem' and len(new_materials) > 1:
            idx = random.randint(0, len(new_materials)-1)
            new_materials.pop(idx)
            new_thicknesses.pop(idx)
            valid_move = True
        
        if not valid_move:
            continue
        
        if sum(new_thicknesses) <= max_total_nm and all(0 <= t <= max_layer_nm for t in new_thicknesses):
            new_score = compute_reflectance_discrete(new_materials, new_thicknesses)
            if new_score < best_score:
                best_score = new_score
                best_materials = new_materials
                best_thicknesses = new_thicknesses
    
    return best_materials, best_thicknesses
