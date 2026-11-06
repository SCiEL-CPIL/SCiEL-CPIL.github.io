"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _char_matrix_method(wavelengths, thicknesses, indices, ambient_n, substrate_n):
    """Compute reflectance using characteristic matrix method."""
    n_layers = len(thicknesses)
    reflectance = np.zeros(len(wavelengths))
    
    for i, wl in enumerate(wavelengths):
        M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        
        for j in range(n_layers):
            n_j = indices[j]
            d_j = thicknesses[j]
            delta_j = 2.0 * np.pi * n_j * d_j / wl
            
            cos_d = np.cos(delta_j)
            sin_d = np.sin(delta_j)
            
            if abs(sin_d) < 1e-15:
                sin_d = 1e-15 if sin_d >= 0 else -1e-15
            
            M_j = np.array([
                [cos_d, 1j * sin_d / n_j],
                [1j * n_j * sin_d, cos_d]
            ], dtype=complex)
            
            M = M @ M_j
        
        n_s = substrate_n
        n_0 = ambient_n
        
        B = M[0, 0] + M[0, 1] * n_s
        C = M[1, 0] * n_0 + M[1, 1] * n_s
        
        if abs(C + B) < 1e-15:
            reflectance[i] = 1.0
        else:
            r = (C - B) / (C + B)
            reflectance[i] = np.abs(r) ** 2
    
    return reflectance

def _mean_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance in percent over the band."""
    wavelengths = np.linspace(problem["band_nm"][0], problem["band_nm"][1], 401)
    indices = [problem["materials"][m] for m in materials]
    refl = _char_matrix_method(wavelengths, np.array(thicknesses), np.array(indices),
                               problem["ambient_index"], problem["substrate_index"])
    return 100.0 * np.mean(refl)

def _is_valid(materials, thicknesses, problem):
    """Check if the design is valid."""
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > problem["max_layers"]:
        return False
    if any(t < 0 or t > problem["max_layer_nm"] for t in thicknesses):
        return False
    if sum(thicknesses) > problem["max_total_nm"]:
        return False
    if any(m not in problem["materials"] for m in materials):
        return False
    return True

def _nelder_mead(materials, thicknesses, problem, deadline):
    """Optimize thicknesses using Nelder-Mead with bounds."""
    n = len(thicknesses)
    if n == 0:
        return list(materials), list(thicknesses), _mean_reflectance(materials, thicknesses, problem)
    
    max_layer = problem["max_layer_nm"]
    max_total = problem["max_total_nm"]
    
    def clip_thick(x):
        x = np.clip(x, 0, max_layer)
        s = np.sum(x)
        if s > max_total:
            x = x * (max_total / s)
            x = np.clip(x, 0, max_layer)
        return x
    
    x0 = np.array(thicknesses, dtype=float)
    x0 = clip_thick(x0)
    
    # Initial simplex
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        step = max(10.0, 0.1 * x0[i])
        xi[i] += step
        simplex.append(clip_thick(xi))
    
    scores = [_mean_reflectance(materials, list(simplex[i]), problem) for i in range(n + 1)]
    best_score = min(scores)
    best_x = simplex[scores.index(best_score)].copy()
    
    max_iter = 500
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Sort
        order = np.argsort(scores)
        simplex = [simplex[i] for i in order]
        scores = [scores[i] for i in order]
        
        # Convergence check
        if max(scores) - min(scores) < 1e-8:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        alpha = 1.0
        xr = clip_thick(centroid + alpha * (centroid - simplex[-1]))
        fr = _mean_reflectance(materials, list(xr), problem)
        
        if fr < scores[0]:
            # Expansion
            gamma = 2.0
            xe = clip_thick(centroid + gamma * (xr - centroid))
            fe = _mean_reflectance(materials, list(xe), problem)
            if fe < fr:
                simplex[-1] = xe
                scores[-1] = fe
            else:
                simplex[-1] = xr
                scores[-1] = fr
        elif fr < scores[-2]:
            simplex[-1] = xr
            scores[-1] = fr
        else:
            # Contraction
            rho = 0.5
            xc = clip_thick(centroid + rho * (simplex[-1] - centroid))
            fc = _mean_reflectance(materials, list(xc), problem)
            if fc < scores[-1]:
                simplex[-1] = xc
                scores[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = clip_thick(simplex[0] + 0.5 * (simplex[i] - simplex[0]))
                    scores[i] = _mean_reflectance(materials, list(simplex[i]), problem)
        
        # Track best
        if scores[0] < best_score:
            best_score = scores[0]
            best_x = simplex[0].copy()
    
    return list(materials), list(best_x), best_score

def design(problem):
    """Design antireflection coating with multiple seeds and Nelder-Mead optimization."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    L_index = problem["materials"]["L"]
    H_index = problem["materials"]["H"]
    center_wl = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    
    # Generate diverse seed designs
    seeds = []
    
    # Seed 1: Single L layer (quarter-wave)
    seeds.append((["L"], [center_wl / (4.0 * L_index)]))
    
    # Seed 2: Two-layer L-H quarter-wave
    seeds.append((["L", "H"], [center_wl / (4.0 * L_index), center_wl / (4.0 * H_index)]))
    
    # Seed 3: Three-layer L-H-L
    seeds.append((["L", "H", "L"], [center_wl / (4.0 * L_index), center_wl / (4.0 * H_index), center_wl / (4.0 * L_index)]))
    
    # Seed 4: Classic 10-layer H|LHLHLHLHL|H quarter-wave design
    pattern = ["H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H"]
    thicknesses_10 = []
    for m in pattern:
        n_idx = problem["materials"][m]
        thicknesses_10.append(center_wl / (4.0 * n_idx))
    if sum(thicknesses_10) <= problem["max_total_nm"]:
        seeds.append((pattern, thicknesses_10))
    
    # Seed 5: 5-layer alternating starting with L
    pattern5 = ["L", "H", "L", "H", "L"]
    thicknesses_5 = []
    for m in pattern5:
        n_idx = problem["materials"][m]
        thicknesses_5.append(center_wl / (4.0 * n_idx))
    if sum(thicknesses_5) <= problem["max_total_nm"]:
        seeds.append((pattern5, thicknesses_5))
    
    # Seed 6: 7-layer alternating starting with L
    pattern7 = ["L", "H", "L", "H", "L", "H", "L"]
    thicknesses_7 = []
    for m in pattern7:
        n_idx = problem["materials"][m]
        thicknesses_7.append(center_wl / (4.0 * n_idx))
    if sum(thicknesses_7) <= problem["max_total_nm"]:
        seeds.append((pattern7, thicknesses_7))
    
    # Evaluate all seeds
    best_score = float('inf')
    best_materials = None
    best_thicknesses = None
    
    for mats, thicks in seeds:
        if not _is_valid(mats, thicks, problem):
            continue
        score = _mean_reflectance(mats, thicks, problem)
        if score < best_score:
            best_score = score
            best_materials = list(mats)
            best_thicknesses = list(thicks)
    
    if best_materials is None:
        best_materials = ["L"]
        best_thicknesses = [center_wl / (4.0 * L_index)]
        best_score = _mean_reflectance(best_materials, best_thicknesses, problem)
    
    # Optimize each seed with Nelder-Mead, keeping the best result
    for mats, thicks in seeds:
        if time.time() > deadline - 10:
            break
        if not _is_valid(mats, thicks, problem):
            continue
        
        opt_mats, opt_thicks, opt_score = _nelder_mead(mats, thicks, problem, deadline - 10)
        
        if opt_score < best_score:
            best_score = opt_score
            best_materials = opt_mats
            best_thicknesses = opt_thicks
    
    # Final polish with Nelder-Mead on the best solution
    if time.time() < deadline - 5 and _is_valid(best_materials, best_thicknesses, problem):
        opt_mats, opt_thicks, opt_score = _nelder_mead(best_materials, best_thicknesses, problem, deadline - 5)
        if opt_score < best_score:
            best_score = opt_score
            best_materials = opt_mats
            best_thicknesses = opt_thicks
    
    # Ensure validity
    if not _is_valid(best_materials, best_thicknesses, problem):
        best_materials = ["L"]
        best_thicknesses = [center_wl / (4.0 * L_index)]
    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
