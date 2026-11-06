"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _mean_reflectance_vec(materials, thicknesses, problem):
    """Compute mean reflectance in percent over the band using vectorized numpy."""
    if len(materials) == 0:
        n0 = problem["ambient_index"]
        ns = problem["substrate_index"]
        r = (n0 - ns) / (n0 + ns)
        return 100.0 * r * r
    
    band = problem["band_nm"]
    wavelengths = np.linspace(band[0], band[1], 401)
    mat_dict = problem["materials"]
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    
    # Initialize total matrix for all wavelengths
    # M is shape (401, 2, 2)
    M = np.zeros((len(wavelengths), 2, 2), dtype=complex)
    M[:, 0, 0] = 1.0
    M[:, 1, 1] = 1.0
    
    for mat, thick in zip(materials, thicknesses):
        n = mat_dict[mat]
        d = thick
        if d == 0:
            continue
        beta = 2.0 * np.pi * n * d / wavelengths
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        # Layer matrix
        m00 = cos_b
        m01 = 1.0j * sin_b / n
        m10 = n * 1.0j * sin_b
        m11 = cos_b
        
        # Multiply M @ m
        M_new = np.empty_like(M)
        M_new[:, 0, 0] = M[:, 0, 0] * m00 + M[:, 0, 1] * m10
        M_new[:, 0, 1] = M[:, 0, 0] * m01 + M[:, 0, 1] * m11
        M_new[:, 1, 0] = M[:, 1, 0] * m00 + M[:, 1, 1] * m10
        M_new[:, 1, 1] = M[:, 1, 0] * m01 + M[:, 1, 1] * m11
        M = M_new
        
    # Substrate admittance
    Ys = ns
    
    # Total admittance Y = M[1, 1] / M[0, 1]
    Y = M[:, 1, 1] / M[:, 0, 1]
    
    # Reflectance
    r = (n0 - Y) / (n0 + Y)
    R = np.abs(r) ** 2 * 100.0
    
    return np.mean(R)

def _is_valid(materials, thicknesses, problem):
    """Check if the design is valid."""
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    if len(materials) > max_layers:
        return False
    if len(materials) != len(thicknesses):
        return False
    if sum(thicknesses) > max_total_nm:
        return False
    for t in thicknesses:
        if t < 0 or t > max_layer_nm:
            return False
    return True

def design(problem):
    """Design an antireflection coating stack."""
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s limit
    
    centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    
    dL = centre / (4.0 * nL)
    dH = centre / (4.0 * nH)
    
    # Generate candidate structures with different layer counts
    candidate_structures = []
    
    # 1-layer L
    candidate_structures.append((["L"], [dL]))
    
    # 3-layer L-H-L
    candidate_structures.append((["L", "H", "L"], [dL, dH, dL]))
    
    # 5-layer L-H-L-H-L
    candidate_structures.append((["L", "H", "L", "H", "L"], [dL, dH, dL, dH, dL]))
    
    # 7-layer L-H-L-H-L-H-L
    candidate_structures.append((["L", "H", "L", "H", "L", "H", "L"], 
                                 [dL, dH, dL, dH, dL, dH, dL]))
    
    # 9-layer L-H-L-H-L-H-L-H-L
    candidate_structures.append((["L", "H", "L", "H", "L", "H", "L", "H", "L"], 
                                 [dL, dH, dL, dH, dL, dH, dL, dH, dL]))
    
    # 11-layer L-H-L-H-L-H-L-H-L-H-L
    candidate_structures.append((["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"], 
                                 [dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL]))
    
    # 13-layer L-H-L-H-L-H-L-H-L-H-L-H-L
    candidate_structures.append((["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"], 
                                 [dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL]))
    
    # 15-layer L-H-L-H-L-H-L-H-L-H-L-H-L-H-L
    candidate_structures.append((["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"], 
                                 [dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL]))
    
    # 17-layer L-H-L-H-L-H-L-H-L-H-L-H-L-H-L-H-L
    candidate_structures.append((["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"], 
                                 [dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL]))
    
    # 19-layer L-H-L-H-L-H-L-H-L-H-L-H-L-H-L-H-L-H-L
    candidate_structures.append((["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"], 
                                 [dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL]))
    
    best_score = float('inf')
    best_result = candidate_structures[0]
    
    # Evaluate initial candidates
    for mats, thicks in candidate_structures:
        if not _is_valid(mats, thicks, problem):
            continue
        score = _mean_reflectance_vec(mats, thicks, problem)
        if score < best_score:
            best_score = score
            best_result = (mats, thicks)
        if time.time() > deadline - 10.0:
            break
    
    # Local optimization using L-BFGS-B with constraints
    def optimize_lbfgsb(materials, thicknesses, problem, time_limit):
        """Optimize thicknesses using L-BFGS-B with constraints."""
        current_thicks = np.array(thicknesses, dtype=float)
        n_params = len(current_thicks)
        
        # Objective function
        def objective(x):
            x = np.clip(x, 0, problem["max_layer_nm"])
            # Check total thickness constraint
            if np.sum(x) > problem["max_total_nm"]:
                return 1000.0  # Large penalty
            return _mean_reflectance_vec(materials, x, problem)
        
        # Constraints: total thickness <= max_total_nm
        constraints = [
            {
                'type': 'ineq',
                'fun': lambda x: problem["max_total_nm"] - np.sum(x)
            }
        ]
        
        # Bounds: each thickness in [0, max_layer_nm]
        bounds = [(0, problem["max_layer_nm"]) for _ in range(n_params)]
        
        # Run L-BFGS-B
        try:
            result = minimize(
                objective,
                current_thicks,
                method='L-BFGS-B',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success:
                opt_thicks = np.clip(result.x, 0, problem["max_layer_nm"])
                # Ensure total thickness constraint is satisfied
                if np.sum(opt_thicks) > problem["max_total_nm"]:
                    # Scale down proportionally
                    scale = problem["max_total_nm"] / np.sum(opt_thicks)
                    opt_thicks *= scale
                    opt_thicks = np.clip(opt_thicks, 0, problem["max_layer_nm"])
                
                opt_score = _mean_reflectance_vec(materials, opt_thicks, problem)
                return opt_thicks.tolist(), opt_score
        except Exception:
            pass
        
        return current_thicks.tolist(), _mean_reflectance_vec(materials, current_thicks, problem)
    
    # Optimize the best candidate found so far
    if time.time() < deadline - 20.0:
        mats, thicks = best_result
        opt_thicks, opt_score = optimize_lbfgsb(mats, thicks, problem, deadline - 20.0)
        if opt_score < best_score:
            best_score = opt_score
            best_result = (mats, opt_thicks)
    
    # If time remains, try optimizing other top candidates
    if time.time() < deadline - 30.0:
        # Try the 5-layer and 7-layer structures
        for alt_mats, alt_thicks in [candidate_structures[2], candidate_structures[3]]:
            if time.time() > deadline - 15.0:
                break
            if _is_valid(alt_mats, alt_thicks, problem):
                opt_thicks, opt_score = optimize_lbfgsb(alt_mats, alt_thicks, problem, deadline - 15.0)
                if opt_score < best_score:
                    best_score = opt_score
                    best_result = (alt_mats, opt_thicks)
    
    # Final validation
    mats, thicks = best_result
    if not _is_valid(mats, thicks, problem):
        # Fallback to the simplest valid design
        return (["L"], [dL])
    
    return best_result
# EVOLVE-BLOCK-END
