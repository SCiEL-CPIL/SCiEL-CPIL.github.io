# EVOLVE-BLOCK-START
import numpy as np
import random
from scipy.optimize import minimize

def _compute_reflectance_vec(thicknesses, materials, problem):
    """
    Compute mean reflectance in percent over the band for a fixed material sequence.
    thicknesses: np.array of floats
    """
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    n_dict = problem["materials"]
    lam0, lam1 = problem["band_nm"]
    n_waves = 401
    lambdas = np.linspace(lam0, lam1, n_waves)
    
    n_layers = len(materials)
    if n_layers == 0:
        r = (n0 - ns) / (n0 + ns)
        return 100.0 * r * r

    n_vals = np.array([n_dict[m] for m in materials])
    d_vals = np.array(thicknesses)
    
    # Vectorize over wavelengths
    deltas = 2.0 * np.pi * np.outer(lambdas, n_vals * d_vals) # (401, n_layers)
    
    # Initialize matrices for all wavelengths
    A = np.ones(n_waves, dtype=complex)
    B = np.zeros(n_waves, dtype=complex)
    C = np.zeros(n_waves, dtype=complex)
    D = np.ones(n_waves, dtype=complex)
    
    for i in range(n_layers):
        n_i = n_vals[i]
        delta_i = deltas[:, i]
        cos_d = np.cos(delta_i)
        sin_d = np.sin(delta_i)
        
        M11 = cos_d
        M12 = 1j * sin_d / n_i
        M21 = 1j * n_i * sin_d
        M22 = cos_d
        
        nA = A * M11 + B * M21
        nB = A * M12 + B * M22
        nC = C * M11 + D * M21
        nD = C * M12 + D * M22
        
        A, B, C, D = nA, nB, nC, nD
        
    Y_in = (C + D * ns) / (A + B * ns)
    r = (n0 - Y_in) / (n0 + Y_in)
    R = np.abs(r) ** 2
    total_r = np.mean(R)
    
    return 100.0 * total_r

def design(problem):
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Generate candidate structures
    candidates = []
    
    # Structured patterns: alternating L/H
    for length in range(1, 15):
        # Start with L
        mats_L = ['L' if i % 2 == 0 else 'H' for i in range(length)]
        # Start with H
        mats_H = ['H' if i % 2 == 0 else 'L' for i in range(length)]
        candidates.append(mats_L)
        candidates.append(mats_H)
        
    # Random patterns
    random.seed(42)
    for _ in range(50):
        length = random.randint(1, 12)
        mats = [random.choice(['L', 'H']) for _ in range(length)]
        candidates.append(mats)
    
    best_result = None
    best_score = float('inf')
    
    # Shuffle and take top 30 to ensure diversity and time efficiency
    random.shuffle(candidates)
    candidates = candidates[:30]
    
    # Initial guess: Quarter wave at lambda = 600nm
    lambda_center = 600.0
    
    for mats in candidates:
        n_layers = len(mats)
        if n_layers == 0:
            continue
            
        n_vals = np.array([problem["materials"][m] for m in mats])
        
        # Initial thicknesses: quarter wave
        t0 = lambda_center / (4.0 * n_vals)
        # Ensure sum is within max_total_nm
        sum_t0 = np.sum(t0)
        if sum_t0 > max_total_nm:
            t0 = t0 * (max_total_nm / sum_t0)
        
        # Define objective
        def objective(thick):
            # Penalty if out of bounds (should not happen with bounds)
            if np.any(thick < 0) or np.any(thick > max_layer_nm):
                return 100.0
            if np.sum(thick) > max_total_nm:
                return 100.0 + (np.sum(thick) - max_total_nm)
            
            return _compute_reflectance_vec(thick, mats, problem)
        
        # Bounds
        bounds = [(0, max_layer_nm) for _ in range(n_layers)]
        
        # Constraints: sum(thick) <= max_total_nm
        # SLSQP uses constraints as dict or list of dicts
        # type='ineq' means fun(x) >= 0
        # So fun(x) = max_total_nm - sum(x) >= 0
        constraints = ({'type': 'ineq', 'fun': lambda x: max_total_nm - np.sum(x)})
        
        try:
            res = minimize(objective, t0, method='SLSQP', bounds=bounds, 
                           constraints=constraints, options={'maxiter': 200, 'ftol': 1e-8})
            
            if res.success or res.fun < best_score:
                score = res.fun
                if score < best_score:
                    best_score = score
                    best_result = (mats, list(res.x))
        except Exception:
            continue
            
    # If no result found, return a simple fallback
    if best_result is None:
        # Simple 1-layer AR
        mats = ['L']
        n_L = problem["materials"]["L"]
        t = lambda_center / (4.0 * n_L)
        if t > max_layer_nm: t = max_layer_nm
        best_result = (mats, [t])
        
    return best_result
# EVOLVE-BLOCK-END
