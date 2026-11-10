"""
Broadband AR coating design using Analytical Quarter-Wave Initialization 
followed by Fast Vectorized Nelder-Mead Optimization.
"""
import numpy as np
import time
import sys
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _reflectance_calc(mats, thicks, n0, ns, materials, lambdas):
    """
    Fast vectorized reflectance calculation.
    mats: list of 'L'/'H'
    thicks: numpy array of thicknesses
    """
    N = len(thicks)
    if N == 0:
        r = (n0 - ns) / (n0 + ns)
        return np.full_like(lambdas, np.abs(r)**2)
        
    # Precompute sin/cos for all layers and wavelengths
    # We process layer by layer but vectorize over wavelengths.
    # M_total = M1 * M2 * ... * MN
    # Start with identity
    M11 = np.ones(len(lambdas), dtype=complex)
    M12 = np.zeros(len(lambdas), dtype=complex)
    M21 = np.zeros(len(lambdas), dtype=complex)
    M22 = np.ones(len(lambdas), dtype=complex)
    
    for i in range(N):
        n_i = materials[mats[i]]
        d_i = thicks[i]
        
        delta = 2.0 * np.pi * n_i * d_i / lambdas
        c = np.cos(delta)
        s = np.sin(delta)
        
        m11 = c
        m12 = 1j * s / n_i
        m21 = 1j * n_i * s
        m22 = c
        
        # Matrix multiplication
        new_M11 = M11 * m11 + M12 * m21
        new_M12 = M11 * m12 + M12 * m22
        new_M21 = M21 * m11 + M22 * m21
        new_M22 = M21 * m12 + M22 * m22
        
        M11 = new_M11
        M12 = new_M12
        M21 = new_M21
        M22 = new_M22
        
    Y_s = ns
    Y_in = (M21 + M22 * Y_s) / (M11 + M12 * Y_s)
    r = (Y_in - n0) / (Y_in + n0)
    R = np.abs(r)**2
    return R

def design(problem):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s wall clock limit
    
    band = problem["band_nm"]
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    materials = problem["materials"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    lambdas = np.linspace(band[0], band[1], 401)
    center_lambda = 0.5 * (band[0] + band[1])
    
    # Helper to evaluate a specific state
    def evaluate(mats, thicks):
        if len(mats) == 0:
            return np.mean(_reflectance_calc(mats, np.array([]), n0, ns, materials, lambdas)) * 100.0
        return np.mean(_reflectance_calc(mats, thicks, n0, ns, materials, lambdas)) * 100.0

    # Helper to check constraints
    def is_valid(mats, thicks):
        if len(mats) != len(thicks):
            return False
        if len(mats) > max_layers:
            return False
        if np.sum(thicks) > max_total_nm:
            return False
        if np.any(thicks < 0) or np.any(thicks > max_layer_nm):
            return False
        return True

    best_score = np.inf
    best_mats = []
    best_thicks = np.array([])
    
    # Strategy: Try multiple structural configurations (depths and patterns)
    # For each, initialize with quarter-wave thicknesses, then optimize with Nelder-Mead.
    
    # Possible depths
    depths = [5, 7, 9, 11, 13, 15]
    
    # Pre-compute quarter-wave thicknesses for center wavelength
    qw_L = center_lambda / (4.0 * materials['L'])
    qw_H = center_lambda / (4.0 * materials['H'])
    
    for depth in depths:
        if time.time() > deadline - 10.0:
            break
            
        # Define a few pattern templates for this depth
        # Pattern 1: Alternating L/H starting with L
        # Pattern 2: Alternating H/L starting with H
        # Pattern 3: L H L H ... (standard)
        # Pattern 4: H L H L ...
        
        patterns = []
        
        # Standard alternating
        p1 = []
        for i in range(depth):
            p1.append('L' if i % 2 == 0 else 'H')
        patterns.append(p1)
        
        # Inverted alternating
        p2 = []
        for i in range(depth):
            p2.append('H' if i % 2 == 0 else 'L')
        patterns.append(p2)
        
        # If depth is odd, ensure it ends with H (high index next to glass is usually better for impedance matching if n_glass > n_H? No, n_glass=1.52, n_H=2.35. 
        # Actually, standard AR coatings often end with high index if substrate is high, or low index if substrate is low. 
        # Glass (1.52) is between L (1.38) and H (2.35). 
        # Let's just try both.)
        
        for mats in patterns:
            if time.time() > deadline - 5.0:
                break
                
            # Initialize thicknesses with quarter-wave values
            thicks = np.zeros(depth)
            for i in range(depth):
                if mats[i] == 'L':
                    thicks[i] = qw_L
                else:
                    thicks[i] = qw_H
            
            # Check if total thickness exceeds limit
            total_t = np.sum(thicks)
            if total_t > max_total_nm:
                # Scale down
                thicks *= (max_total_nm / total_t)
            
            if not is_valid(mats, thicks):
                continue
                
            current_score = evaluate(mats, thicks)
            
            if current_score < best_score:
                best_score = current_score
                best_mats = mats.copy()
                best_thicks = thicks.copy()
                
            # Optimize using Nelder-Mead
            # We optimize the thicknesses directly.
            # Objective function for scipy
            def objective(thick_array):
                # Ensure validity
                if np.any(thick_array < 0) or np.any(thick_array > max_layer_nm):
                    return 100.0
                if np.sum(thick_array) > max_total_nm:
                    return 100.0
                return evaluate(mats, thick_array)
            
            # Initial simplex
            # Perturb the initial point slightly to avoid flat spots
            x0 = thicks.copy()
            
            # Set options for Nelder-Mead
            options = {
                'maxiter': 200,
                'maxfev': 200,
                'xatol': 1e-4,
                'fatol': 1e-6
            }
            
            try:
                result = minimize(objective, x0, method='Nelder-Mead', options=options)
                
                if result.success and is_valid(mats, result.x):
                    opt_score = result.fun
                    if opt_score < best_score:
                        best_score = opt_score
                        best_mats = mats.copy()
                        best_thicks = result.x.copy()
                        
            except Exception:
                pass
                
    # Final Polish: If we have time, do a second pass of Nelder-Mead on the best solution found
    if time.time() < deadline - 10.0 and len(best_mats) > 0:
        def objective_final(thick_array):
            if np.any(thick_array < 0) or np.any(thick_array > max_layer_nm):
                return 100.0
            if np.sum(thick_array) > max_total_nm:
                return 100.0
            return evaluate(best_mats, thick_array)
        
        x0 = best_thicks.copy()
        
        options = {
            'maxiter': 500,
            'maxfev': 500,
            'xatol': 1e-5,
            'fatol': 1e-7
        }
        
        try:
            result = minimize(objective_final, x0, method='Nelder-Mead', options=options)
            if result.success and is_valid(best_mats, result.x):
                opt_score = result.fun
                if opt_score < best_score:
                    best_score = opt_score
                    best_thicks = result.x.copy()
        except Exception:
            pass

    # Ensure types are correct: list of strings, list of floats
    return best_mats, list(best_thicks)

# EVOLVE-BLOCK-END
