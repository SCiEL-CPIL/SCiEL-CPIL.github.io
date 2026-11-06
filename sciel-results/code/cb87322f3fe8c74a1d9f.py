import numpy as np
import time
from scipy.optimize import minimize

def design(problem):
    start_time = time.time()
    timeout = 110.0 # 10s margin
    
    # Problem parameters
    n_amb = problem.get('ambient_index', 1.0)
    n_sub = problem.get('substrate_index', 1.52)
    mat_L = problem.get('materials', {}).get('L', 1.38)
    mat_H = problem.get('materials', {}).get('H', 2.35)
    band = problem.get('band_nm', [400, 800])
    max_layers = problem.get('max_layers', 20)
    max_total = problem.get('max_total_nm', 1000)
    max_layer = problem.get('max_layer_nm', 400)
    
    # Wavelengths
    wavelengths = np.linspace(band[0], band[1], 401)
    
    def calc_reflectance(thicknesses, materials):
        # thicknesses: array of floats
        # materials: array of indices
        N = len(thicknesses)
        # Precompute delta for all wavelengths and layers
        # delta[j, k] = 2 * pi * n[j] * d[j] / lambda[k]
        # To save memory/time, we can compute the matrix product iteratively for each wavelength
        # But vectorizing over wavelengths is better.
        
        # M_total starts as identity
        # We compute the product of matrices for each wavelength.
        # Since N is small (5-20), we can loop over wavelengths or vectorize.
        # Vectorizing:
        # delta shape: (N, 401)
        delta = 2 * np.pi * np.outer(materials, thicknesses) / wavelengths
        # cos_delta, sin_delta
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        # We need to multiply matrices.
        # M = M0 * M1 * ... * MN
        # This is tricky to vectorize fully without a loop over layers because the matrix multiplication is not element-wise.
        # However, we can use the fact that the matrices are 2x2.
        # Let's loop over layers and update a matrix for all wavelengths at once.
        
        # Initialize M as identity for all wavelengths
        # M11, M12, M21, M22 are arrays of shape (401,)
        M11 = np.ones_like(wavelengths)
        M12 = np.zeros_like(wavelengths)
        M21 = np.zeros_like(wavelengths)
        M22 = np.ones_like(wavelengths)
        
        for j in range(N):
            n_j = materials[j]
            c_j = cos_d[j]
            s_j = sin_d[j]
            
            # Matrix A_j = [[cos, i/n sin], [i n sin, cos]]
            # Note: i is imaginary unit.
            # M_new = M_old * A_j
            # M11_new = M11*c_j + M12*(1j/n_j)*s_j
            # M12_new = M11*(1j/n_j)*s_j + M12*c_j
            # M21_new = M21*c_j + M22*(1j*n_j)*s_j
            # M22_new = M21*(1j*n_j)*s_j + M22*c_j
            
            term12 = (1j / n_j) * s_j
            term21 = (1j * n_j) * s_j
            
            new_M11 = M11 * c_j + M12 * term12
            new_M12 = M11 * term12 + M12 * c_j
            new_M21 = M21 * c_j + M22 * term21
            new_M22 = M21 * term21 + M22 * c_j
            
            M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
            
        # Admittance Y = (M21 + M22 * n_sub) / (M11 + M12 * n_sub)
        Y = (M21 + M22 * n_sub) / (M11 + M12 * n_sub)
        
        # Reflectance R = | (Y - n_amb) / (Y + n_amb) |^2
        R = np.abs((Y - n_amb) / (Y + n_amb))**2
        
        return np.mean(R)
    
    def objective(x):
        # x is thicknesses
        # Check time
        if time.time() - start_time > timeout:
            return 1e9 # Huge penalty to stop
        
        # Calculate mean reflectance
        # We need to assign materials to layers.
        # Let's assume a fixed pattern for the optimization, e.g., L, H, L, H, L
        # This reduces the search space to just thicknesses.
        # We can try a few patterns and pick the best.
        
        # For now, this function assumes a specific material assignment is passed or global.
        # Let's make the material assignment part of the outer loop.
        pass

    # We will define a helper that takes thicknesses and a material pattern
    def evaluate(thicknesses, pattern):
        materials = np.array([mat_L if p == 'L' else mat_H for p in pattern])
        return calc_reflectance(thicknesses, materials)
    
    best_score = 1.0 # Bare glass is ~4.26%, but score is relative? No, score is mean reflectance in percent?
    # "Score (minimize): mean normal-incidence reflectance in percent"
    # So 4.26 is the baseline. We want to go lower.
    
    # Try a few patterns
    patterns = [
        "LHLHL",
        "HLHLH",
        "LLHHLLHHL", # Just random longer ones? No, keep it simple first.
        "LHLHLHL" # 7 layers
    ]
    
    best_result = None
    best_val = float('inf')
    
    for pattern in patterns:
        if len(pattern) > max_layers:
            continue
            
        N = len(pattern)
        # Initial guess: quarter wave at 600nm
        d0_L = 600 / (4 * mat_L)
        d0_H = 600 / (4 * mat_H)
        x0 = np.array([d0_L if p == 'L' else d0_H for p in pattern])
        
        # Bounds
        bounds = [(0, max_layer) for _ in range(N)]
        
        # Objective function for this pattern
        def obj(x):
            if time.time() - start_time > timeout:
                return 1e9
            r = evaluate(x, pattern)
            # Penalty for total thickness
            total = np.sum(x)
            if total > max_total:
                r += 1000 * (total - max_total) # Strong penalty
            return r * 100 # Convert to percent? The evaluator uses percent.
            
        try:
            res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8})
            if res.success and res.fun < best_val:
                # Check constraints
                if np.sum(res.x) <= max_total and np.all(res.x >= 0) and np.all(res.x <= max_layer):
                    best_val = res.fun
                    best_result = (list(pattern), list(res.x))
        except Exception as e:
            pass
            
    if best_result is None:
        # Fallback to seed
        centre = 0.5 * (band[0] + band[1])
        return ["L"], [centre / (4.0 * mat_L)]
        
    return best_result

