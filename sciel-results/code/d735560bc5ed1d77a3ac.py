"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _reflectance_vectorized(n_layers, n_layers_arr, thicknesses, wavelengths, n_ambient, n_substrate):
    """
    Calculate reflectance spectrum using characteristic matrix method.
    n_layers_arr: array of refractive indices for the stack
    thicknesses: array of physical thicknesses in nm
    wavelengths: array of wavelengths in nm
    """
    # k = 2 * pi * n * d / lambda
    # For each wavelength, we need to compute the product of matrices.
    # Vectorization trick: we can't easily vectorize the matrix product over layers 
    # without loops, but we can vectorize over wavelengths for each layer.
    
    # Let's compute phase factors for all layers and all wavelengths
    # shape: (n_layers, n_wavelengths)
    k = 2.0 * np.pi * np.outer(n_layers_arr, thicknesses) / wavelengths[np.newaxis, :]
    
    # Initialize M as identity for all wavelengths
    # M will be (n_wavelengths, 2, 2)
    M = np.zeros((len(wavelengths), 2, 2), dtype=np.complex128)
    M[:, 0, 0] = 1.0
    M[:, 1, 1] = 1.0
    
    # Iterate through layers
    for i in range(n_layers):
        n_i = n_layers_arr[i]
        # cos(k_i) and sin(k_i)
        cos_k = np.cos(k[i, :])
        sin_k = np.sin(k[i, :])
        
        # Matrix for layer i: [[cos, i*sin/n], [i*n*sin, cos]]
        # We need to multiply M = M @ M_i
        # M is (N, 2, 2). M_i is conceptually (2, 2) but parameters vary with wavelength.
        
        # Let's extract components of M
        m11 = M[:, 0, 0]
        m12 = M[:, 0, 1]
        m21 = M[:, 1, 0]
        m22 = M[:, 1, 1]
        
        # New M11 = m11 * cos_k + m12 * (1j * sin_k / n_i)
        # New M12 = m11 * (1j * n_i * sin_k) + m12 * cos_k
        # New M21 = m21 * cos_k + m22 * (1j * sin_k / n_i)
        # New M22 = m21 * (1j * n_i * sin_k) + m22 * cos_k
        
        new_m11 = m11 * cos_k + m12 * (1j * sin_k / n_i)
        new_m12 = m11 * (1j * n_i * sin_k) + m12 * cos_k
        new_m21 = m21 * cos_k + m22 * (1j * sin_k / n_i)
        new_m22 = m21 * (1j * n_i * sin_k) + m22 * cos_k
        
        M[:, 0, 0] = new_m11
        M[:, 0, 1] = new_m12
        M[:, 1, 0] = new_m21
        M[:, 1, 1] = new_m22
        
    # Final step: multiply by substrate interface matrix? 
    # Standard CM method: 
    # M_total = M_1 M_2 ... M_N
    # Then Y = M_total @ [1, n_sub] / n_sub? 
    # Actually, the characteristic matrix M maps the field at the bottom of the stack to the top.
    # Let [E, H]_top = M @ [E, H]_bottom
    # At the bottom (substrate), H = n_sub * E (for normal incidence, relative admittance).
    # So [E, H]_bottom = [1, n_sub]^T * E_bottom
    # Then [E, H]_top = M @ [1, n_sub]^T * E_bottom
    # Input wave at top: E_in + E_r, H_in + H_r. 
    # Transmitted wave: E_t.
    # Boundary conditions:
    # E_top = E_in + E_r
    # H_top = H_in - H_r (assuming H direction convention, usually H = n * E for forward, -n*E for backward? 
    # Let's use the standard formula: R = (n_0 - Y) / (n_0 + Y) where Y is the input admittance.
    # Y = (M_11 * n_sub + M_12) / (M_21 * n_sub + M_22)
    
    m11 = M[:, 0, 0]
    m12 = M[:, 0, 1]
    m21 = M[:, 1, 0]
    m22 = M[:, 1, 1]
    
    Y = (m11 * n_substrate + m12) / (m21 * n_substrate + m22)
    
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.abs(r) ** 2
    
    return R

def _mean_reflectance(materials, thicknesses, problem):
    n_L = problem['materials']['L']
    n_H = problem['materials']['H']
    n_ambient = problem['ambient_index']
    n_sub = problem['substrate_index']
    band = problem['band_nm']
    
    # Fixed structure for this strategy: LHHLHL
    # This is a common starting point for broadband AR coatings (quarter wave stack variations)
    # We will optimize the thicknesses.
    labels = ['L', 'H', 'H', 'L', 'L', 'H']
    n_layers = len(labels)
    
    if len(thicknesses) != n_layers:
        return 100.0 # Invalid
        
    n_layers_arr = np.array([n_L if l == 'L' else n_H for l in labels], dtype=float)
    
    # Check constraints
    max_total = problem.get('max_total_nm', 1000)
    max_layer = problem.get('max_layer_nm', 400)
    
    if np.sum(thicknesses) > max_total:
        return 100.0
    if np.any(thicknesses < 0) or np.any(thicknesses > max_layer):
        return 100.0
        
    # Wavelengths: 401 points from 400 to 800
    wavelengths = np.linspace(band[0], band[1], 401)
    
    R = _reflectance_vectorized(n_layers, n_layers_arr, thicknesses, wavelengths, n_ambient, n_sub)
    
    return np.mean(R) * 100.0

def _nelder_mead(func, x0, bounds, max_iter=200, tol=1e-6):
    """
    Simple Nelder-Mead implementation with bounds handling (reflection only if out of bounds, else shrink).
    """
    n = len(x0)
    simplex = [x0]
    for i in range(n):
        x = x0.copy()
        # Perturb
        if x[i] + 0.05 * (bounds[i][1] - bounds[i][0]) > bounds[i][1]:
            x[i] = bounds[i][1] - 0.01
        else:
            x[i] = x[i] + 0.05 * (bounds[i][1] - bounds[i][0])
        simplex.append(x)
        
    simplex = np.array(simplex)
    f_simplex = np.array([func(x) for x in simplex])
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for i in range(max_iter):
        # Sort simplex
        order = np.argsort(f_simplex)
        simplex = simplex[order]
        f_simplex = f_simplex[order]
        
        if np.abs(f_simplex[0] - f_simplex[-1]) < tol:
            break
            
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = centroid + alpha * (centroid - simplex[-1])
        
        # Check bounds
        if np.any(x_r < bounds[:, 0]) or np.any(x_r > bounds[:, 1]):
            # Shrink towards best
            x_s = simplex[0] + sigma * (simplex[-1] - simplex[0])
            f_s = func(x_s)
            simplex[-1] = x_s
            f_simplex[-1] = f_s
            continue
            
        f_r = func(x_r)
        
        if f_r < f_simplex[0]:
            # Expansion
            x_e = centroid + gamma * (x_r - centroid)
            if np.any(x_e < bounds[:, 0]) or np.any(x_e > bounds[:, 1]):
                # Accept reflection if expansion is out of bounds
                simplex[-1] = x_r
                f_simplex[-1] = f_r
            else:
                f_e = func(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_simplex[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
        elif f_r < f_simplex[-2]:
            simplex[-1] = x_r
            f_simplex[-1] = f_r
        else:
            # Contraction
            x_c = centroid + rho * (simplex[-1] - centroid)
            if np.any(x_c < bounds[:, 0]) or np.any(x_c > bounds[:, 1]):
                # Shrink
                x_s = simplex[0] + sigma * (simplex[-1] - simplex[0])
                f_s = func(x_s)
                simplex[-1] = x_s
                f_simplex[-1] = f_s
                continue
                
            f_c = func(x_c)
            if f_c < f_r:
                simplex[-1] = x_c
                f_simplex[-1] = f_c
            else:
                # Shrink
                for j in range(1, n+1):
                    simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                    f_simplex[j] = func(simplex[j])
                    
    return simplex[0], f_simplex[0]

def design(problem):
    start_time = time.time()
    deadline = start_time + 50.0 # Leave safety margin
    
    n_L = problem['materials']['L']
    n_H = problem['materials']['H']
    max_layer = problem.get('max_layer_nm', 400)
    max_total = problem.get('max_total_nm', 1000)
    
    # Structure: LHHLHL
    # Initial guess: Quarter wave at center wavelength 600nm
    lam0 = 600.0
    d_L = lam0 / (4 * n_L)
    d_H = lam0 / (4 * n_H)
    
    # Bounds for each layer
    # We want to keep total < max_total. 
    # Simple bounds: [1, max_layer] for each, but we might hit total limit.
    # Nelder-Mead with bounds is tricky. 
    # Let's use a penalty in the objective function for total thickness.
    
    def objective(x):
        # x is thicknesses
        if np.any(x < 0.1) or np.any(x > max_layer):
            return 1000.0
        if np.sum(x) > max_total:
            # Penalty proportional to excess
            return _mean_reflectance(problem['materials'], x, problem) + 10 * (np.sum(x) - max_total)
        return _mean_reflectance(problem['materials'], x, problem)
    
    best_x = None
    best_f = float('inf')
    
    # Multiple starts
    starts = [
        [d_L, d_H, d_H, d_L, d_L, d_H],
        [d_L * 0.8, d_H * 0.8, d_H * 0.8, d_L * 0.8, d_L * 0.8, d_H * 0.8],
        [d_L * 1.2, d_H * 1.2, d_H * 1.2, d_L * 1.2, d_L * 1.2, d_H * 1.2],
        [d_L, d_H, d_H, d_L, d_L, d_H], # Duplicate for safety? No, let's vary more
        [d_L * 0.5, d_H * 0.5, d_H * 0.5, d_L * 0.5, d_L * 0.5, d_H * 0.5],
        [d_L * 1.5, d_H * 1.5, d_H * 1.5, d_L * 1.5, d_L * 1.5, d_H * 1.5],
    ]
    
    # Add some random starts if time permits
    np.random.seed(42)
    for _ in range(5):
        x_rand = np.random.uniform(10, max_layer, 6)
        # Scale to fit roughly within total
        scale = max_total / np.sum(x_rand) * 0.9
        x_rand = x_rand * scale
        starts.append(x_rand)
        
    bounds = np.array([[1.0, max_layer]] * 6)
    
    for x0 in starts:
        if time.time() > deadline:
            break
            
        # Clip initial point to bounds
        x0 = np.clip(x0, bounds[:, 0], bounds[:, 1])
        
        x_opt, f_opt = _nelder_mead(objective, x0, bounds, max_iter=100, tol=1e-5)
        
        if f_opt < best_f:
            best_f = f_opt
            best_x = x_opt
            
    if best_x is None:
        # Fallback to seed
        centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
        return ["L"], [centre / (4.0 * problem["materials"]["L"])]
        
    labels = ['L', 'H', 'H', 'L', 'L', 'H']
    return labels, list(best_x)
# EVOLVE-BLOCK-END
