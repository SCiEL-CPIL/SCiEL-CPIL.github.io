"""Antireflection coating design via global search with gradient parameterization and local refinement."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

def _reflectance_spectrum(thicknesses, labels, n_amb, n_sub, n_L, n_H, wavelengths):
    """
    Compute reflectance (fraction, 0-1) at each wavelength using the
    characteristic-matrix method (vectorized).
    """
    n = len(thicknesses)
    if n == 0:
        # Bare substrate
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return np.full_like(wavelengths, r * r, dtype=float)
    
    # Precompute refractive indices for each layer
    n_layers = np.array([n_L if lab == 'L' else n_H for lab in labels], dtype=float)
    t_arr = np.array(thicknesses, dtype=float)
    
    # wavelengths shape: (N,)
    # n_layers shape: (L,)
    # We want beta shape: (L, N)
    # beta[j, i] = 2*pi*n_layers[j]*t_arr[j] / wavelengths[i]
    beta = 2.0 * np.pi * (n_layers * t_arr)[:, np.newaxis] / wavelengths[np.newaxis, :]
    
    c = np.cos(beta) # (L, N)
    s = np.sin(beta) # (L, N)
    
    # Initialize total matrix M for all wavelengths
    # M shape: (2, 2, N)
    M = np.zeros((2, 2, len(wavelengths)), dtype=complex)
    M[0, 0, :] = 1.0
    M[1, 1, :] = 1.0
    
    # Multiply layer matrices
    for j in range(n):
        n_j = n_layers[j]
        
        M00 = M[0, 0, :]
        M01 = M[0, 1, :]
        M10 = M[1, 0, :]
        M11 = M[1, 1, :]
        
        c_j = c[j, :]
        s_j = s[j, :]
        
        M_j_00 = c_j
        M_j_01 = 1j * s_j / n_j
        M_j_10 = 1j * n_j * s_j
        M_j_11 = c_j
        
        M[0, 0, :] = M00 * M_j_00 + M01 * M_j_10
        M[0, 1, :] = M00 * M_j_01 + M01 * M_j_11
        M[1, 0, :] = M10 * M_j_00 + M11 * M_j_10
        M[1, 1, :] = M10 * M_j_01 + M11 * M_j_11
        
    M00 = M[0, 0, :]
    M01 = M[0, 1, :]
    M10 = M[1, 0, :]
    M11 = M[1, 1, :]
    
    numerator = n_amb * M00 + M10 - n_amb * M01 - M11
    denominator = n_amb * M00 + M10 + n_amb * M01 + M11
    
    r = numerator / denominator
    reflectances = np.abs(r) ** 2
    
    return reflectances

def design(problem):
    """
    Design an antireflection coating stack.
    Returns (materials, thicknesses).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n_amb = problem.get("ambient_index", 1.0)
    n_sub = problem.get("substrate_index", 1.52)
    materials = problem.get("materials", {'L': 1.38, 'H': 2.35})
    n_L = materials['L']
    n_H = materials['H']
    band = problem.get("band_nm", [400, 800])
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    # Wavelengths for evaluation
    wavelengths = np.linspace(band[0], band[1], 401)
    
    def objective(labels, thicknesses):
        """Return mean reflectance in percent."""
        n = len(thicknesses)
        if n == 0:
            r = (n_amb - n_sub) / (n_amb + n_sub)
            return 100.0 * r * r
        refl = _reflectance_spectrum(thicknesses, labels, n_amb, n_sub, n_L, n_H, wavelengths)
        return 100.0 * np.mean(refl)
    
    def _check_time():
        return time.time() < deadline

    best_labels = []
    best_thicknesses = []
    best_score = float('inf')
    
    # Strategy:
    # 1. Try several stack sizes (n_layers).
    # 2. For each size, use Differential Evolution to optimize a linear gradient parameterization.
    # 3. Refine the best DE result using Nelder-Mead on individual thicknesses.
    
    # Candidate stack sizes: Even numbers often work well for symmetric stacks, but odd is fine too.
    # Let's try 4, 6, 8, 10, 12.
    candidate_sizes = [4, 6, 8, 10, 12]
    
    for n_layers in candidate_sizes:
        if not _check_time():
            break
        if n_layers > max_layers:
            continue
            
        # Define the parameterization for DE
        # thicknesses[i] = base_t + i * gradient
        # We need to ensure 0 <= t_i <= max_layer_nm and sum(t) <= max_total_nm
        # DE bounds for base_t and gradient:
        # If we pick base_t in [0, max_layer_nm] and gradient in [-max_layer_nm, max_layer_nm],
        # we might violate constraints. We will penalize invalid solutions in the objective.
        
        def de_objective(params):
            base_t, gradient = params
            t = base_t + np.arange(n_layers) * gradient
            # Check constraints
            if np.any(t < 0) or np.any(t > max_layer_nm):
                return 1000.0 # Penalty
            if np.sum(t) > max_total_nm:
                return 1000.0 # Penalty
            
            labels = ['L', 'H'] * (n_layers // 2)
            if len(labels) < n_layers:
                labels += ['L'] * (n_layers - len(labels))
            labels = labels[:n_layers]
            
            return objective(labels, t)
        
        bounds = [(0, max_layer_nm), (-max_layer_nm, max_layer_nm)]
        
        try:
            # Run Differential Evolution
            # maxiter and popsize control the search effort.
            # We have ~20s per stack size if we have 5 sizes and 100s total.
            # DE is stochastic, so keep it moderate.
            res = differential_evolution(
                de_objective, 
                bounds, 
                maxiter=50, 
                popsize=10, 
                seed=42,
                tol=1e-4
            )
            
            if res.success or res.fun < best_score:
                base_t, gradient = res.x
                t = base_t + np.arange(n_layers) * gradient
                if np.any(t < 0) or np.any(t > max_layer_nm) or np.sum(t) > max_total_nm:
                    continue
                
                labels = ['L', 'H'] * (n_layers // 2)
                if len(labels) < n_layers:
                    labels += ['L'] * (n_layers - len(labels))
                labels = labels[:n_layers]
                
                # Refine with Nelder-Mead
                if _check_time():
                    def nm_objective(t_arr):
                        if np.any(t_arr < 0) or np.any(t_arr > max_layer_nm):
                            return 1000.0
                        if np.sum(t_arr) > max_total_nm:
                            return 1000.0
                        return objective(labels, t_arr)
                    
                    nm_res = minimize(
                        nm_objective, 
                        t, 
                        method='Nelder-Mead', 
                        options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5}
                    )
                    
                    if nm_res.fun < best_score:
                        best_score = nm_res.fun
                        best_thicknesses = nm_res.x
                        best_labels = labels
                else:
                    # Just take DE result if no time for NM
                    score = objective(labels, t)
                    if score < best_score:
                        best_score = score
                        best_thicknesses = t
                        best_labels = labels
                        
        except Exception:
            pass
    
    # Fallback: If DE failed or didn't improve, use a simple quarter-wave stack
    if not best_labels:
        centre_wl = 0.5 * (band[0] + band[1])
        n_init = 6
        labels = ['L', 'H'] * (n_init // 2)
        thicknesses = [centre_wl / (4.0 * (n_L if lab == 'L' else n_H)) for lab in labels]
        if sum(thicknesses) > max_total_nm:
            scale = max_total_nm / sum(thicknesses)
            thicknesses = [t * scale for t in thicknesses]
        thicknesses = [min(t, max_layer_nm) for t in thicknesses]
        best_labels = labels
        best_thicknesses = thicknesses
        best_score = objective(best_labels, best_thicknesses)
        
    # Ensure final constraints are met
    final_thicknesses = []
    for t in best_thicknesses:
        t = max(0.0, min(float(t), max_layer_nm))
        final_thicknesses.append(t)
    
    if sum(final_thicknesses) > max_total_nm:
        scale = max_total_nm / sum(final_thicknesses)
        final_thicknesses = [t * scale for t in final_thicknesses]
        final_thicknesses = [min(t, max_layer_nm) for t in final_thicknesses]
        
    return best_labels, final_thicknesses
