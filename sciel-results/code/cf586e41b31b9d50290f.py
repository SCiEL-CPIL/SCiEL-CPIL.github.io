"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_reflectance_stack(materials, thicknesses, n_ambient, n_substrate, wavelengths):
    """
    Computes reflectance for a stack of layers using the characteristic matrix method.
    materials: list of refractive indices
    thicknesses: list of thicknesses in nm
    n_ambient: refractive index of ambient
    n_substrate: refractive index of substrate
    wavelengths: array of wavelengths in nm
    Returns: array of reflectances (0 to 1)
    """
    # Initialize characteristic matrix M
    # M = [[1, 0], [0, 1]]
    M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    
    n_layers = len(materials)
    for i in range(n_layers):
        n = materials[i]
        d = thicknesses[i]
        # beta = 2 * pi * n * d / wl
        beta = 2.0 * np.pi * n * d / wavelengths
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        # Characteristic matrix for layer i
        # M_i = [[cos(beta), (1/(i*n))*sin(beta)], [i*n*sin(beta), cos(beta)]]
        # Note: 1/i = -i
        M_i = np.array([
            [cos_b, -1j * sin_b / n],
            [1j * n * sin_b, cos_b]
        ], dtype=complex)
        
        # Multiply M = M_i @ M
        M = M_i @ M
    
    # Calculate admittance Y
    # Y = (M[1,0] + M[1,1]*n_sub) / (M[0,0] + M[0,1]*n_sub)
    Y = (M[1, 0] + M[1, 1] * n_substrate) / (M[0, 0] + M[0, 1] * n_substrate)
    
    # Reflectance R = |(Y - n_ambient) / (Y + n_ambient)|^2
    r = (Y - n_ambient) / (Y + n_ambient)
    R = np.abs(r)**2
    
    return R

def design(problem):
    """
    Designs a broadband antireflection coating.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10s margin for 60s limit
    
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band_start, band_end = problem["band_nm"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Generate wavelengths
    wavelengths = np.linspace(band_start, band_end, 401)
    
    # Define stack structure
    # We use a 5-layer stack: L, H, L, H, L
    # This allows for better broadband performance than 3 layers
    materials_labels = ['L', 'H', 'L', 'H', 'L']
    materials_indices = [n_L, n_H, n_L, n_H, n_L]
    n_layers = len(materials_labels)
    
    # Initial guess: quarter-wave at center wavelength
    center_wl = 0.5 * (band_start + band_end)
    init_thicknesses = []
    for label in materials_labels:
        n = n_L if label == 'L' else n_H
        d = center_wl / (4.0 * n)
        init_thicknesses.append(d)
        
    init_thicknesses = np.array(init_thicknesses)
    
    # Objective function
    def objective(params):
        # Check time limit
        if time.time() > deadline:
            return 1e10  # Large penalty to stop
        
        thicknesses = params
        
        # Constraint checks
        # 1. Thicknesses must be in [0, max_layer_nm]
        if np.any(thicknesses < 0) or np.any(thicknesses > max_layer_nm):
            return 1e10
        
        # 2. Total thickness must be <= max_total_nm
        if np.sum(thicknesses) > max_total_nm:
            return 1e10
            
        # Compute reflectance
        R = _compute_reflectance_stack(materials_indices, thicknesses, n_ambient, n_substrate, wavelengths)
        
        # Mean reflectance in percent
        mean_R = np.mean(R) * 100.0
        
        return mean_R

    # Run optimization
    # Nelder-Mead is a good choice for small dimensional problems with constraints handled by penalty
    best_result = None
    best_score = 1e10
    
    try:
        res = minimize(
            objective, 
            init_thicknesses, 
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        
        if res.success or res.fun < best_score:
            best_thicknesses = res.x
            best_score = res.fun
            best_result = (materials_labels, list(best_thicknesses))
            
    except Exception as e:
        # Fallback to initial guess if optimization fails
        pass

    if best_result is None:
        # Return initial guess if no better result found
        best_result = (materials_labels, list(init_thicknesses))
        
    # Final validation
    labels, thicks = best_result
    if np.sum(thicks) > max_total_nm or np.any(np.array(thicks) > max_layer_nm) or np.any(np.array(thicks) < 0):
        # Fallback to a very thin single layer or just the initial guess if it was valid
        # Initial guess is usually valid
        best_result = (materials_labels, list(init_thicknesses))
        
    return best_result
# EVOLVE-BLOCK-END
