import numpy as np
import time
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _compute_reflectance_vectorized(materials, thicknesses, wavelengths, n0, ns, nL, nH):
    """
    Compute reflectance for a stack of layers at multiple wavelengths.
    materials: list of 'L' or 'H'
    thicknesses: list of floats (nm)
    wavelengths: numpy array of wavelengths (nm)
    Returns numpy array of reflectances (0 to 1).
    """
    n_layers = len(materials)
    if n_layers == 0:
        r = (n0 - ns) / (n0 + ns)
        return np.full_like(wavelengths, abs(r)**2)
    
    # Pre-compute layer indices
    n_layers_arr = np.array([nL if m == 'L' else nH for m in materials])
    d_layers_arr = np.array(thicknesses)
    
    # Initialize total matrix as identity for all wavelengths
    # M_total shape: (num_wavelengths, 2, 2)
    M_total = np.zeros((len(wavelengths), 2, 2), dtype=complex)
    M_total[:, 0, 0] = 1.0
    M_total[:, 0, 1] = 0.0
    M_total[:, 1, 0] = 0.0
    M_total[:, 1, 1] = 1.0
    
    for i in range(n_layers):
        n_i = n_layers_arr[i]
        d_i = d_layers_arr[i]
        
        # beta = 2*pi*n*d / lambda
        # Use complex dtype to be safe, though for lossless it's real
        beta = 2.0 * np.pi * n_i * d_i / wavelengths
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        # M_layer = [[cos_b, sin_b/(i*n)], [i*n*sin_b, cos_b]]
        # Note: 1/i = -i, so sin_b/(i*n) = -1j * sin_b / n
        # i*n*sin_b = 1j * n * sin_b
        
        M_layer = np.zeros((len(wavelengths), 2, 2), dtype=complex)
        M_layer[:, 0, 0] = cos_b
        M_layer[:, 0, 1] = -1j * sin_b / n_i
        M_layer[:, 1, 0] = 1j * n_i * sin_b
        M_layer[:, 1, 1] = cos_b
        
        # M_total = M_total @ M_layer
        # Matrix multiplication for each wavelength
        # M_total[:, 0, 0] = M_total[:, 0, 0]*M_layer[:, 0, 0] + M_total[:, 0, 1]*M_layer[:, 1, 0]
        # etc.
        M_total[:, 0, 0] = M_total[:, 0, 0] * M_layer[:, 0, 0] + M_total[:, 0, 1] * M_layer[:, 1, 0]
        M_total[:, 0, 1] = M_total[:, 0, 0] * M_layer[:, 0, 1] + M_total[:, 0, 1] * M_layer[:, 1, 1]
        M_total[:, 1, 0] = M_total[:, 1, 0] * M_layer[:, 0, 0] + M_total[:, 1, 1] * M_layer[:, 1, 0]
        M_total[:, 1, 1] = M_total[:, 1, 0] * M_layer[:, 0, 1] + M_total[:, 1, 1] * M_layer[:, 1, 1]
        
        # Wait, the above in-place update is incorrect because M_total[:, 0, 0] is used in the next line.
        # Need to store old values or use temporary arrays.
        
    # Let's rewrite the multiplication properly to avoid in-place errors
    M_total = np.zeros((len(wavelengths), 2, 2), dtype=complex)
    M_total[:, 0, 0] = 1.0
    M_total[:, 1, 1] = 1.0
    
    for i in range(n_layers):
        n_i = n_layers_arr[i]
        d_i = d_layers_arr[i]
        
        beta = 2.0 * np.pi * n_i * d_i / wavelengths
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        M_l11 = cos_b
        M_l12 = -1j * sin_b / n_i
        M_l21 = 1j * n_i * sin_b
        M_l22 = cos_b
        
        # M_new = M_total @ M_layer
        M_t11 = M_total[:, 0, 0]
        M_t12 = M_total[:, 0, 1]
        M_t21 = M_total[:, 1, 0]
        M_t22 = M_total[:, 1, 1]
        
        M_total[:, 0, 0] = M_t11 * M_l11 + M_t12 * M_l21
        M_total[:, 0, 1] = M_t11 * M_l12 + M_t12 * M_l22
        M_total[:, 1, 0] = M_t21 * M_l11 + M_t22 * M_l21
        M_total[:, 1, 1] = M_t21 * M_l12 + M_t22 * M_l22
        
    # Admittance Y_in = (M21 + M22*ns) / (M11 + M12*ns)
    M11 = M_total[:, 0, 0]
    M12 = M_total[:, 0, 1]
    M21 = M_total[:, 1, 0]
    M22 = M_total[:, 1, 1]
    
    Y_in = (M21 + M22 * ns) / (M11 + M12 * ns)
    
    # Reflectance R = |(n0 - Y_in) / (n0 + Y_in)|^2
    r = (n0 - Y_in) / (n0 + Y_in)
    R = np.abs(r)**2
    
    return R

def design(problem):
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    band0, band1 = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Define wavelengths for evaluation
    wavelengths = np.linspace(band0, band1, 401)
    
    # Initial design: Symmetric stack of 10 layers (5 pairs of LH)
    # Center wavelength 600 nm
    lambda_c = 0.5 * (band0 + band1)
    opt_thick = lambda_c / 4.0
    dL_init = opt_thick / nL
    dH_init = opt_thick / nH
    
    # Create a 10-layer stack: L H L H L H L H L H
    # Or symmetric: L H L H L H L H L H is not symmetric in index if we just alternate.
    # A symmetric stack in terms of optical thicknesses is better.
    # Let's try 12 layers: L H H L L H H L L H H L ? No, let's stick to simple alternation first.
    # Actually, a "graded" index is best. 
    # Let's start with 20 layers, alternating L and H, starting with L (closest to air n=1.0)
    # nL=1.38 is closer to 1.0 than nH=2.35. So start with L.
    
    n_layers = 20
    materials = []
    thicknesses = []
    for i in range(n_layers):
        if i % 2 == 0:
            materials.append('L')
            thicknesses.append(dL_init)
        else:
            materials.append('H')
            thicknesses.append(dH_init)
            
    # Check validity
    if sum(thicknesses) > max_total_nm:
        # Scale down
        scale = max_total_nm / sum(thicknesses)
        thicknesses = [t * scale for t in thicknesses]
        
    def objective(thick_params):
        if time.time() > deadline:
            return float('inf')
            
        # Ensure thicknesses are within bounds
        thicks = np.clip(thick_params, 0, max_layer_nm)
        
        # Check total thickness constraint
        if np.sum(thicks) > max_total_nm:
            return float('inf')
            
        # Compute mean reflectance
        R = _compute_reflectance_vectorized(materials, thicks, wavelengths, n0, ns, nL, nH)
        return np.mean(R) * 100.0 # Return in percent
        
    # Optimize
    x0 = np.array(thicknesses)
    
    # Use Nelder-Mead for local optimization
    # This is robust and doesn't require gradients
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'xatol': 0.1, 'fatol': 0.001, 'maxiter': 1000, 'maxfev': 1000})
        
        if res.success and res.fun < float('inf'):
            best_thicks = np.clip(res.x, 0, max_layer_nm)
            # Re-check total thickness
            if np.sum(best_thicks) <= max_total_nm:
                thicknesses = list(best_thicks)
            else:
                # Scale if necessary
                scale = max_total_nm / np.sum(best_thicks)
                thicknesses = list(best_thicks * scale)
                
    except Exception:
        pass # Keep initial design if optimization fails
        
    # Final check and return
    if sum(thicknesses) > max_total_nm:
        scale = max_total_nm / sum(thicknesses)
        thicknesses = [t * scale for t in thicknesses]
        
    # Ensure no negative or too large
    thicknesses = [max(0, min(t, max_layer_nm)) for t in thicknesses]
    
    return materials, thicknesses

# EVOLVE-BLOCK-END
