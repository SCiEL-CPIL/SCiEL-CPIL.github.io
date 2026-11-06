"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _characteristic_matrix(ng, d, lam, n_sub):
    """Compute the characteristic matrix for a stack of layers.
    
    ng: list of refractive indices
    d: list of physical thicknesses in nm
    lam: wavelength in nm
    n_sub: substrate index
    """
    N = len(ng)
    # Start with the substrate admittance
    Y = n_sub
    for i in range(N - 1, -1, -1):
        n_i = ng[i]
        d_i = d[i]
        phase = 2.0 * np.pi * n_i * d_i / lam
        cos_phase = np.cos(phase)
        sin_phase = np.sin(phase)
        # Characteristic matrix for layer i
        M = np.array([[cos_phase, 1.0j * sin_phase / n_i],
                      [1.0j * n_i * sin_phase, cos_phase]], dtype=complex)
        # Combine with current Y
        Y = (M[0, 0] * Y + M[0, 1]) / (M[1, 0] * Y + M[1, 1])
    return Y

def _reflectance_stack(ng, d, lam, n_amb, n_sub):
    """Compute reflectance at a single wavelength."""
    Y = _characteristic_matrix(ng, d, lam, n_sub)
    # Reflectance
    r = (n_amb - Y) / (n_amb + Y)
    R = np.abs(r) ** 2
    return R

def _mean_reflectance(ng, d, band_nm, n_amb, n_sub):
    """Compute mean reflectance over the band."""
    lam_min, lam_max = band_nm
    wavelengths = np.linspace(lam_min, lam_max, 401)
    total_R = 0.0
    for lam in wavelengths:
        total_R += _reflectance_stack(ng, d, lam, n_amb, n_sub)
    return total_R / len(wavelengths) * 100.0  # in percent

def _objective(x, ng_labels, band_nm, n_amb, n_sub, max_layer_nm, max_total_nm):
    """Objective function for optimization.
    
    x: array of optical thicknesses (n_i * d_i) in nm
    ng_labels: list of 'L'/'H' labels
    """
    # Check constraints
    if len(x) != len(ng_labels):
        return 1000.0
    
    # Convert optical thicknesses to physical thicknesses
    materials = {'L': 1.38, 'H': 2.35}
    ng = []
    d = []
    total_d = 0.0
    for i, label in enumerate(ng_labels):
        n_i = materials[label]
        opt_thick = x[i]
        if opt_thick < 0 or opt_thick > n_i * max_layer_nm:
            return 1000.0
        d_i = opt_thick / n_i
        if d_i > max_layer_nm:
            return 1000.0
        ng.append(n_i)
        d.append(d_i)
        total_d += d_i
    
    if total_d > max_total_nm:
        return 1000.0
    
    return _mean_reflectance(ng, d, band_nm, n_amb, n_sub)

def _powell_optimize(ng_labels, band_nm, n_amb, n_sub, max_layer_nm, max_total_nm, x0, max_iter=200, tol=1e-8):
    """Run Powell optimization for given layer labels."""
    materials = {'L': 1.38, 'H': 2.35}
    n_layers = len(ng_labels)
    
    # Bounds for optical thicknesses
    bounds = []
    for label in ng_labels:
        n_i = materials[label]
        bounds.append((0.0, n_i * max_layer_nm))
    
    # Objective wrapper
    def obj(x):
        return _objective(x, ng_labels, band_nm, n_amb, n_sub, max_layer_nm, max_total_nm)
    
    try:
        result = minimize(obj, x0, method='Powell', bounds=bounds, 
                         options={'maxiter': max_iter, 'tol': tol})
        return result.x, result.fun
    except Exception:
        return x0, obj(x0)

def design(problem):
    """Design an antireflection coating stack."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    band_nm = problem["band_nm"]
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    materials = problem["materials"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    lam_min, lam_max = band_nm
    centre = 0.5 * (lam_min + lam_max)
    
    # Best solution found
    best_score = _mean_reflectance([materials['L']], [centre / (4.0 * materials['L'])], band_nm, n_amb, n_sub)
    best_materials = ['L']
    best_thicknesses = [centre / (4.0 * materials['L'])]
    
    # Try different numbers of layers with alternating H/L starting with L
    # We'll try 2, 4, 6, 8, 10, 12, 14 layers
    layer_counts = [2, 4, 6, 8, 10, 12, 14]
    
    for n_layers in layer_counts:
        if time.time() > deadline - 5.0:
            break
        if n_layers > max_layers:
            continue
        
        # Create alternating pattern starting with L
        ng_labels = []
        for i in range(n_layers):
            if i % 2 == 0:
                ng_labels.append('L')
            else:
                ng_labels.append('H')
        
        # Initial guess: quarter-wave at center wavelength
        x0 = []
        for label in ng_labels:
            n_i = materials[label]
            x0.append(n_i * centre / (4.0 * n_i))  # = centre/4
        x0 = np.array(x0)
        
        # Run Powell optimization
        x_opt, score = _powell_optimize(ng_labels, band_nm, n_amb, n_sub, max_layer_nm, max_total_nm, x0)
        
        if score < best_score:
            best_score = score
            best_materials = ng_labels[:]
            # Convert optical thicknesses to physical
            best_thicknesses = []
            for i, label in enumerate(ng_labels):
                n_i = materials[label]
                best_thicknesses.append(x_opt[i] / n_i)
        
        # Try another starting point: half-wave
        if time.time() > deadline - 5.0:
            break
        x0_half = []
        for label in ng_labels:
            n_i = materials[label]
            x0_half.append(n_i * centre / (2.0 * n_i))  # = centre/2
        x0_half = np.array(x0_half)
        
        x_opt2, score2 = _powell_optimize(ng_labels, band_nm, n_amb, n_sub, max_layer_nm, max_total_nm, x0_half)
        
        if score2 < best_score:
            best_score = score2
            best_materials = ng_labels[:]
            best_thicknesses = []
            for i, label in enumerate(ng_labels):
                n_i = materials[label]
                best_thicknesses.append(x_opt2[i] / n_i)
    
    # Try starting with H instead of L for even layer counts
    for n_layers in [2, 4, 6, 8]:
        if time.time() > deadline - 5.0:
            break
        if n_layers > max_layers:
            continue
        
        ng_labels = []
        for i in range(n_layers):
            if i % 2 == 0:
                ng_labels.append('H')
            else:
                ng_labels.append('L')
        
        x0 = []
        for label in ng_labels:
            n_i = materials[label]
            x0.append(n_i * centre / (4.0 * n_i))
        x0 = np.array(x0)
        
        x_opt, score = _powell_optimize(ng_labels, band_nm, n_amb, n_sub, max_layer_nm, max_total_nm, x0)
        
        if score < best_score:
            best_score = score
            best_materials = ng_labels[:]
            best_thicknesses = []
            for i, label in enumerate(ng_labels):
                n_i = materials[label]
                best_thicknesses.append(x_opt[i] / n_i)
    
    # Verify constraints
    total = sum(best_thicknesses)
    if total > max_total_nm:
        # Scale down
        scale = max_total_nm / total
        best_thicknesses = [t * scale for t in best_thicknesses]
    
    for i, t in enumerate(best_thicknesses):
        if t > max_layer_nm:
            best_thicknesses[i] = max_layer_nm
        if t < 0:
            best_thicknesses[i] = 0.0
    
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
