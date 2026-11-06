"""
Broadband antireflection coating design via adjoint-based gradient + L-BFGS-B.

Strategy:
- Represent the stack as a piecewise-constant index profile
- Use adjoint method to compute exact gradients of mean reflectance w.r.t. thicknesses
- Optimize with L-BFGS-B from multiple diverse initializations
- Search over several layer configurations (alternating L/H)
"""
import numpy as np
import time
from scipy.optimize import minimize

def _compute_reflectance(wavelengths_nm, indices, thicknesses_nm, ambient_idx, substrate_idx):
    """
    Compute reflectance (fraction, 0-1) at each wavelength using the characteristic matrix method.
    Also returns the Y_in (input admittance) for adjoint computation.
    """
    n_layers = len(indices)
    if n_layers == 0:
        r0 = (ambient_idx - substrate_idx) / (ambient_idx + substrate_idx)
        return np.full(len(wavelengths_nm), r0 * r0), None
    
    k = 2.0 * np.pi / wavelengths_nm
    
    M00 = np.ones(len(wavelengths_nm), dtype=complex)
    M01 = np.zeros(len(wavelengths_nm), dtype=complex)
    M10 = np.zeros(len(wavelengths_nm), dtype=complex)
    M11 = np.ones(len(wavelengths_nm), dtype=complex)
    
    for i in range(n_layers):
        n_i = indices[i]
        d_i = thicknesses_nm[i]
        delta = k * n_i * d_i
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        B00 = cos_d
        B01 = 1j * sin_d / n_i
        B10 = 1j * n_i * sin_d
        B11 = cos_d
        
        new_M00 = M00 * B00 + M01 * B10
        new_M01 = M00 * B01 + M01 * B11
        new_M10 = M10 * B00 + M11 * B10
        new_M11 = M10 * B01 + M11 * B11
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    Y_out = substrate_idx
    Y_in = (M10 + M11 * Y_out) / (M00 + M01 * Y_out)
    
    r = (ambient_idx - Y_in) / (ambient_idx + Y_in)
    R = np.abs(r) ** 2
    
    return R, Y_in

def _objective_and_gradient(x, indices, wavelengths_nm, ambient_idx, substrate_idx,
                             max_layer_nm, max_total_nm):
    """
    Compute mean reflectance in percent and its gradient w.r.t. thicknesses.
    Uses adjoint method for exact gradient.
    """
    n_layers = len(indices)
    x = np.asarray(x, dtype=float)
    
    # Clip to valid range
    x_clipped = np.clip(x, 0.0, max_layer_nm)
    
    # Check total thickness constraint
    total = np.sum(x_clipped)
    if total > max_total_nm:
        # Scale down
        x_clipped = x_clipped * (max_total_nm / total)
    
    R, Y_in = _compute_reflectance(wavelengths_nm, indices, x_clipped, 
                                    ambient_idx, substrate_idx)
    obj = np.mean(R) * 100.0
    
    # Adjoint gradient computation
    # dR/dx_i = 2*Re(r* * dr/dx_i)
    # dr/dY_in = 2*ambient_idx / (ambient_idx + Y_in)^2
    # dY_in/dx_i computed via adjoint
    
    k = 2.0 * np.pi / wavelengths_nm
    n_wl = len(wavelengths_nm)
    
    r = (ambient_idx - Y_in) / (ambient_idx + Y_in)
    dr_dY = 2.0 * ambient_idx / (ambient_idx + Y_in)**2
    
    # Forward pass: store B matrices and intermediate M matrices
    M00_arr = np.ones((n_wl,), dtype=complex)
    M01_arr = np.zeros((n_wl,), dtype=complex)
    M10_arr = np.zeros((n_wl,), dtype=complex)
    M11_arr = np.ones((n_wl,), dtype=complex)
    
    # Store B matrices for each layer
    B_matrices = []
    for i in range(n_layers):
        n_i = indices[i]
        d_i = x_clipped[i]
        delta = k * n_i * d_i
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        B00 = cos_d
        B01 = 1j * sin_d / n_i
        B10 = 1j * n_i * sin_d
        B11 = cos_d
        
        B_matrices.append((B00, B01, B10, B11, n_i, d_i, delta))
        
        new_M00 = M00_arr * B00 + M01_arr * B10
        new_M01 = M00_arr * B01 + M01_arr * B11
        new_M10 = M10_arr * B00 + M11_arr * B10
        new_M11 = M10_arr * B01 + M11_arr * B11
        
        M00_arr, M01_arr, M10_arr, M11_arr = new_M00, new_M01, new_M10, new_M11
    
    # Backward pass: adjoint variables
    # We need dY_in/dx_i for each i
    # Y_in = (M10 + M11*Y_out)/(M00 + M01*Y_out)
    # dY_in = (dM10 + dM11*Y_out)*(M00+M01*Y_out) - (M10+M11*Y_out)*(dM00+dM01*Y_out) / (M00+M01*Y_out)^2
    
    Y_out = substrate_idx
    denom = M00_arr + M01_arr * Y_out
    numer = M10_arr + M11_arr * Y_out
    
    # Adjoint: propagate sensitivity backward
    # Let lambda be such that d_obj = sum_i <lambda_i, dB_i/dx_i>
    # Simpler: compute dM/dx_i for each i directly
    
    grad = np.zeros(n_layers, dtype=float)
    
    for i in range(n_layers):
        B00, B01, B10, B11, n_i, d_i, delta = B_matrices[i]
        
        # dB/dx_i:
        # delta = k * n_i * x_i
        # d(delta)/dx_i = k * n_i
        dd_dx = k * n_i
        
        # B00 = cos(delta), dB00/dx = -sin(delta) * dd_dx
        dB00 = -np.sin(delta) * dd_dx
        # B01 = 1j*sin(delta)/n_i, dB01/dx = 1j*cos(delta)*dd_dx/n_i
        dB01 = 1j * np.cos(delta) * dd_dx / n_i
        # B10 = 1j*n_i*sin(delta), dB10/dx = 1j*n_i*cos(delta)*dd_dx
        dB10 = 1j * n_i * np.cos(delta) * dd_dx
        # B11 = cos(delta), dB11/dx = -sin(delta) * dd_dx
        dB11 = -np.sin(delta) * dd_dx
        
        # Need M matrices before and after layer i
        # Recompute forward up to layer i-1
        M00_pre = np.ones(n_wl, dtype=complex)
        M01_pre = np.zeros(n_wl, dtype=complex)
        M10_pre = np.zeros(n_wl, dtype=complex)
        M11_pre = np.ones(n_wl, dtype=complex)
        
        for j in range(i):
            B00_j, B01_j, B10_j, B11_j, _, _, _ = B_matrices[j]
            new_M00 = M00_pre * B00_j + M01_pre * B10_j
            new_M01 = M00_pre * B01_j + M01_pre * B11_j
            new_M10 = M10_pre * B00_j + M11_pre * B10_j
            new_M11 = M10_pre * B01_j + M11_pre * B11_j
            M00_pre, M01_pre, M10_pre, M11_pre = new_M00, new_M01, new_M10, new_M11
        
        # M after layer i
        M00_post = M00_pre * B00 + M01_pre * B10
        M01_post = M00_pre * B01 + M01_pre * B11
        M10_post = M10_pre * B00 + M11_pre * B10
        M11_post = M10_pre * B01 + M11_pre * B11
        
        # dM/dx_i (partial derivative of the product through layer i)
        # M_total = M_after * B_i * M_before
        # dM_total/dx_i = M_after * dB_i/dx_i * M_before
        
        # For the full stack, M_total = B_N * ... * B_1
        # dM_total/dx_i = B_N * ... * B_{i+1} * (dB_i/dx_i) * B_{i-1} * ... * B_1
        
        # We need M matrices after layer i (from layer i+1 to N) and before layer i (from 1 to i-1)
        # M_after_layer_i = B_N * ... * B_{i+1}
        # M_before_layer_i = B_{i-1} * ... * B_1
        
        # Actually, let's think of it differently.
        # M_total = M_{N..i+1} * B_i * M_{i-1..1}
        # where M_{N..i+1} is the product from layer N down to i+1
        # and M_{i-1..1} is the product from layer i-1 down to 1
        
        # We have M00_pre, etc. which is M_{i-1..1}
        # We need M_{N..i+1}
        
        M00_suf = np.ones(n_wl, dtype=complex)
        M01_suf = np.zeros(n_wl, dtype=complex)
        M10_suf = np.zeros(n_wl, dtype=complex)
        M11_suf = np.ones(n_wl, dtype=complex)
        
        for j in range(n_layers - 1, i, -1):
            B00_j, B01_j, B10_j, B11_j, _, _, _ = B_matrices[j]
            new_M00 = M00_suf * B00_j + M01_suf * B10_j
            new_M01 = M00_suf * B01_j + M01_suf * B11_j
            new_M10 = M10_suf * B00_j + M11_suf * B10_j
            new_M11 = M10_suf * B01_j + M11_suf * B11_j
            M00_suf, M01_suf, M10_suf, M11_suf = new_M00, new_M01, new_M10, new_M11
        
        # dM_total = M_suf * dB_i * M_pre
        dM00 = M00_suf * dB00 * M00_pre + M00_suf * dB10 * M10_pre
        dM01 = M00_suf * dB01 * M00_pre + M00_suf * dB11 * M10_pre
        dM10 = M10_suf * dB00 * M00_pre + M10_suf * dB10 * M10_pre
        dM11 = M10_suf * dB01 * M00_pre + M10_suf * dB11 * M10_pre
        
        # dY_in = (dM10 + dM11*Y_out)*denom - numer*(dM00 + dM01*Y_out) / denom^2
        dY_in = ((dM10 + dM11 * Y_out) * denom - numer * (dM00 + dM01 * Y_out)) / (denom ** 2)
        
        # dr/dx_i = dr/dY_in * dY_in
        dr_dx = dr_dY * dY_in
        
        # dR/dx_i = 2*Re(r* * dr/dx_i)
        dR_dx = 2.0 * np.real(np.conj(r) * dr_dx)
        
        grad[i] = np.mean(dR_dx) * 100.0
    
    return obj, grad

def _objective_only(x, indices, wavelengths_nm, ambient_idx, substrate_idx,
                     max_layer_nm, max_total_nm):
    """Objective function for optimizers that don't need gradient."""
    x_clipped = np.clip(np.asarray(x, dtype=float), 0.0, max_layer_nm)
    total = np.sum(x_clipped)
    if total > max_total_nm:
        x_clipped = x_clipped * (max_total_nm / total)
    
    R, _ = _compute_reflectance(wavelengths_nm, indices, x_clipped, 
                                  ambient_idx, substrate_idx)
    return np.mean(R) * 100.0

def _objective_with_penalty(x, indices, wavelengths_nm, ambient_idx, substrate_idx,
                             max_layer_nm, max_total_nm):
    """Objective with penalty for constraint violations."""
    x_clipped = np.clip(np.asarray(x, dtype=float), 0.0, max_layer_nm)
    total = np.sum(x_clipped)
    if total > max_total_nm:
        return 100.0 + (total - max_total_nm) * 0.1
    
    R, _ = _compute_reflectance(wavelengths_nm, indices, x_clipped, 
                                  ambient_idx, substrate_idx)
    return np.mean(R) * 100.0

def _generate_initial_designs(indices, n_layers, center_wl, max_total_nm, max_layer_nm, n_starts=20):
    """
    Generate diverse, physically-motivated initial designs.
    """
    designs = []
    
    # Quarter-wave stack
    qw = np.array([center_wl / (4.0 * n_i) for n_i in indices])
    if np.sum(qw) <= max_total_nm:
        designs.append(qw.copy())
    else:
        designs.append(qw * (max_total_nm / np.sum(qw)))
    
    # Half-wave stack
    hw = np.array([center_wl / (2.0 * n_i) for n_i in indices])
    if np.sum(hw) <= max_total_nm:
        designs.append(hw.copy())
    else:
        designs.append(hw * (max_total_nm / np.sum(hw)))
    
    # Various fractions of quarter-wave
    for frac in [0.5, 0.75, 1.25, 1.5, 2.0]:
        trial = np.array([frac * center_wl / (4.0 * n_i) for n_i in indices])
        if np.sum(trial) <= max_total_nm:
            designs.append(trial)
        else:
            designs.append(trial * (max_total_nm / np.sum(trial)))
    
    # Random designs with physical constraints
    rng = np.random.RandomState(42)
    for _ in range(max(0, n_starts - len(designs))):
        trial = rng.uniform(0.1, 1.0, size=n_layers)
        trial = trial * (center_wl / 4.0) / np.array(indices)
        if np.sum(trial) > max_total_nm:
            trial = trial * (max_total_nm / np.sum(trial))
        trial = np.clip(trial, 0, max_layer_nm)
        designs.append(trial)
    
    # Ensure all are valid
    valid_designs = []
    for d in designs:
        d = np.clip(d, 0, max_layer_nm)
        if np.sum(d) <= max_total_nm:
            valid_designs.append(d)
    
    if not valid_designs:
        valid_designs.append(np.array([center_wl / (4.0 * indices[0])] + [0.0] * (n_layers - 1)))
    
    return valid_designs

def design(problem):
    """
    Design a broadband antireflection coating.
    Returns: (labels, thicknesses)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wavelengths_nm = np.linspace(band_nm[0], band_nm[1], 401)
    center_wl = 0.5 * (band_nm[0] + band_nm[1])
    
    best_labels = None
    best_thicknesses = None
    best_score = np.inf
    
    # Try different layer configurations
    layer_configs = [
        (['L' if i % 2 == 0 else 'H' for i in range(n)], n) 
        for n in [5, 7, 9, 11, 13] if n <= max_layers
    ]
    layer_configs += [
        (['H' if i % 2 == 0 else 'L' for i in range(n)], n) 
        for n in [5, 7, 9, 11, 13] if n <= max_layers
    ]
    
    # Also try non-alternating patterns for more diversity
    extra_configs = [
        ['L', 'L', 'H', 'H', 'L', 'L', 'H', 'H', 'L', 'L', 'H'],
        ['H', 'H', 'L', 'L', 'H', 'H', 'L', 'L', 'H', 'H', 'L'],
        ['L', 'H', 'H', 'L', 'L', 'H', 'H', 'L', 'L', 'H', 'H'],
    ]
    for cfg in extra_configs:
        if len(cfg) <= max_layers:
            layer_configs.append((cfg, len(cfg)))
    
    for labels, n_layers in layer_configs:
        if time.time() > deadline - 15:
            break
        
        indices = [materials[label] for label in labels]
        bounds = [(0.0, float(max_layer_nm))] * n_layers
        
        # Generate initial designs
        initial_designs = _generate_initial_designs(indices, n_layers, center_wl, 
                                                     max_total_nm, max_layer_nm, n_starts=8)
        
        for x0 in initial_designs:
            if time.time() > deadline - 10:
                break
            
            x0 = np.clip(x0, 0, max_layer_nm)
            if np.sum(x0) > max_total_nm:
                x0 = x0 * (max_total_nm / np.sum(x0))
            
            # Define objective and gradient for L-BFGS-B
            def obj(x):
                return _objective_with_penalty(x, indices, wavelengths_nm, ambient_idx, 
                                                substrate_idx, max_layer_nm, max_total_nm)
            
            def grad(x):
                _, g = _objective_and_gradient(x, indices, wavelengths_nm, ambient_idx, 
                                                substrate_idx, max_layer_nm, max_total_nm)
                return g
            
            # Run L-BFGS-B
            try:
                result = minimize(obj, x0, jac=grad, method='L-BFGS-B', 
                                   bounds=bounds, options={'maxiter': 200, 
                                                           'ftol': 1e-10,
                                                           'gtol': 1e-6,
                                                           'maxls': 50})
                
                if result.success or result.fun < best_score:
                    x_opt = np.clip(result.x, 0, max_layer_nm)
                    if np.sum(x_opt) > max_total_nm:
                        x_opt = x_opt * (max_total_nm / np.sum(x_opt))
                    
                    score = _objective_only(x_opt, indices, wavelengths_nm, ambient_idx, 
                                             substrate_idx, max_layer_nm, max_total_nm)
                    
                    if score < best_score:
                        best_score = score
                        best_labels = labels.copy()
                        best_thicknesses = x_opt.copy()
                        
            except Exception:
                pass
    
    # Fallback if no valid solution found
    if best_labels is None:
        best_labels = ['L']
        best_thicknesses = np.array([center_wl / (4.0 * materials['L'])])
    
    # Final constraint check
    thicknesses = np.array(best_thicknesses)
    thicknesses = np.clip(thicknesses, 0, max_layer_nm)
    if np.sum(thicknesses) > max_total_nm:
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
    
    return best_labels, thicknesses.tolist()
