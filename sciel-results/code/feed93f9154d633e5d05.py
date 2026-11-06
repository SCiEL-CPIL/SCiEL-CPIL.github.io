"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _reflectance_spectrum(thicknesses, materials, ambient_index, substrate_index, band_nm):
    """Compute reflectance spectrum using characteristic matrix method.
    
    Args:
        thicknesses: list/array of physical thicknesses in nm
        materials: dict with 'L' and 'H' indices
        ambient_index: float
        substrate_index: float
        band_nm: [min_wavelength, max_wavelength]
    
    Returns:
        array of reflectance values (0 to 1) for 401 equally spaced wavelengths
    """
    if len(thicknesses) == 0:
        # Bare substrate reflectance
        r0 = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return np.full(401, r0 * r0)
    
    n_L = materials['L']
    n_H = materials['H']
    
    # Build the stack: ambient -> layer 1 -> ... -> layer N -> substrate
    # Each layer has a refractive index and thickness
    # We need to map thicknesses to indices. The problem says we return labels and thicknesses.
    # For this internal function, we assume thicknesses is paired with labels externally.
    # Let's make a more general helper.
    
    # Actually, let's just inline the computation for speed in the optimizer.
    # We'll create a function that takes labels and thicknesses.
    pass


def _compute_reflectance(labels, thicknesses, materials, ambient_index, substrate_index, band_nm):
    """Compute mean reflectance in percent over the band."""
    if len(labels) == 0:
        r0 = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return r0 * r0 * 100.0
    
    n_L = materials['L']
    n_H = materials['H']
    
    # Get 401 equally spaced wavelengths
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    # Build index stack: ambient, layers..., substrate
    # n_stack[0] = ambient, n_stack[-1] = substrate
    # d_stack[i] = thickness of layer i (for i in 1..N)
    
    n_layers = len(labels)
    # n_indices: array of length n_layers+2
    # n_indices[0] = ambient
    # n_indices[1..n_layers] = layer indices
    # n_indices[n_layers+1] = substrate
    
    n_indices = np.empty(n_layers + 2, dtype=float)
    n_indices[0] = ambient_index
    n_indices[-1] = substrate_index
    for i, lab in enumerate(labels):
        if lab == 'L':
            n_indices[i + 1] = n_L
        else:
            n_indices[i + 1] = n_H
    
    d = np.array(thicknesses, dtype=float)
    
    # Compute reflectance for each wavelength
    # R = |(1 - Y)/(1 + Y)|^2 where Y is the admittance
    # Using characteristic matrix:
    # [M] = prod over layers of [[cos(delta), i*sin(delta)/n], [i*n*sin(delta), cos(delta)]]
    # Y = (m11*n_sub + m12*n_sub*n_sub) / (m21*n_sub + m22*n_sub) ... wait, let me get the formula right.
    
    # Standard formula:
    # For a stack with indices n_1, ..., n_N between ambient n_0 and substrate n_{N+1}:
    # The characteristic matrix M = M_1 * M_2 * ... * M_N
    # where M_j = [[cos(delta_j), i*sin(delta_j)/n_j], [i*n_j*sin(delta_j), cos(delta_j)]]
    # delta_j = 2*pi*n_j*d_j / lambda
    # Then the admittance Y = (n_{N+1} * M[1,1] + n_{N+1} * M[1,0]) / (n_{N+1} * M[0,1] + n_{N+1} * M[0,0])
    # Wait, let me look up the exact formula.
    
    # Actually, the standard formula is:
    # Y = (m11 + m12 * n_s) / (m21 + m22 * n_s)
    # where m11, m12, m21, m22 are elements of the total characteristic matrix M.
    # And R = |(n_0 - Y)/(n_0 + Y)|^2
    
    # Let me verify: for a single layer:
    # M = [[cos(d), i*sin(d)/n], [i*n*sin(d), cos(d)]]
    # Y = (cos(d) + i*sin(d)/n * n_s) / (i*n*sin(d) + cos(d) * n_s)
    # Hmm, that doesn't look right either.
    
    # Let me use the transfer matrix method properly.
    # The transfer matrix relates the field at the input to the field at the output.
    # For normal incidence:
    # [E, H]^T at layer 1 input = M * [E, H]^T at layer N output
    # The admittance Y = H/E at the input.
    # Y = (m11 * n_s + m12 * n_s * n_s) / (m21 * n_s + m22 * n_s)
    # No wait.
    
    # Let me just use the standard result:
    # Y = (m11 + m12 * n_s) / (m21 + m22 * n_s)
    # where M = [[m11, m12], [m21, m22]] is the product of individual layer matrices.
    # And each layer matrix is:
    # M_j = [[cos(delta_j), i*sin(delta_j)/n_j], [i*n_j*sin(delta_j), cos(delta_j)]]
    # delta_j = 2*pi*n_j*d_j / lambda
    # n_s is the substrate index.
    # Then R = |(n_0 - Y)/(n_0 + Y)|^2
    
    # Let me test with no layers: M = identity, Y = (1 + 0) / (0 + 1) = 1. R = |(n_0 - 1)/(n_0 + 1)|^2. That's not right.
    # The issue is that for no layers, the admittance should be n_s, not 1.
    
    # I think the correct formula involves the substrate index differently.
    # Let me reconsider. The characteristic matrix M transforms the field at the back of the stack to the front.
    # At the back, we have only a forward wave with amplitude E_s, so H_s = n_s * E_s.
    # At the front, E = m11 * E_s + m12 * n_s * E_s = (m11 + m12 * n_s) * E_s
    # H = m21 * E_s + m22 * n_s * E_s = (m21 + m22 * n_s) * E_s
    # Y = H/E = (m21 + m22 * n_s) / (m11 + m12 * n_s)
    # R = |(n_0 - Y)/(n_0 + Y)|^2
    
    # Let me test with no layers: M = I, Y = (0 + 1*n_s) / (1 + 0) = n_s. R = |(n_0 - n_s)/(n_0 + n_s)|^2. Correct!
    
    # For a single layer:
    # M = [[cos(d), i*sin(d)/n], [i*n*sin(d), cos(d)]]
    # Y = (i*n*sin(d) + cos(d)*n_s) / (cos(d) + i*sin(d)/n * n_s)
    # This looks right.
    
    reflectances = np.zeros(401, dtype=float)
    for idx, lam in enumerate(wavelengths):
        # Compute delta for each layer
        # delta_j = 2*pi*n_j*d_j / lambda
        # n_j is n_indices[1..n_layers], d_j is d[j-1]
        
        # Build the total matrix M
        # Start with identity
        m11 = 1.0 + 0.0j
        m12 = 0.0 + 0.0j
        m21 = 0.0 + 0.0j
        m22 = 1.0 + 0.0j
        
        for j in range(n_layers):
            n_j = n_indices[j + 1]
            d_j = d[j]
            delta_j = 2.0 * np.pi * n_j * d_j / lam
            cos_d = np.cos(delta_j)
            sin_d = np.sin(delta_j)
            
            # Layer matrix
            lm11 = cos_d
            lm12 = 1j * sin_d / n_j
            lm21 = 1j * n_j * sin_d
            lm22 = cos_d
            
            # M = M * layer_matrix
            new_m11 = m11 * lm11 + m12 * lm21
            new_m12 = m11 * lm12 + m12 * lm22
            new_m21 = m21 * lm11 + m22 * lm21
            new_m22 = m21 * lm12 + m22 * lm22
            
            m11 = new_m11
            m12 = new_m12
            m21 = new_m21
            m22 = new_m22
        
        # Compute admittance
        Y = (m21 + m22 * substrate_index) / (m11 + m12 * substrate_index)
        
        # Compute reflectance
        r = (ambient_index - Y) / (ambient_index + Y)
        reflectances[idx] = np.abs(r) ** 2
    
    return np.mean(reflectances) * 100.0


def design(problem):
    """Design an antireflection coating stack."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    materials = problem["materials"]
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    n_L = materials['L']
    n_H = materials['H']
    
    # Initial design: quarter-wave stack centered at 600 nm
    centre = 0.5 * (band_nm[0] + band_nm[1])
    
    # Try a few initial designs
    initial_designs = []
    
    # Design 1: Single quarter-wave L layer
    initial_designs.append((["L"], [centre / (4.0 * n_L)]))
    
    # Design 2: Two-layer quarter-wave stack (L, H)
    d_L = centre / (4.0 * n_L)
    d_H = centre / (4.0 * n_H)
    initial_designs.append((["L", "H"], [d_L, d_H]))
    
    # Design 3: Three-layer quarter-wave stack (H, L, H)
    initial_designs.append((["H", "L", "H"], [d_H, d_L, d_H]))
    
    # Design 4: Five-layer quarter-wave stack (H, L, H, L, H)
    initial_designs.append((["H", "L", "H", "L", "H"], [d_H, d_L, d_H, d_L, d_H]))
    
    # Design 5: Quarter-wave stack with optimized center wavelength
    # We'll optimize the center wavelength for a 3-layer stack
    
    best_labels = None
    best_thicknesses = None
    best_score = float('inf')
    
    # Evaluate initial designs
    for labels, thicknesses in initial_designs:
        if sum(thicknesses) > max_total_nm:
            continue
        if any(t < 0 or t > max_layer_nm for t in thicknesses):
            continue
        score = _compute_reflectance(labels, thicknesses, materials, ambient_index, substrate_index, band_nm)
        if score < best_score:
            best_score = score
            best_labels = labels[:]
            best_thicknesses = thicknesses[:]
    
    # Now use optimization to improve the best design
    # We'll optimize the thicknesses for a given set of labels
    # Start with the best initial design and try to optimize
    
    # Also try different numbers of layers
    for n_layers in range(1, min(max_layers + 1, 7)):
        if time.time() > deadline - 5.0:
            break
        
        # Try alternating starting with L and H
        for start_mat in ['L', 'H']:
            if time.time() > deadline - 5.0:
                break
            
            # Create initial labels
            labels = []
            thicknesses = []
            for i in range(n_layers):
                mat = start_mat if i % 2 == 0 else ('H' if start_mat == 'L' else 'L')
                labels.append(mat)
                d = centre / (4.0 * (n_L if mat == 'L' else n_H))
                thicknesses.append(d)
            
            # Check constraints
            if sum(thicknesses) > max_total_nm:
                # Scale down
                scale = max_total_nm / sum(thicknesses)
                thicknesses = [t * scale for t in thicknesses]
            
            # Optimize thicknesses
            def objective(x):
                if time.time() > deadline - 2.0:
                    return 100.0
                x = np.clip(x, 0, max_layer_nm)
                if np.sum(x) > max_total_nm:
                    return 100.0
                return _compute_reflectance(labels, list(x), materials, ambient_index, substrate_index, band_nm)
            
            x0 = np.array(thicknesses, dtype=float)
            
            try:
                result = minimize(objective, x0, method='L-BFGS-B', 
                                  bounds=[(0, max_layer_nm)] * n_layers,
                                  options={'maxiter': 50, 'maxfun': 100})
                if result.success and result.fun < best_score:
                    x_opt = np.clip(result.x, 0, max_layer_nm)
                    if np.sum(x_opt) <= max_total_nm:
                        best_score = result.fun
                        best_labels = labels[:]
                        best_thicknesses = list(x_opt)
            except Exception:
                pass
    
    # Try a more systematic approach: optimize a fixed pattern with more layers
    # Use a 5-layer or 7-layer quarter-wave stack and optimize all thicknesses
    
    for n_layers in [3, 5, 7]:
        if n_layers > max_layers:
            break
        if time.time() > deadline - 10.0:
            break
        
        for start_mat in ['L', 'H']:
            if time.time() > deadline - 10.0:
                break
            
            labels = []
            thicknesses = []
            for i in range(n_layers):
                mat = start_mat if i % 2 == 0 else ('H' if start_mat == 'L' else 'H')
                labels.append(mat)
                d = centre / (4.0 * (n_L if mat == 'L' else n_H))
                thicknesses.append(d)
            
            if sum(thicknesses) > max_total_nm:
                scale = max_total_nm / sum(thicknesses)
                thicknesses = [t * scale for t in thicknesses]
            
            def objective(x):
                if time.time() > deadline - 2.0:
                    return 100.0
                x = np.clip(x, 0, max_layer_nm)
                if np.sum(x) > max_total_nm:
                    return 100.0
                return _compute_reflectance(labels, list(x), materials, ambient_index, substrate_index, band_nm)
            
            x0 = np.array(thicknesses, dtype=float)
            
            try:
                result = minimize(objective, x0, method='Nelder-Mead',
                                  options={'maxiter': 200, 'xatol': 1e-3, 'fatol': 1e-6})
                if result.fun < best_score:
                    x_opt = np.clip(result.x, 0, max_layer_nm)
                    if np.sum(x_opt) <= max_total_nm:
                        best_score = result.fun
                        best_labels = labels[:]
                        best_thicknesses = list(x_opt)
            except Exception:
                pass
    
    # If we didn't find anything better, return the initial best
    if best_labels is None:
        best_labels = ["L"]
        best_thicknesses = [centre / (4.0 * n_L)]
    
    # Final check
    if sum(best_thicknesses) > max_total_nm:
        scale = max_total_nm / sum(best_thicknesses)
        best_thicknesses = [t * scale for t in best_thicknesses]
    
    for i, t in enumerate(best_thicknesses):
        if t < 0:
            best_thicknesses[i] = 0
        if t > max_layer_nm:
            best_thicknesses[i] = max_layer_nm
    
    return best_labels, best_thicknesses

# EVOLVE-BLOCK-END
