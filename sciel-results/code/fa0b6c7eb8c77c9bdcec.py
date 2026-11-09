"""Design a broadband antireflection coating stack."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START

def _reflectance(materials, thicknesses, n_amb, n_sub, n_L, n_H, wavelengths):
    """Compute normal-incidence reflectance (fraction) using characteristic matrix method."""
    # wavelengths: array of wavelengths in nm
    # returns array of reflectance values
    k = 2 * np.pi / wavelengths  # wavenumber in 1/nm
    # phase thickness for each layer
    # materials: list of 'L' or 'H'
    # thicknesses: list of thicknesses in nm
    
    n_layers = len(materials)
    n_wl = len(wavelengths)
    
    # Build matrices for each wavelength
    # For efficiency, we compute for all wavelengths at once using vectorized ops
    # But matrix multiplication is per-wavelength, so we loop over wavelengths
    # Actually, let's vectorize: for each wavelength, build 2x2 matrices and multiply
    
    reflectances = np.zeros(n_wl)
    
    for i in range(n_wl):
        ki = k[i]
        # Start with identity
        M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        
        for j in range(n_layers):
            n_j = n_L if materials[j] == 'L' else n_H
            d_j = thicknesses[j]
            # Phase thickness: delta = (2*pi/lambda) * n * d
            delta = ki * n_j * d_j
            c = math.cos(delta)
            s = math.sin(delta)
            # Layer matrix: [[n_j*c, i*n_j*s], [i*s/n_j, c]]
            # Wait, the standard characteristic matrix for a layer of index n and thickness d:
            # M_layer = [[n*cos(delta), i*n*sin(delta)], [i*sin(delta)/n, cos(delta)]]
            # where delta = (2*pi/lambda)*n*d
            layer_M = np.array([[n_j * c, 1j * n_j * s],
                                [1j * s / n_j, c]], dtype=complex)
            M = M @ layer_M
        
        # Total matrix: M_total = M_amb @ M_layer1 @ ... @ M_layerN
        # Actually, the total characteristic matrix is:
        # M_total = M_1 @ M_2 @ ... @ M_N
        # where M_j is the j-th layer matrix
        # Then:
        # B = M[0,0], C = M[0,1], D = M[1,0], E = M[1,1]
        # For incidence from air (index n_amb) to stack on substrate (index n_sub):
        # Y = (n_amb * B) / (n_sub * C) ... no wait
        # Actually:
        # The admittance Y = (n_amb * B) / C  ... no
        # Let me recall:
        # For a stack with total matrix [[B, C], [D, E]],
        # the input admittance Y = (n_amb * B) / C  ... no
        # Standard formula:
        # Y = (n_amb * B) / C  is wrong
        # Correct: 
        # The characteristic matrix of the whole stack is M = [[B, C], [D, E]]
        # The input admittance (relative to ambient) is:
        # Y = (n_amb * B) / C  ... no
        # Let me be precise:
        # For normal incidence, the characteristic matrix of layer j is:
        # M_j = [[cos(delta_j), i*sin(delta_j)/n_j], [i*n_j*sin(delta_j), cos(delta_j)]]
        # Wait, there are two conventions. Let me use the one from Hecht or Born & Wolf.
        # Convention: M_j = [[n_j*cos(delta_j), i*n_j*sin(delta_j)], [i*sin(delta_j)/n_j, cos(delta_j)]]
        # Total matrix: M = M_1 @ M_2 @ ... @ M_N
        # Then B = M[0,0], C = M[0,1], D = M[1,0], E = M[1,1]
        # The effective admittance Y = (n_amb * B) / C  ... no
        # Actually:
        # Y = (n_amb * B) / C  is not right.
        # The correct formula for reflectance:
        # r = (Y - n_amb) / (Y + n_amb)  where Y is the input admittance
        # And Y = (n_amb * B) / C  ... no
        # Let me look it up properly:
        # The total characteristic matrix M = [[B, C], [D, E]]
        # The input admittance is Y = (n_amb * B) / C  ... no
        # Actually, for the stack between ambient and substrate:
        # The boundary condition gives:
        # E_inc + E_ref = E_trans_at_input
        # H_inc - H_ref = H_trans_at_input
        # With H = n * E for normal incidence (in normalized units)
        # The characteristic matrix relates fields:
        # [E_inc]   [B  C] [E_sub]
        # [H_inc] = [D  E] [H_sub]
        # where H = n*E for the substrate
        # So: E_inc = B * E_sub + C * H_sub = B * E_sub + C * n_sub * E_sub = (B + C*n_sub) * E_sub
        # H_inc = D * E_sub + E * H_sub = D * E_sub + E * n_sub * E_sub = (D + E*n_sub) * E_sub
        # The input admittance Y = H_inc / E_inc = (D + E*n_sub) / (B + C*n_sub)
        # Wait, but we also need to account for the ambient.
        # Actually, the reflectance is:
        # r = (Y - n_amb) / (Y + n_amb)
        # where Y = (D + E*n_sub) / (B + C*n_sub)
        # Hmm, let me double-check with a single layer.
        # For a single layer of index n_1, thickness d_1, on substrate n_s, in ambient n_0:
        # M = [[n_1*cos(delta), i*n_1*sin(delta)], [i*sin(delta)/n_1, cos(delta)]]
        # B = n_1*cos(delta), C = i*n_1*sin(delta), D = i*sin(delta)/n_1, E = cos(delta)
        # Y = (D + E*n_s) / (B + C*n_s) = (i*sin(delta)/n_1 + cos(delta)*n_s) / (n_1*cos(delta) + i*n_1*sin(delta)*n_s)
        # r = (Y - n_0) / (Y + n_0)
        # This should give the correct result.
        
        B = M[0, 0]
        C = M[0, 1]
        D = M[1, 0]
        E = M[1, 1]
        
        Y = (D + E * n_sub) / (B + C * n_sub)
        r = (Y - n_amb) / (Y + n_amb)
        reflectances[i] = np.abs(r) ** 2
    
    return reflectances


def _mean_reflectance_pct(materials, thicknesses, n_amb, n_sub, n_L, n_H, band_nm):
    """Compute mean reflectance in percent over the band."""
    n_wl = 401
    wavelengths = np.linspace(band_nm[0], band_nm[1], n_wl)
    refl = _reflectance(materials, thicknesses, n_amb, n_sub, n_L, n_H, wavelengths)
    return 100.0 * np.mean(refl)


def _valid(materials, thicknesses, max_layers, max_total_nm, max_layer_nm):
    """Check if the design is valid."""
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > max_layers:
        return False
    for t in thicknesses:
        if t < 0 or t > max_layer_nm:
            return False
    if sum(thicknesses) > max_total_nm:
        return False
    for m in materials:
        if m not in ('L', 'H'):
            return False
    return True


def design(problem):
    """Design a broadband antireflection coating stack."""
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    centre_wl = 0.5 * (band_nm[0] + band_nm[1])
    
    # Initialize with a reasonable starting point
    # Multi-layer quarter-wave stack
    best_materials = None
    best_thicknesses = None
    best_score = float('inf')
    
    # Try a few initial designs
    initial_designs = []
    
    # Single quarter-wave layer of low index
    d_L = centre_wl / (4.0 * n_L)
    initial_designs.append((["L"], [d_L]))
    
    # Single quarter-wave layer of high index
    d_H = centre_wl / (4.0 * n_H)
    initial_designs.append((["H"], [d_H]))
    
    # Two-layer: L-H quarter-wave
    initial_designs.append((["L", "H"], [d_L, d_H]))
    initial_designs.append((["H", "L"], [d_H, d_L]))
    
    # Three-layer: L-H-L quarter-wave
    initial_designs.append((["L", "H", "L"], [d_L, d_H, d_L]))
    initial_designs.append((["H", "L", "H"], [d_H, d_L, d_H]))
    
    # Five-layer: L-H-L-H-L quarter-wave
    initial_designs.append((["L", "H", "L", "H", "L"], [d_L, d_H, d_L, d_H, d_L]))
    initial_designs.append((["H", "L", "H", "L", "H"], [d_H, d_L, d_H, d_L, d_H]))
    
    # Nine-layer: L-H-L-H-L-H-L-H-L quarter-wave
    initial_designs.append((["L", "H", "L", "H", "L", "H", "L", "H", "L"], 
                            [d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L]))
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H", "L", "H"],
                            [d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H]))
    
    # Thirteen-layer
    for pattern in [["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"],
                    ["H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H"]]:
        thicknesses = [centre_wl / (4.0 * (n_L if m == 'L' else n_H)) for m in pattern]
        if sum(thicknesses) <= max_total_nm:
            initial_designs.append((pattern, thicknesses))
    
    # Seventeen-layer
    for pattern in [["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"],
                    ["H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H"]]:
        thicknesses = [centre_wl / (4.0 * (n_L if m == 'L' else n_H)) for m in pattern]
        if sum(thicknesses) <= max_total_nm:
            initial_designs.append((pattern, thicknesses))
    
    # Nineteen-layer
    for pattern in [["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"],
                    ["H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H"]]:
        thicknesses = [centre_wl / (4.0 * (n_L if m == 'L' else n_H)) for m in pattern]
        if sum(thicknesses) <= max_total_nm:
            initial_designs.append((pattern, thicknesses))
    
    # Evaluate initial designs
    for mats, thicks in initial_designs:
        if not _valid(mats, thicks, max_layers, max_total_nm, max_layer_nm):
            continue
        score = _mean_reflectance_pct(mats, thicks, n_amb, n_sub, n_L, n_H, band_nm)
        if score < best_score:
            best_score = score
            best_materials = mats[:]
            best_thicknesses = thicks[:]
    
    if best_materials is None:
        # Fallback: single layer
        d_L = centre_wl / (4.0 * n_L)
        return ["L"], [d_L]
    
    # Simulated annealing to optimize thicknesses
    # Keep the material pattern fixed, optimize thicknesses
    current_materials = best_materials[:]
    current_thicknesses = best_thicknesses[:]
    current_score = best_score
    
    n_layers = len(current_materials)
    
    # SA parameters
    T_initial = 0.5  # Initial temperature (in percent reflectance units)
    T_final = 0.001
    n_iterations = 50000
    cooling_rate = (T_final / T_initial) ** (1.0 / n_iterations)
    
    T = T_initial
    rng = random.Random(42)
    
    for it in range(n_iterations):
        if time.time() > deadline - 10.0:
            break
        
        # Propose a change: pick a random layer and perturb its thickness
        idx = rng.randint(0, n_layers - 1)
        delta = rng.gauss(0, 20.0)  # Perturbation in nm
        new_thickness = current_thicknesses[idx] + delta
        new_thickness = max(0.0, min(max_layer_nm, new_thickness))
        
        # Check total thickness constraint
        new_total = sum(current_thicknesses) - current_thicknesses[idx] + new_thickness
        if new_total > max_total_nm:
            continue
        
        old_thickness = current_thicknesses[idx]
        current_thicknesses[idx] = new_thickness
        
        new_score = _mean_reflectance_pct(current_materials, current_thicknesses, n_amb, n_sub, n_L, n_H, band_nm)
        
        delta_score = new_score - current_score
        if delta_score < 0 or rng.random() < math.exp(-delta_score / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_materials = current_materials[:]
                best_thicknesses = current_thicknesses[:]
        else:
            current_thicknesses[idx] = old_thickness
        
        T *= cooling_rate
    
    # Nelder-Mead polish
    if time.time() < deadline - 5.0 and n_layers > 0:
        try:
            from scipy.optimize import minimize
            
            def objective(thicks):
                thicks = np.clip(thicks, 0, max_layer_nm)
                if np.sum(thicks) > max_total_nm:
                    return 100.0  # Penalty
                return _mean_reflectance_pct(current_materials, list(thicks), n_amb, n_sub, n_L, n_H, band_nm)
            
            x0 = np.array(current_thicknesses, dtype=float)
            bounds = [(0, max_layer_nm)] * n_layers
            
            result = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 10000, 'xatol': 0.1, 'fatol': 0.001})
            
            if result.success or result.fun < current_score:
                new_thicks = np.clip(result.x, 0, max_layer_nm)
                if np.sum(new_thicks) <= max_total_nm:
                    new_score = objective(new_thicks)
                    if new_score < best_score:
                        best_score = new_score
                        best_materials = current_materials[:]
                        best_thicknesses = list(new_thicks)
        except Exception:
            pass
    
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
