"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def _char_matrix(n, d, wl):
    """Compute the characteristic matrix for a single layer at wavelength wl."""
    beta = 2.0 * np.pi * n * d / wl
    cos_b = np.cos(beta)
    sin_b = np.sin(beta)
    m = np.array([
        [cos_b, 1.0j * sin_b / n],
        [n * 1.0j * sin_b, cos_b]
    ])
    return m

def _mean_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance in percent over the band."""
    if len(materials) == 0:
        # Bare substrate
        n0 = problem["ambient_index"]
        ns = problem["substrate_index"]
        r = (n0 - ns) / (n0 + ns)
        return 100.0 * r * r
    
    band = problem["band_nm"]
    wavelengths = np.linspace(band[0], band[1], 401)
    mat_dict = problem["materials"]
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    
    total_ref = 0.0
    for wl in wavelengths:
        # Build total characteristic matrix
        M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        for mat, thick in zip(materials, thicknesses):
            n = mat_dict[mat]
            d = thick
            if d == 0:
                continue
            M = M @ _char_matrix(n, d, wl)
        
        # Substrate admittance
        Ys = ns  # Normal incidence, substrate admittance is just n_s
        
        # Total admittance
        Y = M[1, 1] / M[0, 1]
        
        # Reflectance
        r = (n0 - Y) / (n0 + Y)
        R = np.abs(r) ** 2
        total_ref += R * 100.0
    
    return total_ref / len(wavelengths)

def _is_valid(materials, thicknesses, problem):
    """Check if the design is valid."""
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    if len(materials) > max_layers:
        return False
    if len(materials) != len(thicknesses):
        return False
    if sum(thicknesses) > max_total_nm:
        return False
    for t in thicknesses:
        if t < 0 or t > max_layer_nm:
            return False
    return True

def design(problem):
    """Design an antireflection coating stack."""
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    # Initial guess: quarter-wave stack
    centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    dL = centre / (4.0 * problem["materials"]["L"])
    dH = centre / (4.0 * problem["materials"]["H"])
    
    # Try a few simple symmetric stacks
    candidates = []
    
    # 1-layer L
    candidates.append((["L"], [dL]))
    
    # 3-layer L-H-L
    candidates.append((["L", "H", "L"], [dL, dH, dL]))
    
    # 5-layer L-H-L-H-L
    candidates.append((["L", "H", "L", "H", "L"], [dL, dH, dL, dH, dL]))
    
    # 7-layer L-H-L-H-L-H-L
    candidates.append((["L", "H", "L", "H", "L", "H", "L"], [dL, dH, dL, dH, dL, dH, dL]))
    
    # 9-layer L-H-L-H-L-H-L-H-L
    candidates.append((["L", "H", "L", "H", "L", "H", "L", "H", "L"], 
                       [dL, dH, dL, dH, dL, dH, dL, dH, dL]))
    
    # 11-layer L-H-L-H-L-H-L-H-L-H-L
    candidates.append((["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"], 
                       [dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL]))
    
    # 13-layer L-H-L-H-L-H-L-H-L-H-L-H-L
    candidates.append((["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"], 
                       [dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL]))
    
    # 15-layer L-H-L-H-L-H-L-H-L-H-L-H-L-H-L
    candidates.append((["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"], 
                       [dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL]))
    
    # 17-layer L-H-L-H-L-H-L-H-L-H-L-H-L-H-L-H-L
    candidates.append((["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"], 
                       [dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL]))
    
    # 19-layer L-H-L-H-L-H-L-H-L-H-L-H-L-H-L-H-L-H-L
    candidates.append((["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"], 
                       [dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL]))
    
    best_score = float('inf')
    best_result = candidates[0]
    
    for mats, thicks in candidates:
        if not _is_valid(mats, thicks, problem):
            continue
        score = _mean_reflectance(mats, thicks, problem)
        if score < best_score:
            best_score = score
            best_result = (mats, thicks)
        if time.time() > deadline:
            break
    
    # Try to optimize thicknesses with a simple local search
    if time.time() < deadline - 10.0:
        mats, thicks = best_result
        # Try scaling thicknesses
        for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
            new_thicks = [t * scale for t in thicks]
            if _is_valid(mats, new_thicks, problem):
                score = _mean_reflectance(mats, new_thicks, problem)
                if score < best_score:
                    best_score = score
                    best_result = (mats, new_thicks)
            if time.time() > deadline:
                break
    
    return best_result
# EVOLVE-BLOCK-END
