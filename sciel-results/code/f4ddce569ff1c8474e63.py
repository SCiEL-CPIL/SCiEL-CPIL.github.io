"""AR coating design using continuous optimization of layer thicknesses."""
import numpy as np
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START

def _char_matrix_method(wavelengths_nm, indices, thicknesses_nm, ambient_index, substrate_index):
    """
    Compute reflectance using the characteristic matrix method.
    """
    n = len(indices)
    if n == 0:
        # Bare substrate
        r = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return np.full_like(wavelengths_nm, abs(r)**2)
    
    two_pi_over_lambda = 2.0 * np.pi / wavelengths_nm[:, np.newaxis]
    delta = two_pi_over_lambda * indices[np.newaxis, :] * thicknesses_nm[np.newaxis, :]
    
    m11 = np.ones_like(wavelengths_nm, dtype=np.complex128)
    m12 = np.zeros_like(wavelengths_nm, dtype=np.complex128)
    m21 = np.zeros_like(wavelengths_nm, dtype=np.complex128)
    m22 = np.ones_like(wavelengths_nm, dtype=np.complex128)
    
    for j in range(n):
        cos_d = np.cos(delta[:, j])
        sin_d = np.sin(delta[:, j])
        n_j = indices[j]
        
        a = cos_d
        b = 1j * sin_d / n_j
        c = 1j * n_j * sin_d
        d = cos_d
        
        new_m11 = m11 * a + m12 * c
        new_m12 = m11 * b + m12 * d
        new_m21 = m21 * a + m22 * c
        new_m22 = m21 * b + m22 * d
        
        m11 = new_m11
        m12 = new_m12
        m21 = new_m21
        m22 = new_m22
    
    n_s = substrate_index
    Y = (m21 + n_s * m22) / (m11 + n_s * m12)
    
    n_0 = ambient_index
    r = (n_0 - Y) / (n_0 + Y)
    reflectance = np.abs(r)**2
    
    return reflectance


def _mean_reflectance_percent(wavelengths_nm, indices, thicknesses_nm, ambient_index, substrate_index):
    """Compute mean reflectance in percent."""
    refl = _char_matrix_method(wavelengths_nm, indices, thicknesses_nm, ambient_index, substrate_index)
    return 100.0 * np.mean(refl)


def _is_valid(indices, thicknesses_nm, problem):
    """Check if the design is valid."""
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    if len(indices) > max_layers:
        return False
    if len(indices) != len(thicknesses_nm):
        return False
    if np.any(thicknesses_nm < -1e-9) or np.any(thicknesses_nm > max_layer_nm + 1e-9):
        return False
    if np.sum(thicknesses_nm) > max_total_nm + 1e-9:
        return False
    return True


def design(problem):
    """Design an AR coating using continuous optimization of thicknesses."""
    start_time = time.time()
    deadline = start_time + 140.0
    
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    wavelengths_nm = np.linspace(band_nm[0], band_nm[1], 401)
    
    n_L = materials["L"]
    n_H = materials["H"]
    
    best_result = None
    best_score = float('inf')
    
    # Try odd numbers of layers
    for n_layers in range(1, min(max_layers, 20) + 1, 2):
        if time.time() > deadline:
            break
        
        # Generate index sequences
        sequences = []
        
        # Alternating starting with L
        seq1 = ["L" if i % 2 == 0 else "H" for i in range(n_layers)]
        sequences.append(seq1)
        
        # Alternating starting with H
        seq2 = ["H" if i % 2 == 0 else "L" for i in range(n_layers)]
        sequences.append(seq2)
        
        for indices_labels in sequences:
            if time.time() > deadline:
                break
            
            indices = [materials[lab] for lab in indices_labels]
            
            # Initial guess: quarter-wave at center
            centre = 0.5 * (band_nm[0] + band_nm[1])
            x0 = np.array([centre / (4.0 * n) for n in indices])
            
            bounds = [(0.0, max_layer_nm) for _ in range(n_layers)]
            
            def obj(x, indices=indices):
                if not _is_valid(indices, x, problem):
                    return 100.0
                return _mean_reflectance_percent(wavelengths_nm, indices, x,
                                                  ambient_index, substrate_index)
            
            try:
                result = minimize(obj, x0, method='L-BFGS-B', bounds=bounds,
                                  options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
            except Exception:
                continue
            
            if result.fun < best_score and _is_valid(indices, result.x, problem):
                best_score = result.fun
                best_result = (indices_labels, result.x.copy())
            
            # Check initial guess
            score0 = obj(x0)
            if score0 < best_score and _is_valid(indices, x0, problem):
                best_score = score0
                best_result = (indices_labels, x0.copy())
    
    if best_result is None:
        centre = 0.5 * (band_nm[0] + band_nm[1])
        return ["L"], [centre / (4.0 * n_L)]
    
    labels = best_result[0]
    thicknesses = best_result[1]
    
    thicknesses = np.clip(thicknesses, 0.0, max_layer_nm)
    
    final_labels = []
    final_thicknesses = []
    for i in range(len(labels)):
        if thicknesses[i] > 1e-6:
            final_labels.append(labels[i])
            final_thicknesses.append(float(thicknesses[i]))
    
    if len(final_labels) == 0:
        centre = 0.5 * (band_nm[0] + band_nm[1])
        return ["L"], [centre / (4.0 * n_L)]
    
    if np.sum(final_thicknesses) > max_total_nm:
        scale = max_total_nm / np.sum(final_thicknesses)
        final_thicknesses = [t * scale for t in final_thicknesses]
    
    return final_labels, final_thicknesses

# EVOLVE-BLOCK-END
