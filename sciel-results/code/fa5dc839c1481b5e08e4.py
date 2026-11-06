"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

# EVOLVE-BLOCK-START

def _compute_reflectance_vectorized(labels, thicknesses, materials, ambient_index, substrate_index, band_nm):
    """Compute mean reflectance in percent over the band using vectorized characteristic matrix method."""
    if len(labels) == 0:
        r0 = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return r0 * r0 * 100.0
    
    n_L = materials['L']
    n_H = materials['H']
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    n_layers = len(labels)
    
    # Build index array
    n_indices = np.empty(n_layers + 2, dtype=float)
    n_indices[0] = ambient_index
    n_indices[-1] = substrate_index
    for i, lab in enumerate(labels):
        n_indices[i + 1] = n_L if lab == 'L' else n_H
    
    d = np.array(thicknesses, dtype=float)
    
    # Vectorized computation over all wavelengths
    # delta[j, k] = 2*pi*n_j*d_j/lambda_k
    delta = 2.0 * np.pi * np.outer(n_indices[1:-1] * d, 1.0 / wavelengths)
    cos_d = np.cos(delta)  # shape (n_layers, 401)
    sin_d = np.sin(delta)  # shape (n_layers, 401)
    
    # Initialize matrices as arrays of shape (401,)
    m11 = np.ones(401, dtype=complex)
    m12 = np.zeros(401, dtype=complex)
    m21 = np.zeros(401, dtype=complex)
    m22 = np.ones(401, dtype=complex)
    
    for j in range(n_layers):
        n_j = n_indices[j + 1]
        lm11 = cos_d[j]
        lm12 = 1j * sin_d[j] / n_j
        lm21 = 1j * n_j * sin_d[j]
        lm22 = cos_d[j]
        
        new_m11 = m11 * lm11 + m12 * lm21
        new_m12 = m11 * lm12 + m12 * lm22
        new_m21 = m21 * lm11 + m22 * lm21
        new_m22 = m21 * lm12 + m22 * lm22
        
        m11 = new_m11
        m12 = new_m12
        m21 = new_m21
        m22 = new_m22
    
    Y = (m21 + m22 * substrate_index) / (m11 + m12 * substrate_index)
    r = (ambient_index - Y) / (ambient_index + Y)
    reflectances = np.abs(r) ** 2
    
    return np.mean(reflectances) * 100.0


def design(problem):
    """Design an antireflection coating stack using combinatorial search over quarter-wave thicknesses."""
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
    center_wl = 0.5 * (band_nm[0] + band_nm[1])
    
    # Define quarter-wave optical thicknesses for L and H at various center wavelengths
    # Optical thickness = n * d, so physical thickness d = optical_thickness / n
    # Quarter-wave optical thickness at wavelength lambda: lambda/4
    # So physical thickness for L: lambda/(4*n_L), for H: lambda/(4*n_H)
    
    # Use a range of center wavelengths to get diverse thickness options
    thickness_options = []
    for cw in range(450, 751, 25):
        t_L = cw / (4.0 * n_L)
        t_H = cw / (4.0 * n_H)
        if 0 < t_L <= max_layer_nm:
            thickness_options.append(('L', round(t_L, 3)))
        if 0 < t_H <= max_layer_nm:
            thickness_options.append(('H', round(t_H, 3)))
    
    # Deduplicate
    thickness_options = list(set(thickness_options))
    
    # Also add some half-wave and other useful thicknesses
    for cw in range(450, 751, 50):
        t_L_half = cw / (2.0 * n_L)
        t_H_half = cw / (2.0 * n_H)
        if 0 < t_L_half <= max_layer_nm:
            thickness_options.append(('L', round(t_L_half, 3)))
        if 0 < t_H_half <= max_layer_nm:
            thickness_options.append(('H', round(t_H_half, 3)))
    
    thickness_options = list(set(thickness_options))
    thickness_options.sort(key=lambda x: (x[0], x[1]))
    
    # Separate L and H thickness options
    L_thicknesses = sorted(set([t for mat, t in thickness_options if mat == 'L']))
    H_thicknesses = sorted(set([t for mat, t in thickness_options if mat == 'H']))
    
    # Ensure we have reasonable options
    if len(L_thicknesses) == 0:
        L_thicknesses = [center_wl / (4.0 * n_L)]
    if len(H_thicknesses) == 0:
        H_thicknesses = [center_wl / (4.0 * n_H)]
    
    # Fallback design
    best_labels = ["L"]
    best_thicknesses = [center_wl / (4.0 * n_L)]
    best_score = _compute_reflectance_vectorized(best_labels, best_thicknesses, materials, ambient_index, substrate_index, band_nm)
    
    # Try various layer counts and patterns
    # Start with symmetric patterns which are known to be good for broadband AR coatings
    
    def try_design(labels, thicknesses):
        nonlocal best_labels, best_thicknesses, best_score
        if len(labels) > max_layers:
            return
        if any(t < 0 or t > max_layer_nm for t in thicknesses):
            return
        if sum(thicknesses) > max_total_nm:
            return
        score = _compute_reflectance_vectorized(labels, thicknesses, materials, ambient_index, substrate_index, band_nm)
        if score < best_score:
            best_score = score
            best_labels = labels[:]
            best_thicknesses = thicknesses[:]
    
    # Try single layer
    for t in L_thicknesses:
        try_design(['L'], [t])
    
    # Try 2-layer designs
    for t1 in L_thicknesses:
        for t2 in H_thicknesses:
            if time.time() > deadline:
                return best_labels, best_thicknesses
            try_design(['L', 'H'], [t1, t2])
            try_design(['H', 'L'], [t2, t1])
    
    # Try 3-layer designs (symmetric)
    for t1 in L_thicknesses:
        for t2 in H_thicknesses:
            if time.time() > deadline:
                return best_labels, best_thicknesses
            try_design(['H', 'L', 'H'], [t2, t1, t2])
            try_design(['L', 'H', 'L'], [t1, t2, t1])
    
    # Try 4-layer designs
    for t1 in L_thicknesses:
        for t2 in H_thicknesses:
            if time.time() > deadline:
                return best_labels, best_thicknesses
            try_design(['H', 'L', 'H', 'L'], [t2, t1, t2, t1])
            try_design(['L', 'H', 'L', 'H'], [t1, t2, t1, t2])
    
    # Try 5-layer designs (symmetric)
    for t1 in L_thicknesses:
        for t2 in H_thicknesses:
            if time.time() > deadline:
                return best_labels, best_thicknesses
            try_design(['H', 'L', 'H', 'L', 'H'], [t2, t1, t2, t1, t2])
            try_design(['L', 'H', 'L', 'H', 'L'], [t1, t2, t1, t2, t1])
    
    # Try 6-layer designs
    for t1 in L_thicknesses:
        for t2 in H_thicknesses:
            if time.time() > deadline:
                return best_labels, best_thicknesses
            try_design(['H', 'L', 'H', 'L', 'H', 'L'], [t2, t1, t2, t1, t2, t1])
            try_design(['L', 'H', 'L', 'H', 'L', 'H'], [t1, t2, t1, t2, t1, t2])
    
    # Try 7-layer designs (symmetric)
    for t1 in L_thicknesses:
        for t2 in H_thicknesses:
            if time.time() > deadline:
                return best_labels, best_thicknesses
            try_design(['H', 'L', 'H', 'L', 'H', 'L', 'H'], [t2, t1, t2, t1, t2, t1, t2])
            try_design(['L', 'H', 'L', 'H', 'L', 'H', 'L'], [t1, t2, t1, t2, t1, t2, t1])
    
    # Try 8-layer designs
    for t1 in L_thicknesses:
        for t2 in H_thicknesses:
            if time.time() > deadline:
                return best_labels, best_thicknesses
            try_design(['H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'], [t2, t1, t2, t1, t2, t1, t2, t1])
            try_design(['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'], [t1, t2, t1, t2, t1, t2, t1, t2])
    
    # If we have time, try more diverse patterns
    if time.time() < deadline - 20:
        # Try alternating patterns with varying thickness combinations
        for n_layers in range(3, 11, 2):  # Odd layers: symmetric
            if n_layers > max_layers:
                break
            # Generate all combinations of L and H thicknesses
            # For symmetric stacks, we only need to specify the first half
            half = (n_layers + 1) // 2
            # Try different label patterns
            for pattern in product(['L', 'H'], repeat=half):
                if time.time() > deadline:
                    return best_labels, best_thicknesses
                # Build full symmetric pattern
                labels = list(pattern)
                if n_layers % 2 == 1:
                    labels = labels + list(reversed(pattern[:-1]))
                else:
                    labels = labels + list(reversed(pattern))
                
                if len(labels) > max_layers:
                    continue
                
                # Try a few thickness combinations
                for t1 in L_thicknesses[:min(5, len(L_thicknesses))]:
                    for t2 in H_thicknesses[:min(5, len(H_thicknesses))]:
                        if time.time() > deadline:
                            return best_labels, best_thicknesses
                        thicknesses = []
                        for lab in labels:
                            if lab == 'L':
                                thicknesses.append(t1)
                            else:
                                thicknesses.append(t2)
                        try_design(labels, thicknesses)
    
    # Final validation
    if not best_labels or len(best_labels) > max_layers:
        best_labels = ["L"]
        best_thicknesses = [center_wl / (4.0 * n_L)]
    
    if any(t < 0 or t > max_layer_nm for t in best_thicknesses):
        best_labels = ["L"]
        best_thicknesses = [center_wl / (4.0 * n_L)]
    
    if sum(best_thicknesses) > max_total_nm:
        best_labels = ["L"]
        best_thicknesses = [center_wl / (4.0 * n_L)]
    
    return best_labels, best_thicknesses

# EVOLVE-BLOCK-END
