"""Broadband antireflection coating design: Gradient-based (L-BFGS) optimization with multiple layer counts and random restarts."""
import time
import random
import numpy as np
from scipy.optimize import minimize

def _mean_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance (percent) over the band using characteristic matrices."""
    if not materials:
        return 100.0 * ((problem["substrate_index"] - problem["ambient_index"]) /
                        (problem["substrate_index"] + problem["ambient_index"])) ** 2
    
    band = problem["band_nm"]
    wavs = np.linspace(band[0], band[1], 401)
    n_mat = {k: v for k, v in problem["materials"].items()}
    
    ns = np.array([n_mat[m] for m in materials], dtype=float)
    ds = np.asarray(thicknesses, dtype=float)
    
    # delta_j = 2*pi*n_j*d_j/lambda  -> (N, K)
    delta = 2.0 * np.pi * (ns[:, None] * ds[:, None]) / wavs[None, :]
    c = np.cos(delta)
    s = np.sin(delta)
    
    # Initialize characteristic matrix product as identity for all wavelengths
    A = np.ones_like(wavs)
    B = np.zeros_like(wavs)
    C = np.zeros_like(wavs)
    D = np.ones_like(wavs)
    
    i = 1.0j
    for j in range(len(materials)):
        n_j = ns[j]
        cj = c[j]
        sj = s[j]
        A_new = A * cj + 1j * B * n_j * sj
        B_new = 1j * A * sj / n_j + B * cj
        C_new = C * cj + 1j * D * n_j * sj
        D_new = 1j * C * sj / n_j + D * cj
        A, B, C, D = A_new, B_new, C_new, D_new
    
    n_s = problem["substrate_index"]
    n_0 = problem["ambient_index"]
    num = n_0 * (A + B * n_s) - (C + D * n_s)
    den = n_0 * (A + B * n_s) + (C + D * n_s)
    r = num / den
    R = np.abs(r) ** 2
    return 100.0 * np.mean(R)

def design(problem):
    start_time = time.time()
    deadline = start_time + 135  # 150s kill, 15s safety margin
    
    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    materials_dict = problem["materials"]
    
    # Precompute quarter-wave thicknesses at center wavelength
    qw = {}
    for m, n in materials_dict.items():
        qw[m] = wl_center / (4.0 * n)
        
    def make_pattern(n_layers):
        """Create alternating pattern starting with L."""
        pattern = []
        for i in range(n_layers):
            if i % 2 == 0:
                pattern.append('L')
            else:
                pattern.append('H')
        return tuple(pattern)
    
    def objective(x, pattern, problem):
        """Objective function for optimization."""
        # x is the vector of thicknesses
        # Apply constraints: clip to [0, max_layer] and scale if total > max_total
        ths = np.clip(x, 0.0, max_layer)
        total = np.sum(ths)
        if total > max_total:
            ths = ths * (max_total / total)
        
        score = _mean_reflectance(pattern, ths, problem)
        return score
    
    best_score = float('inf')
    best_ths = None
    best_pattern = None
    
    # Try different layer counts: 4, 6, 8, 10, 12, 14, 16, 18, 20
    layer_counts = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    
    random.seed(42)
    
    for n_layers in layer_counts:
        if time.time() > deadline - 5:
            break
            
        pattern = make_pattern(n_layers)
        
        # Multiple random starts for this layer count
        n_starts = 5
        for start_idx in range(n_starts):
            if time.time() > deadline - 5:
                break
                
            # Generate initial guess
            if start_idx == 0:
                # Quarter-wave stack
                init = np.array([qw[m] for m in pattern], dtype=float)
            else:
                # Random perturbation around quarter-wave
                init = np.array([qw[m] * random.uniform(0.7, 1.3) for m in pattern], dtype=float)
            
            # Clip initial guess
            init = np.clip(init, 0.0, max_layer)
            total = np.sum(init)
            if total > max_total:
                init = init * (max_total / total)
            
            # Run L-BFGS-B optimization
            try:
                result = minimize(
                    objective,
                    init,
                    args=(pattern, problem),
                    method='L-BFGS-B',
                    bounds=[(0.0, max_layer)] * n_layers,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if result.success or result.fun < best_score:
                    # Apply final constraints
                    ths = np.clip(result.x, 0.0, max_layer)
                    total = np.sum(ths)
                    if total > max_total:
                        ths = ths * (max_total / total)
                    
                    score = _mean_reflectance(pattern, ths, problem)
                    if score < best_score:
                        best_score = score
                        best_ths = list(ths)
                        best_pattern = pattern
                        
            except Exception:
                continue
    
    # If no valid result, fall back to a simple quarter-wave stack
    if best_ths is None:
        pattern = make_pattern(4)
        best_ths = [qw[m] for m in pattern]
        best_pattern = pattern
    
    return list(best_pattern), list(best_ths)
