"""Broadband antireflection coating design: Fixed graded-index profile with Nelder-Mead optimization."""
import time
import random
import numpy as np

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
    deadline = start_time + 135  # 150s kill, 15s safety
    
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
        
    # Define a fixed pattern that approximates a graded index profile.
    # Start with H (high index) to match substrate better? Or L to match air?
    # Air is 1.0, Substrate 1.52.
    # L=1.38 is closer to Air. H=2.35 is closer to Substrate (in log space? 1.38^2=1.9, 2.35^2=5.5, 1.52^2=2.3. No.
    # Geometric mean of Air(1) and Sub(1.52) is ~1.23. L(1.38) is closer.
    # Geometric mean of Sub(1.52) and L(1.38) is ~1.45.
    # A common broadband AR structure is alternating layers.
    # Let's use a pattern that starts with L (closer to air) and ends with H (closer to sub)?
    # Or just alternate.
    # A 20-layer stack: L H L H ...
    # Let's try starting with L.
    pattern_length = 20
    pattern = tuple("LH" if i % 2 == 0 else "HL" for i in range(pattern_length // 2))
    # Wait, that logic is wrong.
    pattern_list = []
    for i in range(pattern_length):
        if i % 2 == 0:
            pattern_list.append('L')
        else:
            pattern_list.append('H')
    pattern = tuple(pattern_list)
    
    # Ensure we don't exceed max layers
    if len(pattern) > max_layers:
        pattern = pattern[:max_layers]
        
    n_layers = len(pattern)
    
    def constrain_thicknesses(ths):
        total = sum(ths)
        if total > max_total:
            scale = max_total / total
            ths = [t * scale for t in ths]
        for i in range(len(ths)):
            ths[i] = max(0.0, min(max_layer, ths[i]))
        total2 = sum(ths)
        if total2 > max_total:
            scale = max_total / total2
            ths = [t * scale for t in ths]
        return ths

    def nelder_mead(init_ths, max_iters=300):
        """Nelder-Mead simplex local optimization."""
        ths = list(init_ths)
        ths = constrain_thicknesses(ths)
        best_score = _mean_reflectance(pattern, ths, problem)
        best_ths = list(ths)
        n = len(ths)
        
        if n == 0:
            return best_ths, best_score

        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Initial step size
        step = 10.0
        simplex = [list(ths)]
        for i in range(n):
            pt = list(ths)
            pt[i] = max(0.0, min(max_layer, pt[i] + step))
            simplex.append(pt)
        
        scores = [_mean_reflectance(pattern, simplex[i], problem) for i in range(n + 1)]
        
        it = 0
        while it < max_iters and time.time() < deadline:
            it += 1
            order = np.argsort(scores)
            simplex = [simplex[i] for i in order]
            scores = [scores[i] for i in order]
            
            if scores[0] < best_score:
                best_score = scores[0]
                best_ths = list(simplex[0])
                
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = constrain_thicknesses(list(xr))
            fr = _mean_reflectance(pattern, xr, problem)
            
            if fr < scores[0]:
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = constrain_thicknesses(list(xe))
                fe = _mean_reflectance(pattern, xe, problem)
                if fe < fr:
                    simplex[-1] = xe
                    scores[-1] = fe
                else:
                    simplex[-1] = xr
                    scores[-1] = fr
            elif fr < scores[-2]:
                simplex[-1] = xr
                scores[-1] = fr
            else:
                if fr < scores[-1]:
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = constrain_thicknesses(list(xc))
                    fc = _mean_reflectance(pattern, xc, problem)
                    if fc <= fr:
                        simplex[-1] = xc
                        scores[-1] = fc
                    else:
                        break
                else:
                    for i in range(1, n + 1):
                        simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                        simplex[i] = constrain_thicknesses(simplex[i])
                        scores[i] = _mean_reflectance(pattern, simplex[i], problem)
                    if np.max(scores) - np.min(scores) < 1e-7:
                        break
                        
        return best_ths, best_score

    # Multi-start optimization
    best_ths = None
    best_score = float('inf')
    
    # Start 1: Analytical quarter-wave stack
    qw_ths = [qw[m] for m in pattern]
    qw_ths = constrain_thicknesses(qw_ths)
    score_qw = _mean_reflectance(pattern, qw_ths, problem)
    if score_qw < best_score:
        best_score = score_qw
        best_ths = list(qw_ths)
        
    # Start 2-10: Random perturbations of quarter-wave
    random.seed(42)
    for _ in range(10):
        if time.time() > deadline - 10:
            break
        perturbed = []
        for t in qw_ths:
            # Perturb by +/- 20%
            factor = random.uniform(0.8, 1.2)
            perturbed.append(t * factor)
        perturbed = constrain_thicknesses(perturbed)
        score = _mean_reflectance(pattern, perturbed, problem)
        if score < best_score:
            best_score = score
            best_ths = list(perturbed)
            
        # Polish
        polished_ths, polished_score = nelder_mead(perturbed, max_iters=100)
        if polished_score < best_score:
            best_score = polished_score
            best_ths = polished_ths

    # Final polish on the best result found
    if best_ths:
        final_ths, final_score = nelder_mead(best_ths, max_iters=300)
        if final_score < best_score:
            best_ths = final_ths
            best_score = final_score

    if best_ths is None:
        return ["L"], [qw["L"]]
        
    return list(pattern), list(best_ths)
