"""Broadband antireflection coating design: Joint pattern and thickness optimization."""
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
    deadline = start_time + 135  # 15s safety margin before 150s kill
    
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

    def nelder_mead_joint(pattern, init_ths, max_iters=200, pattern_prob=0.05):
        """
        Nelder-Mead optimization that primarily adjusts thicknesses
        but occasionally swaps adjacent layers to explore pattern space.
        """
        ths = list(init_ths)
        ths = constrain_thicknesses(ths)
        pat = list(pattern)
        best_score = _mean_reflectance(pat, ths, problem)
        best_ths = list(ths)
        best_pat = list(pat)
        n = len(ths)
        
        if n == 0:
            return best_pat, best_ths, best_score

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
        
        scores = [_mean_reflectance(pat, simplex[i], problem) for i in range(n + 1)]
        
        it = 0
        while it < max_iters and time.time() < deadline:
            it += 1
            order = np.argsort(scores)
            simplex = [simplex[i] for i in order]
            scores = [scores[i] for i in order]
            
            if scores[0] < best_score:
                best_score = scores[0]
                best_ths = list(simplex[0])
                best_pat = list(pat)
                
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = constrain_thicknesses(list(xr))
            
            # Occasionally try a pattern swap on the worst point to explore structure
            # We apply the swap to the pattern used for evaluating the worst point
            # to see if a different structure with similar thicknesses is better.
            # Actually, simpler: just evaluate the reflection for the current pattern.
            # To explore patterns, we can do a separate "pattern jump" step.
            
            fr = _mean_reflectance(pat, xr, problem)
            
            if fr < scores[0]:
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = constrain_thicknesses(list(xe))
                fe = _mean_reflectance(pat, xe, problem)
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
                    fc = _mean_reflectance(pat, xc, problem)
                    if fc <= fr:
                        simplex[-1] = xc
                        scores[-1] = fc
                    else:
                        break
                else:
                    for i in range(1, n + 1):
                        simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                        simplex[i] = constrain_thicknesses(simplex[i])
                        scores[i] = _mean_reflectance(pat, simplex[i], problem)
                    if np.max(scores) - np.min(scores) < 1e-7:
                        break
                        
            # Pattern Exploration Step:
            # With low probability, try swapping two adjacent layers in the best candidate
            # and see if it improves. If so, update the global best pattern.
            if random.random() < pattern_prob and len(pat) > 1:
                swap_idx = random.randint(0, len(pat) - 2)
                new_pat = list(pat)
                new_pat[swap_idx], new_pat[swap_idx+1] = new_pat[swap_idx+1], new_pat[swap_idx]
                # Evaluate the best thicknesses with the new pattern
                test_score = _mean_reflectance(new_pat, best_ths, problem)
                if test_score < best_score:
                    best_score = test_score
                    best_pat = new_pat
                    # Restart the simplex around this new best to converge on the new pattern's optimum
                    # This is a "restart" within the Nelder-Mead loop
                    pat = new_pat
                    ths = list(best_ths)
                    simplex = [list(ths)]
                    for i in range(n):
                        pt = list(ths)
                        pt[i] = max(0.0, min(max_layer, pt[i] + step))
                        simplex.append(pt)
                    scores = [_mean_reflectance(pat, simplex[i], problem) for i in range(n + 1)]
                    
        return best_pat, best_ths, best_score

    # Multi-start optimization
    best_ths = None
    best_pat = None
    best_score = float('inf')
    
    random.seed(42)
    
    # Generate a diverse set of initial patterns
    patterns_to_try = []
    
    # 1. Standard alternating starting with L
    p1 = []
    for i in range(max_layers):
        p1.append('L' if i % 2 == 0 else 'H')
    patterns_to_try.append(p1)
    
    # 2. Standard alternating starting with H
    p2 = []
    for i in range(max_layers):
        p2.append('H' if i % 2 == 0 else 'L')
    patterns_to_try.append(p2)
    
    # 3. Graded index approximation: L...L H...H (transition in middle)
    # Try a few variations of where the transition happens
    for trans in [5, 8, 10, 12, 15]:
        p3 = ['L'] * trans + ['H'] * (max_layers - trans)
        patterns_to_try.append(p3)
        
    # 4. Random patterns with some structure
    for _ in range(5):
        p4 = []
        current = random.choice(['L', 'H'])
        for i in range(max_layers):
            if i > 0 and random.random() < 0.3: # 30% chance to switch
                current = 'H' if current == 'L' else 'L'
            p4.append(current)
        patterns_to_try.append(p4)

    for pat in patterns_to_try:
        if time.time() > deadline - 15:
            break
            
        # Initialize thicknesses with quarter-wave values
        init_ths = [qw[m] for m in pat]
        init_ths = constrain_thicknesses(init_ths)
        
        # Run joint optimization
        res_pat, res_ths, res_score = nelder_mead_joint(pat, init_ths, max_iters=150, pattern_prob=0.02)
        
        if res_score < best_score:
            best_score = res_score
            best_ths = res_ths
            best_pat = res_pat

    # Final polish on the absolute best found
    if best_ths and best_pat:
        final_pat, final_ths, final_score = nelder_mead_joint(best_pat, best_ths, max_iters=300, pattern_prob=0.01)
        if final_score < best_score:
            best_score = final_score
            best_ths = final_ths
            best_pat = final_pat

    if best_ths is None:
        # Fallback: Single layer
        return ["L"], [qw["L"]]
        
    return list(best_pat), list(best_ths)
