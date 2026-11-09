"""Broadband antireflection coating design: Fixed 20x45nm stack with differentiable binary pattern search."""
import time
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

def _compute_reflectance_from_indices(indices, thicknesses, problem):
    """
    Compute reflectance given continuous indices.
    indices: array of shape (N,) with values between n_L and n_H.
    thicknesses: array of shape (N,).
    """
    band = problem["band_nm"]
    wavs = np.linspace(band[0], band[1], 401)
    
    ns = np.asarray(indices, dtype=float)
    ds = np.asarray(thicknesses, dtype=float)
    
    if len(ns) == 0:
        return 100.0 * ((problem["substrate_index"] - problem["ambient_index"]) /
                        (problem["substrate_index"] + problem["ambient_index"])) ** 2

    delta = 2.0 * np.pi * (ns[:, None] * ds[:, None]) / wavs[None, :]
    c = np.cos(delta)
    s = np.sin(delta)
    
    A = np.ones_like(wavs)
    B = np.zeros_like(wavs)
    C = np.zeros_like(wavs)
    D = np.ones_like(wavs)
    
    i = 1.0j
    for j in range(len(ns)):
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
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    materials_dict = problem["materials"]
    
    n_L = materials_dict['L']
    n_H = materials_dict['H']
    
    # Fixed stack configuration: 20 layers of 45 nm each (Total 900 nm)
    # This satisfies max_total (1000) and max_layer (400) and max_layers (20).
    N = 20
    d_fixed = 45.0
    thicknesses = np.full(N, d_fixed)
    
    # Initial binary pattern: Alternating L/H starting with L (closer to air)
    # L=1.38, H=2.35. Air=1.0. Substrate=1.52.
    # Geometric mean of Air and Substrate is sqrt(1.0*1.52) ~ 1.23.
    # L is closer to Air. H is closer to Substrate? No, 2.35 is far.
    # Usually AR coatings start with low index to match air.
    
    best_score = float('inf')
    best_pattern = None
    
    # Strategy 1: Gradient Descent on Continuous Relaxation
    # We treat the layer indices as continuous variables z in [0, 1].
    # n_j = (1 - z_j) * n_L + z_j * n_H
    # We optimize z using Adam-like gradient descent.
    
    # Initialize z with several random starts and structured starts
    starts = []
    
    # Start 1: Alternating L H L H ... (z=0, 1, 0, 1...)
    z1 = np.zeros(N)
    for i in range(1, N, 2):
        z1[i] = 1.0
    starts.append(z1)
    
    # Start 2: Alternating H L H L ... (z=1, 0, 1, 0...)
    z2 = np.ones(N)
    for i in range(1, N, 2):
        z2[i] = 0.0
    starts.append(z2)
    
    # Start 3: Graded profile L...L H...H
    z3 = np.zeros(N)
    z3[10:] = 1.0
    starts.append(z3)
    
    # Start 4: Graded profile H...H L...L
    z4 = np.ones(N)
    z4[10:] = 0.0
    starts.append(z4)
    
    # Start 5-8: Random
    for _ in range(4):
        starts.append(np.random.rand(N))
        
    # Gradient Descent Loop
    learning_rate = 0.1
    momentum = 0.9
    v = np.zeros(N)
    
    for z_init in starts:
        if time.time() > deadline - 10:
            break
            
        z = z_init.copy()
        v = np.zeros(N)
        
        # Run gradient descent for a fixed number of steps or until time runs out
        for step in range(200):
            if time.time() > deadline - 10:
                break
                
            # Compute reflectance and gradient via finite differences
            # Since N=20, 41 evaluations per step is feasible.
            # But we can be smarter: use complex step or just standard FD.
            # Standard FD: (f(z+h) - f(z-h)) / 2h
            
            h = 0.01
            f0 = _compute_reflectance_from_indices((1-z)*n_L + z*n_H, thicknesses, problem)
            
            grad = np.zeros(N)
            for i in range(N):
                z_plus = z.copy()
                z_plus[i] += h
                z_minus = z.copy()
                z_minus[i] -= h
                
                # Clamp to [0,1] for evaluation? The physics works for any n, 
                # but we want to stay in the convex hull of L/H.
                # For gradient, we can evaluate outside [0,1] as the function is smooth.
                
                f_plus = _compute_reflectance_from_indices((1-z_plus)*n_L + z_plus*n_H, thicknesses, problem)
                f_minus = _compute_reflectance_from_indices((1-z_minus)*n_L + z_minus*n_H, thicknesses, problem)
                
                grad[i] = (f_plus - f_minus) / (2 * h)
            
            # Update z with momentum
            v = momentum * v - learning_rate * grad
            z = z + v
            
            # Clip z to [0, 1] to keep indices within [n_L, n_H]
            z = np.clip(z, 0.0, 1.0)
            
            # Check convergence or time
            if step % 10 == 0:
                f_curr = _compute_reflectance_from_indices((1-z)*n_L + z*n_H, thicknesses, problem)
                if f_curr < best_score:
                    best_score = f_curr
                    # Snap to nearest binary
                    z_bin = (z > 0.5).astype(int)
                    best_pattern = ['H' if x == 1 else 'L' for x in z_bin]
                    
    # Strategy 2: Polish the best binary pattern found with Nelder-Mead on thicknesses
    if best_pattern is None:
        # Fallback if GD didn't find anything better than initial guesses
        # Try the alternating pattern
        best_pattern = ['L' if i % 2 == 0 else 'H' for i in range(N)]
        best_score = _mean_reflectance(best_pattern, thicknesses.tolist(), problem)
        
    # Nelder-Mead Polish
    def nelder_mead_polish(pattern, init_ths, max_iters=100):
        ths = list(init_ths)
        n = len(ths)
        if n == 0:
            return pattern, ths, _mean_reflectance(pattern, ths, problem)
            
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        step = 5.0
        simplex = [list(ths)]
        for i in range(n):
            pt = list(ths)
            pt[i] = max(0.0, min(max_layer, pt[i] + step))
            simplex.append(pt)
        
        scores = [_mean_reflectance(pattern, simplex[i], problem) for i in range(n + 1)]
        
        it = 0
        while it < max_iters and time.time() < deadline - 5:
            it += 1
            order = np.argsort(scores)
            simplex = [simplex[i] for i in order]
            scores = [scores[i] for i in order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = [max(0.0, min(max_layer, x)) for x in xr]
            # Check total thickness
            total = sum(xr)
            if total > max_total:
                scale = max_total / total
                xr = [x * scale for x in xr]
            
            fr = _mean_reflectance(pattern, xr, problem)
            
            if fr < scores[0]:
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = [max(0.0, min(max_layer, x)) for x in xe]
                total = sum(xe)
                if total > max_total:
                    scale = max_total / total
                    xe = [x * scale for x in xe]
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
                    xc = [max(0.0, min(max_layer, x)) for x in xc]
                    total = sum(xc)
                    if total > max_total:
                        scale = max_total / total
                        xc = [x * scale for x in xc]
                    fc = _mean_reflectance(pattern, xc, problem)
                    if fc <= fr:
                        simplex[-1] = xc
                        scores[-1] = fc
                    else:
                        break
                else:
                    for i in range(1, n + 1):
                        simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                        simplex[i] = [max(0.0, min(max_layer, x)) for x in simplex[i]]
                        total = sum(simplex[i])
                        if total > max_total:
                            scale = max_total / total
                            simplex[i] = [x * scale for x in simplex[i]]
                        scores[i] = _mean_reflectance(pattern, simplex[i], problem)
                    if np.max(scores) - np.min(scores) < 1e-7:
                        break
                        
        best_idx = np.argmin(scores)
        return pattern, simplex[best_idx], scores[best_idx]
    
    final_pat, final_ths, final_score = nelder_mead_polish(best_pattern, thicknesses.tolist(), max_iters=150)
    
    if final_score < best_score:
        best_score = final_score
        best_pattern = final_pat
        best_ths = final_ths
    else:
        best_ths = thicknesses.tolist()
        
    return list(best_pattern), list(best_ths)
