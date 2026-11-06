import numpy as np
import time
from scipy.optimize import minimize

def angles(n, edges, p):
    start_time = time.time()
    deadline = 115.0  # safety margin
    
    # Build adjacency and compute C(z) for all z in {0,1}^n
    # C(z) = number of edges cut by partition z
    # Edge (i,j) is cut if z_i != z_j
    
    m = len(edges)
    # Precompute C(z) for all 2^n states
    N = 1 << n
    C_vals = np.zeros(N, dtype=np.float64)
    
    # For each edge, add 1 to C(z) where the two endpoints differ
    for (i, j) in edges:
        # For all z, check if bit i != bit j
        # Bit i is set if (z >> i) & 1
        # We can vectorize: for each z, cut = ((z >> i) ^ (z >> j)) & 1
        mask_i = (np.arange(N) >> i) & 1
        mask_j = (np.arange(N) >> j) & 1
        C_vals += (mask_i ^ mask_j).astype(np.float64)
    
    # Compute optimal cut by brute force
    opt_cut = np.max(C_vals)
    if opt_cut == 0:
        return np.zeros((p, 2))
    
    # Bitwise complement
    complement = np.arange(N) ^ (N - 1)
    
    def expected_cut(params):
        gamma1, beta1, gamma2, beta2 = params
        # A_v = [cos(beta1) * exp(-i*gamma1*C(v)) - i*sin(beta1)*exp(-i*gamma1*C(complement[v))]] * exp(-i*gamma2*C(v))
        C_v = C_vals
        C_cv = C_vals[complement]
        
        # Complex amplitudes
        # A_v = (cos(beta1)*exp(-i*gamma1*C_v) - 1j*sin(beta1)*exp(-i*gamma1*C_cv)) * exp(-i*gamma2*C_v)
        phase1_v = np.exp(-1j * gamma1 * C_v)
        phase1_cv = np.exp(-1j * gamma1 * C_cv)
        phase2_v = np.exp(-1j * gamma2 * C_v)
        
        A_v = (np.cos(beta1) * phase1_v - 1j * np.sin(beta1) * phase1_cv) * phase2_v
        A_cv = (np.cos(beta1) * phase1_cv - 1j * np.sin(beta1) * phase1_v) * np.exp(-1j * gamma2 * C_cv)
        
        # Final amplitude: cos(beta2)*A_v - i*sin(beta2)*A_cv
        amp = np.cos(beta2) * A_v - 1j * np.sin(beta2) * A_cv
        prob = np.abs(amp) ** 2
        
        return np.sum(prob * C_v) / N
    
    def neg_ratio(params):
        ec = expected_cut(params)
        return -ec / opt_cut
    
    # Grid search for initial points
    best_val = -np.inf
    best_params = None
    
    # Coarse grid
    gamma_range = np.linspace(0.0, np.pi, 20)
    beta_range = np.linspace(0.0, np.pi/2, 15)
    
    # Use symmetry: gamma in [0, pi], beta in [0, pi/2] typically
    # Actually beta can be in [0, pi] but by symmetry we can restrict
    
    count = 0
    total_grid = len(gamma_range) ** 2 * len(beta_range) ** 2
    
    for g1 in gamma_range:
        for b1 in beta_range:
            for g2 in gamma_range:
                for b2 in beta_range:
                    params = np.array([g1, b1, g2, b2])
                    val = -neg_ratio(params)
                    if val > best_val:
                        best_val = val
                        best_params = params.copy()
                    count += 1
                    if time.time() - start_time > deadline:
                        break
                if time.time() - start_time > deadline:
                    break
            if time.time() - start_time > deadline:
                break
        if time.time() - start_time > deadline:
            break
    
    # Local refinement from best point and several other good points
    candidates = [best_params]
    
    # Add some random perturbations around best
    if best_params is not None:
        for _ in range(5):
            pert = best_params + np.random.normal(0, 0.1, 4)
            pert = np.clip(pert, [0, 0, 0, 0], [np.pi, np.pi, np.pi, np.pi])
            candidates.append(pert)
    
    # Also try a few random points
    for _ in range(10):
        cand = np.random.uniform([0, 0, 0, 0], [np.pi, np.pi, np.pi, np.pi])
        candidates.append(cand)
    
    for cand in candidates:
        if time.time() - start_time > deadline:
            break
        try:
            res = minimize(neg_ratio, cand, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < -best_val:
                best_val = -res.fun
                best_params = res.x
        except:
            pass
    
    # Final polish with tighter tolerance
    if best_params is not None and time.time() - start_time < deadline - 5:
        try:
            res = minimize(neg_ratio, best_params, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < -best_val:
                best_val = -res.fun
                best_params = res.x
        except:
            pass
    
    # Format output: p rows of (gamma, beta)
    result = np.array([[best_params[0], best_params[1]],
                       [best_params[2], best_params[3]]])
    return result
