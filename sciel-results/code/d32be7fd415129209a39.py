import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _union_length_vec(starts, width):
    """
    Compute total length of union of 1-D intervals [starts[i], starts[i]+width].
    Returns the length.
    """
    n = len(starts)
    if n == 0:
        return 0.0
    
    # Sort by start point
    order = np.argsort(starts)
    s_sorted = starts[order]
    e_sorted = s_sorted + width
    
    # Merge intervals
    total = 0.0
    cs = s_sorted[0]
    ce = e_sorted[0]
    
    # We can vectorize this if needed, but n=8 is small, so loop is fine
    for k in range(1, n):
        if s_sorted[k] > ce:
            total += ce - cs
            cs = s_sorted[k]
            ce = e_sorted[k]
        else:
            if e_sorted[k] > ce:
                ce = e_sorted[k]
    total += ce - cs
    return total

def _exact_volume_sep(X, Y, N=8, n_c=200):
    """
    Calculate exact volume for separable case x_ij = X_i, y_ij = Y_j.
    Volume = integral_0^1 Lx(c) * Ly(c) dc
    where Lx(c) is the length of union of intervals [X_i + c*i/N, X_i + c*i/N + 1/N]
    and Ly(c) is the length of union of intervals [Y_j + c*j/N, Y_j + c*j/N + 1/N]
    """
    invN = 1.0 / N
    c_samples = np.linspace(0, 1, n_c)
    
    # Pre-compute indices
    i_idx = np.arange(N)
    j_idx = np.arange(N)
    
    Lx = np.zeros(n_c)
    Ly = np.zeros(n_c)
    
    for k in range(n_c):
        c = c_samples[k]
        # x_starts for this c
        x_starts = X + c * i_idx * invN
        # y_starts for this c
        y_starts = Y + c * j_idx * invN
        
        Lx[k] = _union_length_vec(x_starts, invN)
        Ly[k] = _union_length_vec(y_starts, invN)
        
    # Trapezoidal rule or simple Riemann sum. 
    # With 200 points, simple sum is very accurate.
    vol = np.sum(Lx * Ly) * (1.0 / n_c)
    return vol

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    deadline = time.time() + 600.0  # Safety margin
    
    # Keich binary construction as initial guess
    pos_k = int(np.log2(N))
    j_terms = np.arange(1, pos_k + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos_k - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos_k)
    
    # The separable assumption: X_i and Y_j
    # We optimize X and Y independently? No, the volume couples them via product Lx(c)Ly(c).
    # But we can optimize the joint vector [X, Y] of size 16.
    
    best_params = np.concatenate([keich, keich])
    best_vol = _exact_volume_sep(best_params[:N], best_params[N:], N, n_c=500)
    
    def objective(params):
        if time.time() > deadline - 10:
            return best_vol
        X = params[:N]
        Y = params[N:]
        return _exact_volume_sep(X, Y, N, n_c=200)
    
    rng = np.random.default_rng(12345)
    
    # Phase 1: Global Search using Differential Evolution
    # This is robust for finding the global minimum in low dimensions.
    # Bounds: The positions can be anywhere, but relative positions matter.
    # We can fix X[0]=0, Y[0]=0 to remove translational symmetry? 
    # The volume is translation invariant.
    # However, DE works better with bounded spaces. 
    # Let's estimate a reasonable bound. The union length is at most 1 (if disjoint) 
    # but actually max length is 1 + 7/8 = 1.875? No.
    # If X_i are spread out, Lx(c) can be large. 
    # But we want to minimize volume, so we want them close/overlapping.
    # Let's bound X and Y to [-1, 2] relative to each other.
    # Actually, let's just use a wide bound and let DE explore.
    bounds = [(-1.0, 2.0)] * (2 * N)
    
    try:
        if time.time() < deadline - 100:
            # DE is stochastic and can be slow, but N=16 is small.
            res_de = differential_evolution(
                objective, 
                bounds, 
                seed=12345, 
                maxiter=100, 
                tol=1e-8,
                polish=False, # We will polish later
                init='sobol'
            )
            if res_de.fun < best_vol:
                best_vol = res_de.fun
                best_params = res_de.x
    except Exception:
        pass
        
    # Phase 2: Local Refinement with Nelder-Mead
    # Multiple restarts around the best point found by DE (or initial guess if DE failed)
    
    for restart in range(50):
        if time.time() > deadline - 20:
            break
            
        if restart == 0:
            p_start = best_params.copy()
        else:
            # Perturb best_params
            # Scale of perturbation should be small but allow escaping local minima
            scale = 0.01 + 0.1 * (1.0 / (restart + 1))
            p_start = best_params + rng.normal(0, scale, size=2 * N)
            
        try:
            result = minimize(
                objective, 
                p_start, 
                method='Nelder-Mead', 
                options={
                    'maxiter': 1000, 
                    'xatol': 1e-10, 
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            if result.fun < best_vol:
                best_vol = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

    # Final high-precision check and potential further refinement
    if time.time() < deadline - 5:
        # One last very tight Nelder-Mead
        try:
            result = minimize(
                objective, 
                best_params, 
                method='Nelder-Mead', 
                options={
                    'maxiter': 2000, 
                    'xatol': 1e-12, 
                    'fatol': 1e-14,
                    'adaptive': True
                }
            )
            if result.fun < best_vol:
                best_vol = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

    # Construct output array
    # The task requires a (64, 2) array.
    # Row k corresponds to (i, j) with i = k // 8, j = k % 8.
    # x_k = best_params[i]
    # y_k = best_params[N + j]
    
    X_best = best_params[:N]
    Y_best = best_params[N:]
    
    positions = np.empty((N * N, 2))
    for i in range(N):
        for j in range(N):
            k = i * N + j
            positions[k, 0] = X_best[i]
            positions[k, 1] = Y_best[j]
            
    return positions
