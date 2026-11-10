import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Optimizes the base positions x_i for 64 Kakeya needle triangles to minimize
    the area of their union.
    
    Uses a fast vectorized area computation with coordinate descent and
    Nelder-Mead refinement.
    """
    n = 64
    M = 20000
    y = (np.arange(M) + 0.5) / M
    a = (np.arange(n) + 1) / n
    w = (1 - y) / n

    def compute_area(x):
        """
        Computes the area of the union of the triangles.
        x: shape (n,)
        """
        # L_i(y) = x_i + a_i * y
        L = x[None, :] + a[None, :] * y[:, None]
        L_sorted = np.sort(L, axis=1)
        gaps = np.diff(L_sorted, axis=1)
        min_gaps = np.minimum(gaps, w[:, None])
        union_len = w + np.sum(min_gaps, axis=1)
        return np.mean(union_len)

    def compute_area_smooth(x, beta=100.0):
        """
        Smooth surrogate using soft-min: min(a,b) ≈ -(1/beta)*log(exp(-beta*a)+exp(-beta*b))
        This removes kinks for gradient-based methods.
        """
        L = x[None, :] + a[None, :] * y[:, None]
        L_sorted = np.sort(L, axis=1)
        gaps = np.diff(L_sorted, axis=1)
        # soft-min of gaps and w
        # min(g, w) ≈ -(1/beta)*log(exp(-beta*g) + exp(-beta*w))
        # For numerical stability: min(g,w) = g + w - max(g-w, 0)
        # soft: g + w - softplus((g-w)/beta)*beta ... let's use:
        # min(g,w) ≈ -1/beta * log(exp(-beta*g) + exp(-beta*w))
        # = -1/beta * log(exp(-beta*g) + exp(-beta*w))
        # For stability, factor out exp(-beta*min(g,w)):
        # = -1/beta * ( -beta*min(g,w) + log(exp(-beta*(g-min)) + exp(-beta*(w-min))) )
        # = min(g,w) - 1/beta * log(exp(-beta*(g-min)) + exp(-beta*(w-min)))
        # = min(g,w) - 1/beta * log(1 + exp(-beta*|g-w|))
        # So: soft_min(g,w) = min(g,w) - (1/beta)*log(1 + exp(-beta*abs(g-w)))
        
        gmin = np.minimum(gaps, w[:, None])
        diff = np.abs(gaps - w[:, None])
        # For large beta*diff, exp(-beta*diff) ≈ 0, so correction ≈ 0
        # For small beta*diff, correction ≈ (1/beta)*log(2)
        # Use log1p for stability
        correction = (1.0 / beta) * np.log1p(np.exp(-beta * diff))
        soft_min_gaps = gmin - correction
        union_len = w + np.sum(soft_min_gaps, axis=1)
        return np.mean(union_len)

    # Initializations
    inits = []
    
    # Aligned: all converge to (0.5, 1)
    x_aligned = 0.5 - a
    inits.append(x_aligned.copy())
    
    # Converging to 0
    x_conv0 = -a.copy()
    inits.append(x_conv0)
    
    # Keich
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    inits.append(np.array(positions, dtype=np.float64))
    
    # Random
    np.random.seed(42)
    inits.append(np.random.uniform(-0.5, 0.5, n))

    # Evaluate all inits
    best_x = inits[0].copy()
    best_val = compute_area(best_x)
    for x0 in inits[1:]:
        val = compute_area(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()

    start_time = time.time()
    deadline = start_time + 600
    safety_margin = 30

    def coordinate_descent_breakpoint(x, max_rounds=15, tol=1e-12):
        """
        Coordinate descent using exact breakpoint analysis.
        
        For fixed x_j (j≠i), the area as a function of x_i is piecewise linear.
        The breakpoints occur when L_i(y_m) = L_j(y_m) for some m, j,
        i.e., x_i + a_i*y_m = x_j + a_j*y_m
        => x_i = x_j + (a_j - a_i)*y_m
        
        Also breakpoints when the sorted order changes at a given y_m.
        
        Strategy: collect candidate breakpoints, evaluate at each, pick best.
        """
        x = x.copy()
        prev_val = compute_area(x)
        
        for round_num in range(max_rounds):
            if time.time() > deadline - safety_margin:
                break
            
            improved = False
            for i in range(n):
                if time.time() > deadline - safety_margin:
                    break
                
                x_old = x[i]
                
                # Collect breakpoints: x_i values where L_i(y_m) = L_j(y_m)
                # x_i = x_j + (a_j - a_i) * y_m
                # For each j ≠ i and each m, this gives a candidate.
                # That's 63 * 20000 = 1.26M candidates - too many.
                # 
                # Instead, note that the function is piecewise linear in x_i
                # with slopes that change at these breakpoints. The minimum
                # of a piecewise linear function occurs at a breakpoint or
                # where the slope changes sign.
                #
                # Practical approach: the relevant breakpoints are those
                # near the current x_i. Use a local window.
                #
                # Alternative: use the fact that for a given y_m,
                # L_i(y_m) = x_i + a_i*y_m. The union length at height m
                # depends on the sorted order of all L_j(y_m).
                # 
                # Let's use a smarter approach: evaluate at a set of
                # "natural" breakpoints.
                
                # Natural breakpoints from other coordinates at a few key heights
                # Use y_m for m = 0, M/4, M/2, 3M/4, M-1
                key_ms = [0, M//4, M//2, 3*M//4, M-1]
                candidates = set()
                candidates.add(x_old)
                
                for j in range(n):
                    if j == i:
                        continue
                    for m in key_ms:
                        ym = y[m]
                        bp = x[j] + (a[j] - a[i]) * ym
                        candidates.add(bp)
                
                # Also add some offset from current
                for delta in [-0.01, -0.001, 0.001, 0.01, -0.05, 0.05]:
                    candidates.add(x_old + delta)
                
                # Evaluate at all candidates
                best_local_val = prev_val
                best_local_x = x_old
                
                for gx in candidates:
                    if not np.isfinite(gx):
                        continue
                    x[i] = gx
                    val = compute_area(x)
                    if val < best_local_val:
                        best_local_val = val
                        best_local_x = gx
                
                # If we found an improvement, do a local refinement
                if best_local_x != x_old:
                    # Fine grid around best_local_x
                    span = 0.02
                    grid = np.linspace(best_local_x - span, best_local_x + span, 200)
                    for gx in grid:
                        x[i] = gx
                        val = compute_area(x)
                        if val < best_local_val:
                            best_local_val = val
                            best_local_x = gx
                    
                    x[i] = best_local_x
                    if abs(best_local_val - prev_val) > tol:
                        improved = True
                        prev_val = best_local_val
                else:
                    x[i] = x_old
            
            if not improved:
                break
        
        return x, prev_val

    # Run coordinate descent from best init
    x_cd, val_cd = coordinate_descent_breakpoint(best_x, max_rounds=10)
    if val_cd < best_val:
        best_x = x_cd
        best_val = val_cd

    # Nelder-Mead with smooth surrogate (handles kinks better)
    if time.time() < deadline - 90:
        nm_options = {
            'maxiter': 500,
            'xatol': 1e-8,
            'fatol': 1e-10,
            'adaptive': True
        }
        try:
            res = minimize(
                lambda x: compute_area_smooth(x, beta=200.0),
                best_x,
                method='Nelder-Mead',
                options=nm_options
            )
            if res.fun < best_val:
                # Verify with exact area
                exact_val = compute_area(res.x)
                if exact_val < best_val:
                    best_x = res.x
                    best_val = exact_val
        except Exception:
            pass

    # Second Nelder-Mead with raw area (nonsmooth)
    if time.time() < deadline - 60:
        nm_options2 = {
            'maxiter': 300,
            'xatol': 1e-9,
            'fatol': 1e-11,
            'adaptive': True
        }
        try:
            res2 = minimize(
                compute_area,
                best_x,
                method='Nelder-Mead',
                options=nm_options2
            )
            if res2.fun < best_val:
                best_x = res2.x
                best_val = res2.fun
        except Exception:
            pass

    # Final coordinate descent pass
    if time.time() < deadline - 30:
        x_cd2, val_cd2 = coordinate_descent_breakpoint(best_x, max_rounds=8)
        if val_cd2 < best_val:
            best_x = x_cd2
            best_val = val_cd2

    # Final polish with very tight Nelder-Mead
    if time.time() < deadline - 15:
        nm_options3 = {
            'maxiter': 200,
            'xatol': 1e-10,
            'fatol': 1e-12,
            'adaptive': True
        }
        try:
            res3 = minimize(
                compute_area,
                best_x,
                method='Nelder-Mead',
                options=nm_options3
            )
            if res3.fun < best_val:
                best_x = res3.x
                best_val = res3.fun
        except Exception:
            pass

    if not np.all(np.isfinite(best_x)):
        return inits[0].copy()  # Fallback to aligned

    return best_x
