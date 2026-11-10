import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Optimizes the base positions x_i for 64 Kakeya needle triangles to minimize
    the area of their union.
    
    Uses aligned initialization with coordinate descent (grid search + refinement)
    followed by Nelder-Mead polish.
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
        L = x[None, :] + a[None, :] * y[:, None]
        L_sorted = np.sort(L, axis=1)
        gaps = np.diff(L_sorted, axis=1)
        min_gaps = np.minimum(gaps, w[:, None])
        union_len = w + np.sum(min_gaps, axis=1)
        return np.mean(union_len)

    # Aligned initialization: all converge to (0.5, 1)
    x_aligned = 0.5 - a
    best_x = x_aligned.copy()
    best_val = compute_area(best_x)

    # Also try converging to 0
    x_conv0 = -a.copy()
    val_conv0 = compute_area(x_conv0)
    if val_conv0 < best_val:
        best_val = val_conv0
        best_x = x_conv0.copy()

    start_time = time.time()
    deadline = start_time + 600
    safety_margin = 45

    def coordinate_descent(x, max_rounds=8, tol=1e-10):
        """Perform coordinate descent on the area function."""
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
                
                # Grid search over a reasonable range
                # The optimal x[i] is likely within [-1, 1]
                lo = x_old - 0.5
                hi = x_old + 0.5
                
                # Coarse grid
                grid = np.linspace(lo, hi, 80)
                best_local_val = prev_val
                best_local_x = x_old
                
                for gx in grid:
                    x[i] = gx
                    val = compute_area(x)
                    if val < best_local_val:
                        best_local_val = val
                        best_local_x = gx
                
                # Fine refinement around best
                if best_local_x != x_old:
                    step = (hi - lo) / 80
                    lo2 = best_local_x - step
                    hi2 = best_local_x + step
                    grid2 = np.linspace(lo2, hi2, 40)
                    for gx in grid2:
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
    x_cd, val_cd = coordinate_descent(best_x, max_rounds=6)
    if val_cd < best_val:
        best_x = x_cd
        best_val = val_cd

    # Nelder-Mead polish if time permits
    if time.time() < deadline - 60:
        nm_options = {
            'maxiter': 150,
            'xatol': 1e-7,
            'fatol': 1e-9,
            'adaptive': True
        }
        try:
            res = minimize(
                compute_area,
                best_x,
                method='Nelder-Mead',
                options=nm_options
            )
            if res.fun < best_val:
                best_x = res.x
                best_val = res.fun
        except Exception:
            pass

    # Final coordinate descent pass
    if time.time() < deadline - 30:
        x_cd2, val_cd2 = coordinate_descent(best_x, max_rounds=3)
        if val_cd2 < best_val:
            best_x = x_cd2
            best_val = val_cd2

    if not np.all(np.isfinite(best_x)):
        return x_aligned.copy()

    return best_x
