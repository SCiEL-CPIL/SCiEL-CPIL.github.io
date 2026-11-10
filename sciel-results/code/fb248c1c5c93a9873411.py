# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def kakeya_needle_3d8() -> np.ndarray:
    """
    Minimizes the volume of the union of 64 sheared tubes in 3D.
    The problem decouples into two identical 1D problems for the x and y coordinates.
    We use L-BFGS-B to optimize the 1D base positions.
    """
    N = 8
    w = 1.0 / N
    deadline = time.time() + 590  # 10s safety margin
    
    def union_length_1d(x, c):
        """
        Computes the length of the union of intervals [x_i + c*i/N, x_i + c*i/N + w] for i=0..N-1.
        x: array of shape (N,)
        c: scalar height
        """
        # Shifts: x_i + c * i / N
        shifts = c * np.arange(N) / N
        starts = x + shifts
        ends = starts + w
        
        # Sort intervals by start
        order = np.argsort(starts)
        s = starts[order]
        e = ends[order]
        
        # Merge intervals
        union_len = 0.0
        if N > 0:
            current_start = s[0]
            current_end = e[0]
            for k in range(1, N):
                if s[k] > current_end:
                    union_len += current_end - current_start
                    current_start = s[k]
                    current_end = e[k]
                else:
                    if e[k] > current_end:
                        current_end = e[k]
            union_len += current_end - current_start
        return union_len

    def objective_1d(x):
        """
        Objective function for 1D optimization: integral of union length over c in [0, 1].
        Uses trapezoidal rule with 50 points for speed and accuracy.
        """
        # Check for NaN/Inf
        if not np.all(np.isfinite(x)):
            return 1e10
        
        # Quadrature points
        c_points = np.linspace(0.0, 1.0, 50)
        lengths = np.array([union_length_1d(x, c) for c in c_points])
        return np.trapz(lengths, c_points)

    # Initial guess: Keich construction or simple spread
    # Let's try a simple spread first, then optimize
    x0 = np.linspace(0, 1, N)
    
    # Optimization
    best_x = x0
    best_val = objective_1d(x0)
    
    try:
        # L-BFGS-B is good for this
        res = minimize(
            objective_1d, 
            x0, 
            method='L-BFGS-B', 
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if res.success or res.fun < best_val:
            best_x = res.x
            best_val = res.fun
    except Exception:
        pass

    # If time remains, try a few random restarts or different initial guesses
    if time.time() < deadline - 10:
        rng = np.random.default_rng(42)
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            x_rand = rng.uniform(0, 1, N)
            # Sort x to avoid symmetry issues? The problem is symmetric under permutation if we also permute i, 
            # but i is fixed. However, swapping x_i and x_j swaps the tubes (i,j) and (j,i) in x-projection?
            # No, tube i has shift i/N. So x_i is tied to i.
            # But the objective is invariant under x_i -> x_i + const. We can fix x_0 = 0 to reduce dimensionality?
            # Let's just optimize as is.
            try:
                res = minimize(
                    objective_1d, 
                    x_rand, 
                    method='L-BFGS-B', 
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
            except Exception:
                pass

    # The x and y problems are identical
    x_opt = best_x
    y_opt = best_x
    
    # Construct the 64x2 array
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_opt[i_flat]
    base_y = y_opt[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    # Validation
    if not np.all(np.isfinite(positions)):
        # Fallback to Keich
        pos = int(np.log2(N))
        j_terms = np.arange(1, pos + 1)
        keich = np.empty(N, dtype=np.float64)
        for idx in range(N):
            eps = np.array(list(str(bin(idx))[2:]), dtype=int)
            eps = np.pad(eps, (pos - len(eps), 0), "constant")
            keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        base_x = keich[i_flat]
        base_y = keich[j_flat]
        positions = np.stack([base_x, base_y], axis=1)
        
    return positions

# EVOLVE-BLOCK-END
