# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time

def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    We exploit the separability of the tube definitions. The x-coordinate of tube (i,j)
    depends on i and base x, and the y-coordinate depends on j and base y.
    If we set base x to depend only on i (x_k = X_i) and base y only on j (y_k = Y_j),
    the union of tubes is a product of a 1D union in x and a 1D union in y.
    The volume is the integral over c in [0,1] of (length of 1D x-union at c) * (length of 1D y-union at c).
    To minimize the volume, we minimize the 1D integral independently for x and y.
    
    We solve the 1D problem: find positions z_0, ..., z_7 such that the integral
    of the length of the union of intervals [z_m + c*m/8, z_m + 1/8 + c*m/8] is minimized.
    We use differential evolution for robust global optimization.
    """
    N = 8
    
    # Deadline guard
    start_time = time.time()
    deadline = 600.0
    safety_margin = 10.0

    def union_length_1d(z, c):
        """Calculate length of union of intervals for given c."""
        # z is array of shape (N,)
        # Intervals are [z[m] + c*m/N, z[m] + 1/N + c*m/N]
        m = np.arange(N)
        starts = z + c * m / N
        ends = starts + 1.0 / N
        # Sort by start
        order = np.argsort(starts)
        s_sorted = starts[order]
        e_sorted = ends[order]
        
        # Merge intervals
        total_len = 0.0
        cur_start = s_sorted[0]
        cur_end = e_sorted[0]
        for k in range(1, N):
            if s_sorted[k] <= cur_end:
                if e_sorted[k] > cur_end:
                    cur_end = e_sorted[k]
            else:
                total_len += cur_end - cur_start
                cur_start = s_sorted[k]
                cur_end = e_sorted[k]
        total_len += cur_end - cur_start
        return total_len

    def cost_1d(z):
        """
        Calculate the integral of the union length over c in [0, 1].
        z is array of shape (N-1,), representing z_1 to z_{N-1}. z_0 is fixed to 0.
        """
        full_z = np.zeros(N)
        full_z[1:] = z
        # Sample c with higher resolution for accuracy
        c_samples = np.linspace(0, 1, 201)
        lengths = np.array([union_length_1d(full_z, c) for c in c_samples])
        # Trapezoidal integration
        integral = np.trapz(lengths, c_samples)
        return integral

    # Use differential evolution for global optimization
    # Bounds for z_1 ... z_7
    # The optimal spread is likely within [-1, 1]
    bounds = [(-1.0, 1.0) for _ in range(N - 1)]
    
    best_z = None
    best_cost = np.inf
    
    # Check time before starting expensive optimization
    if time.time() < deadline - safety_margin - 60:
        try:
            # Differential evolution is robust
            res = scipy.optimize.differential_evolution(
                cost_1d, 
                bounds, 
                seed=42, 
                maxiter=100, 
                tol=1e-6,
                popsize=15,
                update_interval=10,
                polish=True
            )
            if res.fun < best_cost:
                best_cost = res.fun
                best_z = res.x
        except Exception:
            pass

    # If DE failed or time was tight, fallback to Nelder-Mead with multiple restarts
    if best_z is None or time.time() < deadline - safety_margin - 10:
        rng = np.random.default_rng(42)
        for run in range(10):
            if time.time() > deadline - safety_margin:
                break
            
            if run == 0:
                start = np.zeros(N - 1)
            else:
                start = rng.uniform(-0.5, 0.5, N - 1)
            
            try:
                res = scipy.optimize.minimize(cost_1d, start, method='Nelder-Mead', 
                                               options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-7})
                if res.fun < best_cost:
                    best_cost = res.fun
                    best_z = res.x
            except Exception:
                pass

    if best_z is None:
        # Fallback to simple linear spread if all else fails
        # z_m = -m * 0.1
        best_z = -np.arange(1, N) * 0.1

    # Construct full z array
    z_full = np.zeros(N)
    z_full[1:] = best_z
    
    # Construct 64 positions
    # i = k // 8, j = k % 8
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = z_full[i_flat]
    base_y = z_full[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions

# EVOLVE-BLOCK-END
