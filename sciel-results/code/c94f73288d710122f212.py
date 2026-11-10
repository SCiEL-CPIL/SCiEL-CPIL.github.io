import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    frac = 1.0 / N
    start_time = time.time()
    timeout = 600  # Evaluator kills at 630s, leave margin
    
    # Pre-compute indices for each of the 64 tubes
    # Tube k corresponds to i = k // 8, j = k % 8
    indices = np.array([(k // N, k % N) for k in range(N * N)], dtype=np.int64)
    i_vals = indices[:, 0].astype(np.float64)
    j_vals = indices[:, 1].astype(np.float64)
    
    # Vectorized Union Length Calculation for 1D intervals
    # Input: starts (array of length M), lengths (scalar or array, but here scalar frac)
    # Returns: total length of union
    def union_length(starts, lengths):
        # Sort by start
        order = np.argsort(starts)
        s_sorted = starts[order]
        e_sorted = s_sorted + lengths[order]
        
        # Merge intervals
        # Initialize with first interval
        cur_start = s_sorted[0]
        cur_end = e_sorted[0]
        total_len = 0.0
        
        # Iterate through remaining intervals
        for k in range(1, len(s_sorted)):
            if s_sorted[k] > cur_end:
                # Disjoint, add current segment
                total_len += cur_end - cur_start
                cur_start = s_sorted[k]
                cur_end = e_sorted[k]
            else:
                # Overlapping, extend current segment if needed
                if e_sorted[k] > cur_end:
                    cur_end = e_sorted[k]
        
        # Add last segment
        total_len += cur_end - cur_start
        return total_len

    def compute_volume(params):
        """
        params: array of shape (128,)
        First 64 elements are x_k, next 64 are y_k.
        """
        x_pos = params[:64]
        y_pos = params[64:]
        
        # We need to integrate over c in [0, 1]
        # Volume = integral_0^1 L_x(c) * L_y(c) dc
        # L_x(c) is union length of intervals [x_k + c * i_k/8, x_k + c * i_k/8 + 1/8]
        # L_y(c) is union length of intervals [y_k + c * j_k/8, y_k + c * j_k/8 + 1/8]
        
        # Use Gauss-Legendre quadrature for efficiency and accuracy
        # 32 points is usually sufficient for this smooth-ish integrand
        c_pts, w_pts = np.polynomial.legendre.leggauss(32)
        c_vals = 0.5 * (c_pts + 1.0) # Map [-1, 1] to [0, 1]
        
        vol = 0.0
        for c, w in zip(c_vals, w_pts):
            # Calculate starts for x-intervals
            x_starts = x_pos + c * (i_vals / N)
            # Calculate starts for y-intervals
            y_starts = y_pos + c * (j_vals / N)
            
            lx = union_length(x_starts, np.full(64, frac))
            ly = union_length(y_starts, np.full(64, frac))
            
            vol += w * lx * ly
            
        return 0.5 * vol # Factor 0.5 for mapping from [-1,1] to [0,1]

    # Initial Guess: Keich Construction
    # The Keich construction provides a good lower bound for the 1D problem.
    # We will use it to initialize both x and y coordinates for all tubes.
    def keich_positions_1d(N):
        pos = int(np.log2(N))
        j_terms = np.arange(1, pos + 1)
        keich = np.empty(N, dtype=np.float64)
        for idx in range(N):
            eps = np.array(list(str(bin(idx))[2:]), dtype=int)
            eps = np.pad(eps, (pos - len(eps), 0), "constant")
            keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        return keich

    u_keich = keich_positions_1d(N)
    v_keich = keich_positions_1d(N)
    
    # Build initial params
    x_init = np.zeros(64)
    y_init = np.zeros(64)
    for k in range(64):
        i = k // N
        j = k % N
        x_init[k] = u_keich[i]
        y_init[k] = v_keich[j]
        
    params_init = np.concatenate([x_init, y_init])
    
    best_params = params_init.copy()
    best_vol = compute_volume(best_params)
    
    # Optimization Strategy:
    # 1. Nelder-Mead on the full 128D space.
    #    This is expensive but allows non-separable solutions.
    #    We need to be careful with the time limit.
    #    Nelder-Mead iterations scale poorly with dimension, but 128D is manageable
    #    if we limit iterations and use a fast objective.
    
    # To improve success rate, we can try a two-phase approach:
    # Phase 1: Optimize separable structure (14 vars) quickly to get a good local min.
    # Phase 2: Use that as initial guess for full 128D Nelder-Mead.
    
    # Phase 1: Separable Optimization (Fast)
    def objective_separable(params_14):
        u = np.zeros(N)
        v = np.zeros(N)
        u[1:] = params_14[:7]
        v[1:] = params_14[7:]
        
        # Map to 128D params for volume calc
        x_pos = np.zeros(64)
        y_pos = np.zeros(64)
        for k in range(64):
            i = k // N
            j = k % N
            x_pos[k] = u[i]
            y_pos[k] = v[j]
            
        p = np.concatenate([x_pos, y_pos])
        return compute_volume(p)
    
    # Initial guess for separable
    p14_init = np.concatenate([u_keich[1:], v_keich[1:]])
    
    try:
        # Use Nelder-Mead for separable phase too, as it's robust to non-smoothness
        res1 = minimize(
            objective_separable, 
            p14_init, 
            method='Nelder-Mead', 
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        if res1.fun < best_vol:
            best_vol = res1.fun
            # Update best_params from res1.x
            u_opt = np.zeros(N)
            v_opt = np.zeros(N)
            u_opt[1:] = res1.x[:7]
            v_opt[1:] = res1.x[7:]
            
            x_opt = np.zeros(64)
            y_opt = np.zeros(64)
            for k in range(64):
                i = k // N
                j = k % N
                x_opt[k] = u_opt[i]
                y_opt[k] = v_opt[j]
            best_params = np.concatenate([x_opt, y_opt])
    except Exception:
        pass

    # Phase 2: Full 128D Optimization
    # Use Nelder-Mead with adaptive simplex.
    # This is the core structural change: allowing x_k to depend on j and y_k on i.
    
    # Check time
    if time.time() - start_time < timeout - 10:
        try:
            # Set initial simplex size based on the scale of the problem
            # Positions are roughly in [0, 1] or [-0.5, 0.5]
            # Let's use a modest initial step size
            initial_simplex = None # Let scipy handle it, or provide one?
            # Providing a custom simplex might help, but default is usually okay.
            
            res2 = minimize(
                compute_volume, 
                best_params, 
                method='Nelder-Mead', 
                options={
                    'maxiter': 20000, # High limit, but we will check time
                    'xatol': 1e-9, 
                    'fatol': 1e-11, 
                    'adaptive': True,
                    'maxfev': 100000
                }
            )
            
            # Nelder-Mead doesn't have a built-in time check in options, 
            # so we rely on maxiter/maxfev. 
            # If it takes too long, we might miss the deadline.
            # To be safe, we should probably implement a callback to check time?
            # Scipy's minimize supports callbacks.
            
            if res2.fun < best_vol:
                best_vol = res2.fun
                best_params = res2.x
        except Exception:
            pass

    # If time remains, try a few random restarts of the full 128D search
    # or perturbations of the current best.
    if time.time() - start_time < timeout - 30:
        rng = np.random.default_rng(12345)
        for _ in range(3):
            if time.time() - start_time > timeout - 15:
                break
            
            # Perturb best solution
            noise = rng.normal(0, 0.01, size=128)
            p_perturbed = best_params + noise
            
            try:
                res_r = minimize(
                    compute_volume, 
                    p_perturbed, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': 5000,
                        'xatol': 1e-8, 
                        'fatol': 1e-10, 
                        'adaptive': True
                    }
                )
                if res_r.fun < best_vol:
                    best_vol = res_r.fun
                    best_params = res_r.x
            except Exception:
                pass

    # Construct final positions
    x_final = best_params[:64]
    y_final = best_params[64:]
    
    positions = np.column_stack((x_final, y_final))
    
    return positions

if __name__ == "__main__":
    pos = kakeya_needle_3d8()
    print(pos.shape)
