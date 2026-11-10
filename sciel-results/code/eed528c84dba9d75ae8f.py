import numpy as np
import time

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    frac = 1.0 / N
    start_time = time.time()
    timeout = 620  # Leave margin for the 630s limit
    
    # Precompute shifts for i and j
    shifts = np.arange(N) * frac  # [0, 1/8, ..., 7/8]

    # Keich construction for initial guess
    def keich_positions(N):
        pos = int(np.log2(N))
        j_terms = np.arange(1, pos + 1)
        keich = np.empty(N, dtype=np.float64)
        for idx in range(N):
            eps = np.array(list(str(bin(idx))[2:]), dtype=int)
            eps = np.pad(eps, (pos - len(eps), 0), "constant")
            keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        return keich

    u_init = keich_positions(N)
    v_init = keich_positions(N)

    # We optimize 14 variables: u_1..u_7, v_1..v_7 (u_0, v_0 fixed at 0)
    # This fixes the "center of mass" or one point to remove translational invariance
    # and keeps the box size reasonable.
    
    def exact_volume(u, v):
        # u, v: arrays of shape (N,)
        # Volume is integral over c in [0,1] of Lx(c) * Ly(c) dc
        # Lx(c) is the length of the union of intervals [u_i + c*i/N, u_i + c*i/N + 1/N]
        
        # To be fast, we evaluate at a fixed set of c points and use trapezoidal rule.
        # Since the union length is piecewise linear in c, we can actually compute it exactly
        # if we track the permutation of intervals.
        # However, for SA, a high-resolution quadrature is sufficient and vectorizable.
        
        # Let's use 100 points for c.
        c_vals = np.linspace(0, 1, 101)
        
        # Compute Lx(c) for all c
        # Intervals: start_i(c) = u_i + c * i/N, end_i(c) = start_i(c) + 1/N
        # We need the length of the union of these N intervals for each c.
        
        # Vectorized sweep line for each c?
        # Or just loop over c (101 iterations) and do a vectorized union length calc for N=8 intervals.
        
        lx_vals = np.zeros_like(c_vals)
        ly_vals = np.zeros_like(c_vals)
        
        for idx, c in enumerate(c_vals):
            # x-intervals
            starts_x = u + c * shifts
            ends_x = starts_x + frac
            # Sort by start
            order_x = np.argsort(starts_x)
            s_x = starts_x[order_x]
            e_x = ends_x[order_x]
            
            # Merge intervals
            # N is small (8), so a simple loop is fast enough, but let's vectorize the merge logic if possible.
            # Actually, for N=8, a Python loop over 8 items is negligible compared to the SA steps.
            # But we can do it in numpy:
            # Merged length = e[0] - s[0] + sum_{k=1}^{N-1} max(0, s[k] - max(e[0..k-1]))
            
            # Let's just use a simple loop for clarity and correctness, it's only 101 * 8 * 2 ops.
            len_x = 0.0
            if N > 0:
                curr_s = s_x[0]
                curr_e = e_x[0]
                for k in range(1, N):
                    if s_x[k] > curr_e:
                        len_x += curr_e - curr_s
                        curr_s = s_x[k]
                        curr_e = e_x[k]
                    else:
                        curr_e = max(curr_e, e_x[k])
                len_x += curr_e - curr_s
            lx_vals[idx] = len_x
            
            # y-intervals
            starts_y = v + c * shifts
            ends_y = starts_y + frac
            order_y = np.argsort(starts_y)
            s_y = starts_y[order_y]
            e_y = ends_y[order_y]
            
            len_y = 0.0
            if N > 0:
                curr_s = s_y[0]
                curr_e = e_y[0]
                for k in range(1, N):
                    if s_y[k] > curr_e:
                        len_y += curr_e - curr_s
                        curr_s = s_y[k]
                        curr_e = e_y[k]
                    else:
                        curr_e = max(curr_e, e_y[k])
                len_y += curr_e - curr_s
            ly_vals[idx] = len_y
        
        # Integrate Lx(c) * Ly(c)
        # Use trapezoidal rule
        vol = np.trapz(lx_vals * ly_vals, c_vals)
        
        # Calculate Box Volume for normalization?
        # The task says: V = (fraction of points inside) * (X1-X0)*(Y1-Y0)
        # This is equivalent to the actual geometric volume of the union.
        # So we just minimize the geometric volume.
        
        return vol

    # Initial guess: Keich
    u0 = np.zeros(N)
    v0 = np.zeros(N)
    u0[1:] = u_init[1:]
    v0[1:] = v_init[1:]
    
    # State: 14 parameters
    x_current = np.concatenate([u0[1:], v0[1:]])
    f_current = exact_volume(u0, v0)
    
    best_x = x_current.copy()
    best_f = f_current
    
    # Simulated Annealing Parameters
    T = 0.01
    T_min = 0.001
    cooling = 0.999
    
    # To speed up SA, we can pre-compute some parts or use a faster volume estimator for the SA loop,
    # and only use the exact one for the best update?
    # 101 c-points * 2 (x,y) * 8 intervals * 2 (start/end) * 2 (sort/merge)
    # Sorting 8 elements is ~8*log(8) ~ 24 ops.
    # 101 * 2 * 24 ~ 5000 ops per eval.
    # 1 million evals ~ 5e9 ops. Might be too slow in Python.
    # Let's optimize the volume calculation.
    
    # Optimization: The union length of N intervals can be computed faster if we avoid the Python loop.
    # But N=8 is so small that the loop overhead dominates.
    # Alternative: Use a fixed permutation assumption? No, SA explores large space.
    # Let's try to vectorize the merge for all c at once?
    # The order of intervals changes with c.
    # However, for a given u, the relative order of starts u_i + c*i/N is determined by u_i - u_j = c*(j/N - i/N).
    # This happens at specific c values.
    # This is complex.
    
    # Let's stick to the loop but reduce c-points for SA?
    # 20 c-points might be enough for SA to find the basin, then refine with 100.
    
    def fast_volume(u, v, num_c=20):
        c_vals = np.linspace(0, 1, num_c + 1)
        lx_vals = np.zeros_like(c_vals)
        ly_vals = np.zeros_like(c_vals)
        
        # Pre-allocate arrays for starts/ends to avoid allocation in loop?
        # Actually, let's just do it.
        
        for idx, c in enumerate(c_vals):
            # x
            sx = u + c * shifts
            ex = sx + frac
            ox = np.argsort(sx)
            sx_s = sx[ox]
            ex_s = ex[ox]
            
            # Merge
            # We can use a vectorized trick for small N?
            # No, let's just use the loop. It's fast enough for 20 points.
            len_x = 0.0
            cs = sx_s[0]
            ce = ex_s[0]
            for k in range(1, N):
                if sx_s[k] > ce:
                    len_x += ce - cs
                    cs = sx_s[k]
                    ce = ex_s[k]
                else:
                    if ex_s[k] > ce: ce = ex_s[k]
            len_x += ce - cs
            lx_vals[idx] = len_x
            
            # y
            sy = v + c * shifts
            ey = sy + frac
            oy = np.argsort(sy)
            sy_s = sy[oy]
            ey_s = ey[oy]
            
            len_y = 0.0
            cs = sy_s[0]
            ce = ey_s[0]
            for k in range(1, N):
                if sy_s[k] > ce:
                    len_y += ce - cs
                    cs = sy_s[k]
                    ce = ey_s[k]
                else:
                    if ey_s[k] > ce: ce = ey_s[k]
            len_y += ce - cs
            ly_vals[idx] = len_y
        
        return np.trapz(lx_vals * ly_vals, c_vals)

    # Re-initialize with fast volume for SA
    f_current = fast_volume(u0, v0)
    best_f = f_current
    
    # SA Loop
    rng = np.random.default_rng(12345)
    
    # Step size decay
    step_size = 0.1
    
    while T > T_min:
        if time.time() - start_time > timeout:
            break
            
        # Generate neighbor
        # Perturb u and v independently
        # u params are x[0:7], v params are x[7:14]
        x_new = x_current.copy()
        # Gaussian noise
        noise = rng.normal(0, step_size, size=14)
        x_new += noise
        
        # Extract u, v
        u_new = np.zeros(N)
        v_new = np.zeros(N)
        u_new[1:] = x_new[:7]
        v_new[1:] = x_new[7:]
        
        f_new = fast_volume(u_new, v_new)
        
        # Acceptance criterion
        if f_new < f_current:
            x_current = x_new
            f_current = f_new
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()
        else:
            # Metropolis criterion
            if rng.random() < np.exp(-(f_new - f_current) / T):
                x_current = x_new
                f_current = f_new
        
        T *= cooling
        step_size *= 0.999 # Slow decay of step size

    # Refinement: Use the best found point and run a local optimizer (L-BFGS-B) with the EXACT volume
    # But L-BFGS-B needs a smooth function. The exact volume is non-smooth (due to sorting/merging).
    # We can use the smooth approximation for the gradient, but evaluate the exact volume for the objective?
    # Or just use Nelder-Mead on the exact volume? Nelder-Mead is derivative-free and works on non-smooth functions.
    
    if time.time() - start_time < timeout - 60:
        # Convert best_x to u, v
        u_best = np.zeros(N)
        v_best = np.zeros(N)
        u_best[1:] = best_x[:7]
        v_best[1:] = best_x[7:]
        
        def nm_obj(params):
            u = np.zeros(N)
            v = np.zeros(N)
            u[1:] = params[:7]
            v[1:] = params[7:14]
            return fast_volume(u, v, num_c=50) # Higher resolution for refinement
        
        from scipy.optimize import minimize
        
        try:
            res = minimize(
                nm_obj,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
        except Exception:
            pass

    # Final construction
    u_final = np.zeros(N)
    v_final = np.zeros(N)
    u_final[1:] = best_x[:7]
    v_final[1:] = best_x[7:]
    
    positions = np.zeros((64, 2))
    for k in range(64):
        i = k // 8
        j = k % 8
        positions[k, 0] = u_final[i]
        positions[k, 1] = v_final[j]
        
    return positions

if __name__ == "__main__":
    pos = kakeya_needle_3d8()
    print(pos.shape)
