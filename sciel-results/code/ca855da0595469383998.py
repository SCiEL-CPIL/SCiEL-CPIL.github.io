import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    deadline = time.time() + 630.0

    # Keich binary construction for initial guess
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    # Scale and shift for better starting point
    X0 = keich * 0.5 + 0.25
    Y0 = keich * 0.5 + 0.25

    # Precompute indices for vectorization
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    def compute_1d_union_length(positions, c, N):
        """Compute total length of union of intervals [pos[i] + c*i/N, pos[i] + c*i/N + 1/N] for i=0..N-1"""
        starts = positions + c * np.arange(N) / N
        ends = starts + 1.0 / N
        order = np.argsort(starts)
        s_sorted = starts[order]
        e_sorted = ends[order]
        # Merge intervals
        total = 0.0
        cur_start = s_sorted[0]
        cur_end = e_sorted[0]
        for k in range(1, N):
            if s_sorted[k] <= cur_end:
                cur_end = max(cur_end, e_sorted[k])
            else:
                total += cur_end - cur_start
                cur_start = s_sorted[k]
                cur_end = e_sorted[k]
        total += cur_end - cur_start
        return total

    def volume_estimate(positions_1d_x, positions_1d_y, n_samples, seed):
        """Estimate volume using separable structure."""
        X = np.asarray(positions_1d_x)
        Y = np.asarray(positions_1d_y)
        
        x_min = np.min(X)
        x_max = np.max(X + (np.arange(N) + 1) / N)
        y_min = np.min(Y)
        y_max = np.max(Y + (np.arange(N) + 1) / N)
        
        box_area = (x_max - x_min) * (y_max - y_min)
        if box_area <= 1e-9:
            return 1e9
        
        rng = np.random.default_rng(seed)
        c_samples = rng.uniform(0, 1, n_samples)
        
        # Vectorize c loop? No, N is small (8), so python loop is fine, but we can batch c.
        # Let's try to vectorize over c if possible.
        # starts_x = X[None, :] + c_samples[:, None] * np.arange(N)[None, :] / N
        # This creates an array of shape (n_samples, N).
        # We need union length for each row.
        # Sorting along axis 1 is possible.
        
        c_samples = rng.uniform(0, 1, n_samples)
        
        # Vectorized union length calculation
        # Shape: (n_samples, N)
        starts_x = X[None, :] + c_samples[:, None] * (np.arange(N) / N)[None, :]
        ends_x = starts_x + 1.0 / N
        
        starts_y = Y[None, :] + c_samples[:, None] * (np.arange(N) / N)[None, :]
        ends_y = starts_y + 1.0 / N
        
        # Sort starts
        order_x = np.argsort(starts_x, axis=1)
        s_sorted_x = np.take_along_axis(starts_x, order_x, axis=1)
        e_sorted_x = np.take_along_axis(ends_x, order_x, axis=1)
        
        order_y = np.argsort(starts_y, axis=1)
        s_sorted_y = np.take_along_axis(starts_y, order_y, axis=1)
        e_sorted_y = np.take_along_axis(ends_y, order_y, axis=1)
        
        # Merge intervals vectorized
        # This is tricky to vectorize fully without loops over N, but N=8 is small.
        # We can do a loop over N for the merge logic, operating on arrays of size n_samples.
        
        cur_start_x = s_sorted_x[:, 0]
        cur_end_x = e_sorted_x[:, 0]
        total_len_x = np.zeros(n_samples)
        
        for k in range(1, N):
            s_k = s_sorted_x[:, k]
            e_k = e_sorted_x[:, k]
            
            # If s_k <= cur_end_x, merge
            # else, add cur_end_x - cur_start_x to total, reset
            mask = s_k <= cur_end_x
            # Update cur_end_x
            cur_end_x = np.where(mask, np.maximum(cur_end_x, e_k), e_k)
            # Update cur_start_x
            cur_start_x = np.where(mask, cur_start_x, s_k)
            # Add to total for those that didn't merge
            total_len_x += np.where(~mask, cur_end_x - s_k, 0.0) # Wait, logic error in vectorization above?
            
            # Correct logic:
            # If merged: total += 0, cur_end = max(cur_end, e_k), cur_start unchanged
            # If not merged: total += cur_end - cur_start, cur_start = s_k, cur_end = e_k
            
            # Let's re-implement the vectorized merge carefully.
            # This is hard to do in a single pass vectorized for arbitrary N without a loop over N.
            # Since N=8, a loop over N is fast.
            
            # Let's revert to the loop over N, but vectorized over c_samples.
            # This is much faster than looping over c_samples in Python.
            
        # Let's write a helper that does the vectorized merge over N steps.
        def vec_union_len(s_sorted, e_sorted):
            # s_sorted, e_sorted are (n_samples, N)
            n_s, n_i = s_sorted.shape
            cur_s = s_sorted[:, 0]
            cur_e = e_sorted[:, 0]
            total = np.zeros(n_s)
            for k in range(1, n_i):
                s_k = s_sorted[:, k]
                e_k = e_sorted[:, k]
                
                is_merge = s_k <= cur_e
                
                # If merge:
                #   cur_e = max(cur_e, e_k)
                #   cur_s = cur_s
                #   total += 0
                # If no merge:
                #   total += cur_e - cur_s
                #   cur_s = s_k
                #   cur_e = e_k
                
                total += np.where(is_merge, 0.0, cur_e - cur_s)
                cur_e = np.where(is_merge, np.maximum(cur_e, e_k), e_k)
                cur_s = np.where(is_merge, cur_s, s_k)
                
            total += cur_e - cur_s
            return total

        x_lens = vec_union_len(s_sorted_x, e_sorted_x)
        y_lens = vec_union_len(s_sorted_y, e_sorted_y)
        
        V = np.mean(x_lens * y_lens)
        return V

    def objective(params, n_samples=10000, seed=12345):
        if time.time() > deadline - 10:
            return 1e9
        X = params[:N]
        Y = params[N:]
        return volume_estimate(X, Y, n_samples, seed)
    
    def objective_fine(params):
        return objective(params, n_samples=50000, seed=12345)

    # Bounds for DE
    # Positions are roughly in [0,1] but can shift.
    # The box is determined by min/max.
    # Let's restrict to [-0.5, 1.5] to keep the box small.
    bounds = [(-0.5, 1.5)] * (2 * N)
    
    initial_params = np.concatenate([X0, Y0])
    
    # Global Search with Differential Evolution
    # popsize=10, maxiter=50? 
    # DE is expensive. Let's estimate cost.
    # One eval: 10000 samples, N=8. Vectorized merge is 7 steps of numpy ops on 10000x8 arrays. Very fast.
    # Let's try popsize=15, maxiter=100.
    
    best_params = initial_params.copy()
    best_score = objective_fine(initial_params)
    
    if time.time() < deadline - 100:
        try:
            # Use coarse objective for DE
            def obj_de(params):
                return objective(params, n_samples=5000, seed=12345)
            
            res_de = differential_evolution(obj_de, bounds, maxiter=200, popsize=15, 
                                            tol=1e-4, seed=12345, polish=False, 
                                            init='sobol')
            
            if res_de.fun < best_score:
                # Verify with fine objective
                fine_score = objective_fine(res_de.x)
                if fine_score < best_score:
                    best_score = fine_score
                    best_params = res_de.x.copy()
        except Exception:
            pass

    # Polish with Nelder-Mead on fine objective
    if time.time() < deadline - 30:
        try:
            res_final = minimize(objective_fine, best_params, method='Nelder-Mead',
                                 options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-9, 'disp': False})
            if res_final.fun < best_score:
                best_score = res_final.fun
                best_params = res_final.x.copy()
        except Exception:
            pass
            
    X_opt = best_params[:N]
    Y_opt = best_params[N:]
    
    base_x = X_opt[i_flat]
    base_y = Y_opt[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions
