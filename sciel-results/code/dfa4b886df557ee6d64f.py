import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    # Quadrature nodes and weights for integral on [0, 1]
    # 20 points should be sufficient for smooth functions
    nodes, weights = np.polynomial.legendre.leggauss(20)
    # Transform from [-1, 1] to [0, 1]
    c_nodes = 0.5 * (nodes + 1)
    c_weights = 0.5 * weights

    def union_length(c, x):
        # x is array of shape (N,)
        # Intervals: [x_i + c*i/N, x_i + c*i/N + 1/N]
        starts = x + c * np.arange(N) / N
        ends = starts + 1.0 / N
        
        # Sort by start
        order = np.argsort(starts)
        s_sorted = starts[order]
        e_sorted = ends[order]
        
        # Merge intervals
        # Initialize total length with 0
        total_len = 0.0
        if N == 0:
            return 0.0
        
        current_start = s_sorted[0]
        current_end = e_sorted[0]
        
        for i in range(1, N):
            if s_sorted[i] <= current_end:
                # Overlapping or adjacent
                if e_sorted[i] > current_end:
                    current_end = e_sorted[i]
            else:
                # Disjoint
                total_len += current_end - current_start
                current_start = s_sorted[i]
                current_end = e_sorted[i]
        total_len += current_end - current_start
        return total_len

    def objective(x):
        # x is array of shape (N,)
        # Compute L(c) for all quadrature nodes
        # c_nodes is shape (20,), x is shape (N,)
        # We need L(c_k) for each k
        # Vectorization: 
        # starts shape (20, N)
        c_expanded = c_nodes[:, np.newaxis]
        x_expanded = x[np.newaxis, :]
        
        starts = x_expanded + c_expanded * np.arange(N) / N
        ends = starts + 1.0 / N
        
        # Sort along axis 1
        order = np.argsort(starts, axis=1)
        s_sorted = np.take_along_axis(starts, order, axis=1)
        e_sorted = np.take_along_axis(ends, order, axis=1)
        
        # Merge intervals for each c
        # current_start, current_end shape (20,)
        current_start = s_sorted[:, 0]
        current_end = e_sorted[:, 0]
        
        total_len = np.zeros(len(c_nodes))
        
        # Loop over N-1 intervals for merging (vectorized over c)
        for i in range(1, N):
            s_i = s_sorted[:, i]
            e_i = e_sorted[:, i]
            
            # Check if s_i <= current_end
            overlap_mask = s_i <= current_end
            
            # If overlap, update current_end
            # new_current_end = max(current_end, e_i) where overlap, else current_end
            # But if no overlap, we add length and reset
            # This logic is tricky to fully vectorize without branching per element.
            # Given N=8 and 20 nodes, a loop over nodes or a simple loop is fine.
            # Let's stick to the loop over i, but vectorize the c dimension.
            
            # Case 1: Overlap
            # current_end becomes max(current_end, e_i)
            # Case 2: No overlap
            # total_len += current_end - current_start
            # current_start = s_i
            # current_end = e_i
            
            new_current_end = np.where(overlap_mask, np.maximum(current_end, e_i), current_end)
            
            # For non-overlapping, add to total
            non_overlap_mask = ~overlap_mask
            if np.any(non_overlap_mask):
                total_len[non_overlap_mask] += current_end[non_overlap_mask] - current_start[non_overlap_mask]
                current_start[non_overlap_mask] = s_i[non_overlap_mask]
                current_end[non_overlap_mask] = e_i[non_overlap_mask]
            
            # For overlapping, update current_end
            if np.any(overlap_mask):
                current_end[overlap_mask] = new_current_end[overlap_mask]
                # current_start remains the same for overlapping
            # Wait, if overlap, current_start is unchanged.
            # If no overlap, current_start is updated to s_i.
            
            # Actually, the above logic updates current_end for all, but only resets start/end for non-overlap.
            # Let's re-verify.
            # If overlap: current_end = max(old, e_i). start unchanged.
            # If no overlap: add length. start = s_i, end = e_i.
            
            # The vectorized update above:
            # current_end is updated to max where overlap.
            # For non-overlap, we manually set start and end.
            # This looks correct.
            
        # Add last segment
        total_len += current_end - current_start
        
        # Objective is sum(w * L^2)
        return np.sum(c_weights * total_len**2)

    # Initial guess: Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    x0 = keich.copy()
    
    # Define bounds
    # The shifts are in [0, 7/8]. The width is 1/8.
    # To allow good packing, x_i should be in a range like [-1, 1]
    bounds = [(-1.0, 1.0) for _ in range(N)]
    
    start_time = time.time()
    time_limit = 500.0 # Safety margin
    
    best_x = x0
    best_val = objective(x0)
    
    # Global optimization
    # differential_evolution is stochastic, so it might take time.
    # We can limit maxiter or use a timeout wrapper.
    # scipy's differential_evolution doesn't have a built-in timeout, but we can check time.
    # However, DE with 8 vars and 2000 pop size might be slow if objective is slow.
    # Our objective is vectorized and fast.
    
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            seed=42,
            tol=1e-8,
            polish=True
        )
        
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x
            
    except Exception as e:
        print(f"DE failed: {e}")
        
    # Local refinement with Powell (robust for non-differentiable, though this is differentiable)
    # Or L-BFGS-B
    try:
        result_local = minimize(
            objective, 
            best_x, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 500}
        )
        if result_local.fun < best_val:
            best_val = result_local.fun
            best_x = result_local.x
    except Exception as e:
        print(f"Local min failed: {e}")
        
    # Check time
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        pass # Should have returned earlier if we had a loop, but DE is one shot.
    
    # Construct the 64 positions
    # Row k = tube (i, j) with i = k // 8, j = k % 8
    # x_k = best_x[i]
    # y_k = best_x[j] (assuming symmetry)
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_x[i_flat]
    base_y = best_x[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions
