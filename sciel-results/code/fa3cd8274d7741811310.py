# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    Returns base positions for 64 tubes.
    """
    N = 8
    start_time = time.time()
    time_limit = 600.0
    margin = 60.0
    
    # 1D Union Length Function (for potential fallback or comparison)
    def union_length_1d(x, c):
        i = np.arange(N)
        starts = x + c * i / N
        ends = starts + 1.0 / N
        idx = np.argsort(starts)
        starts = starts[idx]
        ends = ends[idx]
        length = 0.0
        curr_start = starts[0]
        curr_end = ends[0]
        for i in range(1, N):
            if starts[i] <= curr_end:
                curr_end = max(curr_end, ends[i])
            else:
                length += curr_end - curr_start
                curr_start = starts[i]
                curr_end = ends[i]
        length += curr_end - curr_start
        return length

    # 2D Union Area Function at height c
    # u: x base positions (N)
    # v: y base positions (N)
    # c: height
    def union_area_2d(u, v, c):
        # x intervals
        i_idx = np.arange(N)
        x_starts = u + c * i_idx / N
        x_ends = x_starts + 1.0 / N
        
        # y intervals
        j_idx = np.arange(N)
        y_starts = v + c * j_idx / N
        y_ends = y_starts + 1.0 / N
        
        # Get unique sorted x and y coordinates
        # There are 2N x-coords and 2N y-coords
        x_coords = np.sort(np.concatenate([x_starts, x_ends]))
        y_coords = np.sort(np.concatenate([y_starts, y_ends]))
        
        # Remove duplicates? 
        # If duplicates exist, the cell width is 0, so it doesn't contribute to area.
        # We can just iterate over the intervals between consecutive coords.
        
        # To check coverage efficiently:
        # For each x-interval (between x_coords[k] and x_coords[k+1]), 
        # determine which u-intervals cover it.
        # An interval i covers [x_coords[k], x_coords[k+1]] if 
        # x_starts[i] <= x_coords[k] and x_ends[i] >= x_coords[k+1].
        
        # Precompute coverage masks for x and y
        # num_x_cells = 2N - 1
        # num_y_cells = 2N - 1
        
        num_x_cells = 2 * N - 1
        num_y_cells = 2 * N - 1
        
        # x_covered[k] is True if the k-th x-cell is covered by at least one interval
        x_covered = np.zeros(num_x_cells, dtype=bool)
        for k in range(num_x_cells):
            left = x_coords[k]
            right = x_coords[k+1]
            # Check if any interval i covers [left, right]
            # We can vectorize this
            covers = (x_starts <= left) & (x_ends >= right)
            x_covered[k] = np.any(covers)
            
        y_covered = np.zeros(num_y_cells, dtype=bool)
        for l in range(num_y_cells):
            bottom = y_coords[l]
            top = y_coords[l+1]
            covers = (y_starts <= bottom) & (y_ends >= top)
            y_covered[l] = np.any(covers)
            
        # The area is the sum of areas of cells (k, l) where x_covered[k] and y_covered[l]
        # Area of cell (k, l) = (x_coords[k+1] - x_coords[k]) * (y_coords[l+1] - y_coords[l])
        
        x_widths = np.diff(x_coords)
        y_heights = np.diff(y_coords)
        
        # Create grid of widths and heights
        # area_grid[k, l] = x_widths[k] * y_heights[l]
        # Mask with x_covered[:, None] and y_covered[None, :]
        
        area_grid = np.outer(x_widths, y_heights)
        mask = x_covered[:, None] & y_covered[None, :]
        
        return np.sum(area_grid[mask])

    # Objective function: Integral of union_area_2d over c in [0, 1]
    def objective_2d(params):
        u = params[:N]
        v = params[N:]
        
        # Find breakpoints
        # Breakpoints occur when x_endpoints cross or y_endpoints cross
        x_lines_a = np.concatenate([u, u + 1.0/N])
        x_lines_b = np.concatenate([np.arange(N)/N, np.arange(N)/N])
        
        y_lines_a = np.concatenate([v, v + 1.0/N])
        y_lines_b = np.concatenate([np.arange(N)/N, np.arange(N)/N])
        
        def get_breakpoints(lines_a, lines_b):
            bps = set([0.0, 1.0])
            n = len(lines_a)
            for i in range(n):
                for j in range(i+1, n):
                    if abs(lines_b[i] - lines_b[j]) > 1e-12:
                        c_val = (lines_a[j] - lines_a[i]) / (lines_b[i] - lines_b[j])
                        if 0.0 <= c_val <= 1.0:
                            bps.add(c_val)
            return sorted(bps)
        
        x_bps = get_breakpoints(x_lines_a, x_lines_b)
        y_bps = get_breakpoints(y_lines_a, y_lines_b)
        
        all_bps = sorted(set(x_bps + y_bps))
        
        total_area = 0.0
        for k in range(len(all_bps) - 1):
            c1 = all_bps[k]
            c2 = all_bps[k+1]
            # Area is quadratic in c between breakpoints
            # Use Simpson's rule with 3 points: c1, mid, c2
            c_mid = (c1 + c2) / 2.0
            
            a1 = union_area_2d(u, v, c1)
            a2 = union_area_2d(u, v, c_mid)
            a3 = union_area_2d(u, v, c2)
            
            # Simpson's rule for quadratic is exact
            total_area += (c2 - c1) / 6.0 * (a1 + 4*a2 + a3)
            
        return total_area

    # Initial guess: Keich construction for 1D
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        
    # Normalize/Shift to center around 0? 
    # The problem doesn't restrict the range, but the box size depends on min/max.
    # Smaller box is better. Keich values are small.
    
    x0 = keich.copy()
    y0 = keich.copy()
    params0 = np.concatenate([x0, y0])
    
    # Optimization
    # Use Nelder-Mead for the 16D problem
    # It might be slow, so we set a time limit
    
    res = opt.minimize(objective_2d, params0, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
    
    if time.time() - start_time > time_limit - margin:
        # Fallback to initial guess
        params_opt = params0
    else:
        params_opt = res.x
        
        # Try a second pass with tighter tolerance if time allows
        if time.time() - start_time < time_limit - margin - 30:
             res2 = opt.minimize(objective_2d, params_opt, method='Nelder-Mead', 
                                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
             params_opt = res2.x

    u_opt = params_opt[:N]
    v_opt = params_opt[N:]
    
    # Construct positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = u_opt[i_flat]
    base_y = v_opt[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    # Sanity check
    if not np.all(np.isfinite(positions)):
        base_x = keich[i_flat]
        base_y = keich[j_flat]
        positions = np.stack([base_x, base_y], axis=1)
        
    return positions

# EVOLVE-BLOCK-END
