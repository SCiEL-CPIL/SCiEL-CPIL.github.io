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

    # 1D Union Length Function
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
        for ii in range(1, N):
            if starts[ii] <= curr_end:
                curr_end = max(curr_end, ends[ii])
            else:
                length += curr_end - curr_start
                curr_start = starts[ii]
                curr_end = ends[ii]
        length += curr_end - curr_start
        return length

    # Exact 1D integrated union length
    def objective_1d_exact(x):
        intercepts = np.empty(2 * N, dtype=np.float64)
        intercepts[:N] = x
        intercepts[N:] = x + 1.0 / N
        ep_slopes = np.concatenate([np.arange(N) / N, np.arange(N) / N])
        
        i_idx, j_idx = np.triu_indices(2 * N, k=1)
        da = intercepts[j_idx] - intercepts[i_idx]
        db = ep_slopes[i_idx] - ep_slopes[j_idx]
        
        valid = np.abs(db) > 1e-12
        c_vals = da[valid] / db[valid]
        c_vals = c_vals[(c_vals >= 0.0) & (c_vals <= 1.0)]
        
        breakpoints = np.concatenate([[0.0], c_vals, [1.0]])
        breakpoints = np.unique(np.sort(breakpoints))
        
        if len(breakpoints) < 2:
            return 0.0
            
        lens = np.array([union_length_1d(x, b) for b in breakpoints])
        dc = np.diff(breakpoints)
        return np.sum((lens[:-1] + lens[1:]) / 2.0 * dc)

    # 2D Union Area at height c (rectangles)
    def union_area_2d(x, y, c):
        i_idx = np.arange(N)
        x_starts = x + c * i_idx / N
        x_ends = x_starts + 1.0 / N
        y_starts = y + c * i_idx / N
        y_ends = y_starts + 1.0 / N
        
        x_coords = np.unique(np.concatenate([x_starts, x_ends]))
        area = 0.0
        
        for i in range(len(x_coords) - 1):
            xl = x_coords[i]
            xr = x_coords[i + 1]
            w = xr - xl
            if w < 1e-12: continue
            
            mask_x = (x_starts <= xl) & (x_ends >= xr)
            if not np.any(mask_x): continue
            
            ys = y_starts[mask_x]
            ye = y_ends[mask_x]
            idx = np.argsort(ys)
            ys = ys[idx]
            ye = ye[idx]
            
            l = 0.0
            cs = ys[0]
            ce = ye[0]
            for j in range(1, len(ys)):
                if ys[j] <= ce:
                    if ye[j] > ce: ce = ye[j]
                else:
                    l += ce - cs
                    cs = ys[j]
                    ce = ye[j]
            l += ce - cs
            area += w * l
        return area

    # Exact 2D integrated union volume
    def objective_2d_exact(x, y):
        c_breaks = [0.0, 1.0]
        
        # X crossings
        for i in range(N):
            for j in range(N):
                if i != j:
                    denom = i/N - j/N
                    if abs(denom) > 1e-12:
                        c_val = (x[j] - x[i]) / denom
                        if 0 <= c_val <= 1: c_breaks.append(c_val)
                denom = i/N - j/N
                if abs(denom) > 1e-12:
                    c_val = (x[j] + 1.0/N - x[i]) / denom
                    if 0 <= c_val <= 1: c_breaks.append(c_val)
                denom = i/N - j/N
                if abs(denom) > 1e-12:
                    c_val = (x[j] - x[i] - 1.0/N) / denom
                    if 0 <= c_val <= 1: c_breaks.append(c_val)
        
        # Y crossings
        for i in range(N):
            for j in range(N):
                if i != j:
                    denom = i/N - j/N
                    if abs(denom) > 1e-12:
                        c_val = (y[j] - y[i]) / denom
                        if 0 <= c_val <= 1: c_breaks.append(c_val)
                denom = i/N - j/N
                if abs(denom) > 1e-12:
                    c_val = (y[j] + 1.0/N - y[i]) / denom
                    if 0 <= c_val <= 1: c_breaks.append(c_val)
                denom = i/N - j/N
                if abs(denom) > 1e-12:
                    c_val = (y[j] - y[i] - 1.0/N) / denom
                    if 0 <= c_val <= 1: c_breaks.append(c_val)

        c_breaks = np.unique(np.array(c_breaks))
        
        if len(c_breaks) < 2:
            return 0.0

        areas = np.array([union_area_2d(x, y, c) for c in c_breaks])
        dc = np.diff(c_breaks)
        return np.sum((areas[:-1] + areas[1:]) / 2.0 * dc)

    # Fast grid-based objective for exploration
    c_grid = np.linspace(0, 1, 50)
    dc_grid = c_grid[1] - c_grid[0]
    
    def objective_2d_fast(p):
        x = p[:N]
        y = p[N:]
        total_vol = 0.0
        for c in c_grid:
            total_vol += union_area_2d(x, y, c) * dc_grid
        return total_vol

    # Keich initial guess
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)

    # Phase 1: 1D optimization for x and y independently
    rng_init = np.random.default_rng(42)
    initial_points = [keich.copy()]
    for _ in range(5):
        initial_points.append(keich + rng_init.normal(0, 0.05, size=N))
    for _ in range(3):
        initial_points.append(rng_init.uniform(-0.5, 1.5, size=N))

    best_x = keich.copy()
    best_val = np.inf
    for x0 in initial_points:
        if time.time() - start_time > time_limit - margin: break
        try:
            res = opt.minimize(objective_1d_exact, x0, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except: pass

    x_opt = best_x
    y_opt = best_x.copy()

    # Phase 2: Joint optimization with fast objective for broad exploration
    def objective_joint_fast(p):
        return objective_2d_fast(p)

    p0 = np.concatenate([x_opt, y_opt])
    best_p_fast = p0.copy()
    best_obj_fast = objective_joint_fast(p0)

    joint_starts = [p0.copy()]
    for _ in range(8):
        joint_starts.append(p0 + rng_init.normal(0, 0.03, size=2*N))
    joint_starts.append(p0 + rng_init.uniform(-0.1, 0.1, size=2*N))

    for p_init in joint_starts:
        if time.time() - start_time > time_limit - margin: break
        try:
            res = opt.minimize(objective_joint_fast, p_init, method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            obj_val = objective_joint_fast(res.x)
            if obj_val < best_obj_fast:
                best_obj_fast = obj_val
                best_p_fast = res.x.copy()
        except: pass

    # Phase 3: Refine best candidates from Phase 2 with exact objective
    # Collect top candidates from Phase 2
    candidate_list = [best_p_fast.copy()]
    for p_init in joint_starts[:4]:
        candidate_list.append(p_init.copy())
    
    # Add the original p0
    candidate_list.append(p0.copy())
    
    best_p_exact = best_p_fast.copy()
    best_obj_exact = objective_2d_exact(best_p_exact[:N], best_p_exact[N:])

    for p_cand in candidate_list:
        if time.time() - start_time > time_limit - margin: break
        try:
            res = opt.minimize(objective_2d_exact, p_cand, method='Nelder-Mead',
                               options={'maxiter': 4000, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
            obj_val = objective_2d_exact(res.x[:N], res.x[N:])
            if obj_val < best_obj_exact:
                best_obj_exact = obj_val
                best_p_exact = res.x.copy()
        except: pass

    x_opt = best_p_exact[:N]
    y_opt = best_p_exact[N:]

    # Phase 4: Final high-precision polish
    if time.time() - start_time < time_limit - margin:
        try:
            res = opt.minimize(objective_2d_exact, best_p_exact, method='Nelder-Mead',
                               options={'maxiter': 8000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
            obj_val = objective_2d_exact(res.x[:N], res.x[N:])
            if obj_val < best_obj_exact:
                x_opt = res.x[:N]
                y_opt = res.x[N:]
        except: pass

    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    base_x = x_opt[i_flat]
    base_y = y_opt[j_flat]
    positions = np.stack([base_x, base_y], axis=1)

    if not np.all(np.isfinite(positions)):
        base_x = keich[i_flat]
        base_y = keich[j_flat]
        positions = np.stack([base_x, base_y], axis=1)

    return positions

# EVOLVE-BLOCK-END
