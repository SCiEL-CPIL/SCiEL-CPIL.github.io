# EVOLVE-BLOCK-START
import numpy as np
import time

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    N2 = 64
    i_idx = np.arange(N2) // N
    j_idx = np.arange(N2) % N
    s_x = i_idx.astype(np.float64) / N
    s_y = j_idx.astype(np.float64) / N
    width = 1.0 / N
    
    # Precompute c-samples for fast volume estimation
    # Use 2001 points for good accuracy
    c_samples = np.linspace(0.0, 1.0, 2001)
    dc = c_samples[1] - c_samples[0]
    
    def volume_fast(pos):
        """Fast volume estimate using fixed c-samples."""
        bx = pos[:, 0]
        by = pos[:, 1]
        
        # x-union at each c
        lefts_x = bx[None, :] + c_samples[:, None] * s_x[None, :]
        rights_x = lefts_x + width
        order_x = np.argsort(lefts_x, axis=1)
        ls_x = np.take_along_axis(lefts_x, order_x, axis=1)
        rs_x = np.take_along_axis(rights_x, order_x, axis=1)
        
        # Merge intervals along axis 1
        ulen_x = np.zeros(len(c_samples))
        for ci in range(len(c_samples)):
            cur_l, cur_r = ls_x[ci, 0], rs_x[ci, 0]
            ul = 0.0
            for m in range(1, N2):
                if ls_x[ci, m] > cur_r:
                    ul += cur_r - cur_l
                    cur_l, cur_r = ls_x[ci, m], rs_x[ci, m]
                else:
                    cur_r = max(cur_r, rs_x[ci, m])
            ul += cur_r - cur_l
            ulen_x[ci] = ul
        
        # y-union at each c
        lefts_y = by[None, :] + c_samples[:, None] * s_y[None, :]
        rights_y = lefts_y + width
        order_y = np.argsort(lefts_y, axis=1)
        ls_y = np.take_along_axis(lefts_y, order_y, axis=1)
        rs_y = np.take_along_axis(rights_y, order_y, axis=1)
        
        ulen_y = np.zeros(len(c_samples))
        for ci in range(len(c_samples)):
            cur_l, cur_r = ls_y[ci, 0], rs_y[ci, 0]
            ul = 0.0
            for m in range(1, N2):
                if ls_y[ci, m] > cur_r:
                    ul += cur_r - cur_l
                    cur_l, cur_r = ls_y[ci, m], rs_y[ci, m]
                else:
                    cur_r = max(cur_r, rs_y[ci, m])
            ul += cur_r - cur_l
            ulen_y[ci] = ul
        
        Lx = np.trapz(ulen_x, c_samples)
        Ly = np.trapz(ulen_y, c_samples)
        return Lx * Ly

    # Seed
    def keich_pos(N):
        k = int(np.log2(N))
        j_terms = np.arange(1, k + 1)
        keich = np.empty(N, dtype=np.float64)
        for idx in range(N):
            eps = np.array(list(str(bin(idx))[2:]), dtype=int)
            eps = np.pad(eps, (k - len(eps), 0), "constant")
            keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / k)
        return keich

    keich_1d = keich_pos(N)
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    seed_pos = np.stack([keich_1d[i_flat], keich_1d[j_flat]], axis=1)
    
    t0 = time.time()
    DEADLINE = 620.0
    
    best_vol = volume_fast(seed_pos)
    best_pos = seed_pos.copy()
    
    # Differential Evolution
    try:
        from scipy.optimize import differential_evolution
        bounds = [(-2.0, 3.0)] * 128
        
        def objective_flat(v):
            return volume_fast(v.reshape(64, 2))
        
        if time.time() - t0 < DEADLINE - 30:
            result = differential_evolution(
                objective_flat,
                bounds=bounds,
                seed=42,
                maxiter=100,
                tol=1e-6,
                popsize=12,
                polish=True,
                init='sobol',
            )
            if result.fun < best_vol:
                best_vol = result.fun
                best_pos = result.x.reshape(64, 2)
    except Exception:
        pass
    
    # Coordinate descent polish
    if time.time() - t0 < DEADLINE - 5:
        pos = best_pos.copy()
        search_range = 0.3
        for it in range(30):
            if time.time() - t0 > DEADLINE - 3:
                break
            improved = False
            for k in range(128):
                old_val = pos[k // 2, k % 2]
                a, b = old_val - search_range, old_val + search_range
                def f_single(t):
                    new_pos = pos.copy()
                    new_pos[k // 2, k % 2] = t
                    return volume_fast(new_pos)
                # Golden section
                gr = (np.sqrt(5) - 1) / 2
                c = b - gr * (b - a)
                d = a + gr * (b - a)
                fc, fd = f_single(c), f_single(d)
                for _ in range(40):
                    if b - a < 1e-8:
                        break
                    if fc < fd:
                        b, d, fd = d, c, fc
                        c = b - gr * (b - a)
                        fc = f_single(c)
                    else:
                        a, c, fc = c, d, fc
                        d = a + gr * (b - a)
                        fd = f_single(d)
                opt_val = 0.5 * (a + b)
                new_vol = f_single(opt_val)
                if new_vol < best_vol - 1e-10:
                    pos[k // 2, k % 2] = opt_val
                    best_vol = new_vol
                    improved = True
            if not improved:
                search_range *= 0.5
                if search_range < 1e-5:
                    break
        best_pos = pos
    
    if not np.all(np.isfinite(best_pos)):
        best_pos = seed_pos
    
    return best_pos

# EVOLVE-BLOCK-END
