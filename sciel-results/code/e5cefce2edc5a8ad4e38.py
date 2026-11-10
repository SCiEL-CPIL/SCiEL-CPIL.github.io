import numpy as np
import scipy.optimize as so
import time

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    delta = 1.0 / N
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)

    X0 = keich.copy()
    Y0 = keich.copy()

    def compute_volume(X, Y):
        # X, Y are arrays of length N
        # Build arrays of A and B for X and Y
        A_x = np.empty(2 * N)
        B_x = np.empty(2 * N)
        is_r_x = np.empty(2 * N, dtype=int)
        for i in range(N):
            A_x[2*i] = X[i]
            B_x[2*i] = i / N
            is_r_x[2*i] = 0
            A_x[2*i+1] = X[i] + delta
            B_x[2*i+1] = i / N
            is_r_x[2*i+1] = 1
            
        A_y = np.empty(2 * N)
        B_y = np.empty(2 * N)
        is_r_y = np.empty(2 * N, dtype=int)
        for j in range(N):
            A_y[2*j] = Y[j]
            B_y[2*j] = j / N
            is_r_y[2*j] = 0
            A_y[2*j+1] = Y[j] + delta
            B_y[2*j+1] = j / N
            is_r_y[2*j+1] = 1

        # Find critical points for X
        crit_x = {0.0, 1.0}
        for i in range(2 * N):
            for j in range(i + 1, 2 * N):
                if B_x[i] != B_x[j]:
                    c = (A_x[j] - A_x[i]) / (B_x[i] - B_x[j])
                    if 0.0 < c < 1.0:
                        crit_x.add(c)
                        
        # Find critical points for Y
        crit_y = {0.0, 1.0}
        for i in range(2 * N):
            for j in range(i + 1, 2 * N):
                if B_y[i] != B_y[j]:
                    c = (A_y[j] - A_y[i]) / (B_y[i] - B_y[j])
                    if 0.0 < c < 1.0:
                        crit_y.add(c)
                        
        all_crit = sorted(crit_x.union(crit_y))
        
        def calc_union_length_vec(A, B, is_r, c):
            vals = A + B * c
            order = np.argsort(vals)
            sorted_vals = vals[order]
            sorted_is_r = is_r[order]
            
            # Vectorized sweep line
            # count changes: +1 for left, -1 for right
            count_changes = np.where(sorted_is_r == 0, 1, -1)
            running_count = np.cumsum(count_changes)
            
            # Total length = sum of (val[k] - val[k-1]) where count goes from 0 to 1 at k-1 and stays >=1
            # Actually: total = sum over intervals where count > 0
            # Interval k has length sorted_vals[k+1] - sorted_vals[k] and is covered if running_count[k] > 0
            # Wait, running_count[k] is the count after processing point k
            # The interval between sorted_vals[k] and sorted_vals[k+1] is covered if the count during that interval > 0
            # The count during interval [k, k+1] is running_count[k] (after adding/removing at k)
            # But we need count > 0 during the open interval (sorted_vals[k], sorted_vals[k+1])
            # which equals running_count[k]
            
            # Actually let me think more carefully:
            # At each point, we add or remove an interval endpoint
            # The coverage count between consecutive sorted points is the running count
            # Total covered length = sum_{k=0}^{n-2} (sorted_vals[k+1] - sorted_vals[k]) * I(running_count[k] > 0)
            
            if len(sorted_vals) < 2:
                return 0.0
            
            diffs = np.diff(sorted_vals)
            # running_count has length n, but we only need first n-1 for intervals
            covered = (running_count[:-1] > 0).astype(np.float64)
            return np.sum(diffs * covered)

        total_vol = 0.0
        for k in range(len(all_crit) - 1):
            c_start = all_crit[k]
            c_end = all_crit[k + 1]
            h = c_end - c_start
            if h < 1e-12:
                continue

            Lx_s = calc_union_length_vec(A_x, B_x, is_r_x, c_start)
            Lx_e = calc_union_length_vec(A_x, B_x, is_r_x, c_end)
            Ly_s = calc_union_length_vec(A_y, B_y, is_r_y, c_start)
            Ly_e = calc_union_length_vec(A_y, B_y, is_r_y, c_end)

            dx = Lx_e - Lx_s
            dy = Ly_e - Ly_s

            term = Lx_s * Ly_s
            term += 0.5 * (Lx_s * dy + Ly_s * dx)
            term += (1.0 / 3.0) * dx * dy

            total_vol += h * term

        return total_vol

    def objective(params):
        X = params[:N]
        Y = params[N:]
        return compute_volume(X, Y)

    p0 = np.concatenate([X0, Y0])

    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s kill

    best_p = p0.copy()
    best_v = objective(p0)

    # Use Differential Evolution for global search
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    try:
        res_de = so.differential_evolution(objective, bounds, maxiter=200, popsize=25, seed=42, polish=False, tol=1e-10)
        if time.time() < deadline:
            if res_de.fun < best_v:
                best_v = res_de.fun
                best_p = res_de.x.copy()
    except Exception:
        pass

    # Multi-start local refinement
    rng_local = np.random.default_rng(99)
    local_starts = [best_p.copy()]
    for _ in range(3):
        if time.time() > deadline - 120:
            break
        # Perturb best solution
        perturbation = rng_local.uniform(-0.1, 0.1, size=2 * N)
        candidate = best_p + perturbation
        candidate = np.clip(candidate, -1.0, 1.0)
        local_starts.append(candidate)
    
    for start_idx, p_start in enumerate(local_starts):
        if time.time() > deadline - 60:
            break
        
        # Powell
        if time.time() < deadline - 60:
            try:
                res_pw = so.minimize(objective, p_start, method='Powell', options={'maxiter': 3000, 'xtol': 1e-12, 'ftol': 1e-12})
                if res_pw.fun < best_v:
                    best_v = res_pw.fun
                    best_p = res_pw.x.copy()
            except Exception:
                pass
        
        # L-BFGS-B
        if time.time() < deadline - 30:
            try:
                res_lbfgs = so.minimize(objective, best_p, method='L-BFGS-B', options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-15})
                if res_lbfgs.fun < best_v:
                    best_v = res_lbfgs.fun
                    best_p = res_lbfgs.x.copy()
            except Exception:
                pass
        
        # Nelder-Mead
        if time.time() < deadline - 15:
            try:
                res_nm = so.minimize(objective, best_p, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
                if res_nm.fun < best_v:
                    best_v = res_nm.fun
                    best_p = res_nm.x.copy()
            except Exception:
                pass

    # Final high-precision polish
    if time.time() < deadline - 10:
        try:
            res_final = so.minimize(objective, best_p, method='L-BFGS-B', options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-16})
            if res_final.fun < best_v:
                best_v = res_final.fun
                best_p = res_final.x.copy()
        except Exception:
            pass

    X_opt = best_p[:N]
    Y_opt = best_p[N:]

    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    base_x = X_opt[i_flat]
    base_y = Y_opt[j_flat]
    positions = np.stack([base_x, base_y], axis=1)

    return positions
