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

    def build_eps(X):
        eps = []
        for i in range(N):
            eps.append((X[i], i / N, 0))
            eps.append((X[i] + delta, i / N, 1))
        return eps

    def calc_union_length_typed(eps, c):
        pts = []
        for A, B, is_r in eps:
            val = A + B * c
            pts.append((val, is_r))
        pts.sort()
        count = 0
        start_pos = 0.0
        total = 0.0
        for val, is_r in pts:
            if is_r == 0:
                if count == 0:
                    start_pos = val
                count += 1
            else:
                count -= 1
                if count == 0:
                    total += val - start_pos
        return total

    def compute_volume(X, Y):
        eps_x = build_eps(X)
        eps_y = build_eps(Y)

        crit_x = {0.0, 1.0}
        for i in range(len(eps_x)):
            for j in range(i + 1, len(eps_x)):
                A1, B1, _ = eps_x[i]
                A2, B2, _ = eps_x[j]
                if B1 != B2:
                    c = (A2 - A1) / (B1 - B2)
                    if 0.0 < c < 1.0:
                        crit_x.add(c)

        crit_y = {0.0, 1.0}
        for i in range(len(eps_y)):
            for j in range(i + 1, len(eps_y)):
                A1, B1, _ = eps_y[i]
                A2, B2, _ = eps_y[j]
                if B1 != B2:
                    c = (A2 - A1) / (B1 - B2)
                    if 0.0 < c < 1.0:
                        crit_y.add(c)

        all_crit = sorted(crit_x.union(crit_y))

        total_vol = 0.0
        for k in range(len(all_crit) - 1):
            c_start = all_crit[k]
            c_end = all_crit[k + 1]
            h = c_end - c_start
            if h < 1e-12:
                continue

            Lx_s = calc_union_length_typed(eps_x, c_start)
            Lx_e = calc_union_length_typed(eps_x, c_end)
            Ly_s = calc_union_length_typed(eps_y, c_start)
            Ly_e = calc_union_length_typed(eps_y, c_end)

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
    deadline = start_time + 500.0

    best_p = p0.copy()
    best_v = objective(p0)

    # Multi-restart Powell with time guard
    np.random.seed(42)
    n_restarts = 10
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        if r == 0:
            p_init = p0.copy()
        else:
            p_init = p0 + np.random.normal(0, 0.05, 2 * N)

        try:
            res = so.minimize(objective, p_init, method='Powell',
                              options={'maxiter': 300, 'xtol': 1e-8, 'ftol': 1e-8})
            if res.fun < best_v:
                best_v = res.fun
                best_p = res.x.copy()
        except Exception:
            pass

    # Additional Nelder-Mead polish on the best result
    if time.time() < deadline - 60:
        try:
            res2 = so.minimize(objective, best_p, method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-9})
            if res2.fun < best_v:
                best_v = res2.fun
                best_p = res2.x.copy()
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
