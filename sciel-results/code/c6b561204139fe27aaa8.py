# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_1d_union_length(x, c, N=8):
    """Compute |∪_{i=0}^{N-1} [x_i + c*i/N, x_i + c*i/N + 1/N]| at a single c"""
    starts = x + c * np.arange(N) / N
    ends = starts + 1.0 / N
    order = np.argsort(starts)
    s_sorted = starts[order]
    e_sorted = ends[order]
    union_len = 0.0
    cur_start = s_sorted[0]
    cur_end = e_sorted[0]
    for i in range(1, N):
        if s_sorted[i] > cur_end:
            union_len += cur_end - cur_start
            cur_start = s_sorted[i]
            cur_end = e_sorted[i]
        else:
            cur_end = max(cur_end, e_sorted[i])
    union_len += cur_end - cur_start
    return union_len

def _exact_1d_volume(x, N=8):
    """Compute ∫₀¹ f(c)² dc exactly where f(c) = |∪ᵢ [x_i + c*i/N, x_i + c*i/N + 1/N]|.
    f(c) is piecewise linear in c, so f(c)² is piecewise quadratic.
    Simpson's rule is exact for quadratics on each piece."""
    starts_coeffs = np.arange(N, dtype=np.float64) / N
    critical = [0.0, 1.0]
    for i in range(N):
        for j in range(i + 1, N):
            denom = starts_coeffs[i] - starts_coeffs[j]
            if abs(denom) > 1e-15:
                c_val = (x[j] - x[i]) / denom
                if 0.0 < c_val < 1.0:
                    critical.append(c_val)
    critical = sorted(set(critical))
    total = 0.0
    for k in range(len(critical) - 1):
        c0 = critical[k]
        c1 = critical[k + 1]
        c_mid = 0.5 * (c0 + c1)
        f0 = _compute_1d_union_length(x, c0, N)
        f1 = _compute_1d_union_length(x, c_mid, N)
        f2 = _compute_1d_union_length(x, c1, N)
        total += (c1 - c0) / 6.0 * (f0 * f0 + 4.0 * f1 * f1 + f2 * f2)
    return total

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    deadline = time.time() + 600.0

    # Candidate 1: Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)

    # Candidate 2: Symmetric placement (centered)
    symmetric = -np.arange(N, dtype=np.float64) / (2 * N)

    # Candidate 3: All at origin
    origin = np.zeros(N, dtype=np.float64)

    # Candidate 4: Linearly spaced
    linear = np.linspace(0, 1 - 1 / N, N, dtype=np.float64)

    candidates = [keich, symmetric, origin, linear]

    best_x = keich.copy()
    best_vol = float('inf')

    for x in candidates:
        vol = _exact_1d_volume(x, N)
        if vol < best_vol:
            best_vol = vol
            best_x = x.copy()

    # Local optimization with exact objective
    if time.time() < deadline - 180.0:
        from scipy.optimize import minimize

        def objective(x):
            return _exact_1d_volume(x, N)

        x0 = best_x.copy()
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 20000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_vol:
                best_vol = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

        # Try L-BFGS-B
        if time.time() < deadline - 120.0:
            try:
                res2 = minimize(objective, best_x, method='L-BFGS-B',
                               options={'maxiter': 2000, 'maxls': 100, 'ftol': 1e-14})
                if res2.fun < best_vol:
                    best_vol = res2.fun
                    best_x = res2.x.copy()
            except Exception:
                pass

        # Multiple random restarts
        if time.time() < deadline - 60.0:
            rng = np.random.default_rng(42)
            for _ in range(30):
                x_rand = rng.uniform(-1.0, 1.0, N)
                try:
                    res3 = minimize(objective, x_rand, method='Nelder-Mead',
                                   options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
                    if res3.fun < best_vol:
                        best_vol = res3.fun
                        best_x = res3.x.copy()
                except Exception:
                    pass
                if time.time() > deadline - 60.0:
                    break

    # Build the 64 positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    base_x = best_x[i_flat]
    base_y = best_x[j_flat]
    positions = np.stack([base_x, base_y], axis=1)

    return positions

# EVOLVE-BLOCK-END
