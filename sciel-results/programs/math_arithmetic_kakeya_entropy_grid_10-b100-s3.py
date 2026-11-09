import time
import numpy as np
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximizes D/M where D = H(x-y), M = max(H(x), H(x+y), H(x+2y)).
    Uses smooth max surrogate, log-softmax parameterization, and multi-method optimization.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 590.0  # 40s safety margin from the 630s kill

    # Grid coordinates
    ys, xs = np.mgrid[0:n, 0:n]

    # Flatten for binning
    d_val = (xs - ys).flatten()       # x - y
    m0_val = xs.flatten()              # x
    m1_val = (xs + ys).flatten()       # x + y
    m2_val = (xs + 2 * ys).flatten()   # x + 2y

    # Unique values and inverse mapping
    d_vals, d_inv = np.unique(d_val, return_inverse=True)
    m0_vals, m0_inv = np.unique(m0_val, return_inverse=True)
    m1_vals, m1_inv = np.unique(m1_val, return_inverse=True)
    m2_vals, m2_inv = np.unique(m2_val, return_inverse=True)

    num_d = len(d_vals)
    num_m0 = len(m0_vals)
    num_m1 = len(m1_vals)
    num_m2 = len(m2_vals)

    def shannon(p_dist):
        p_nonzero = p_dist[p_dist > 1e-15]
        if len(p_nonzero) == 0:
            return 0.0
        return -np.sum(p_nonzero * np.log2(p_nonzero))

    def compute_entropies_from_p(p_flat):
        p_d = np.bincount(d_inv, weights=p_flat, minlength=num_d)
        p_m0 = np.bincount(m0_inv, weights=p_flat, minlength=num_m0)
        p_m1 = np.bincount(m1_inv, weights=p_flat, minlength=num_m1)
        p_m2 = np.bincount(m2_inv, weights=p_flat, minlength=num_m2)

        H_d = shannon(p_d)
        H_m0 = shannon(p_m0)
        H_m1 = shannon(p_m1)
        H_m2 = shannon(p_m2)

        M = max(H_m0, H_m1, H_m2)
        return H_d, M

    def get_score(p_flat):
        H_d, M = compute_entropies_from_p(p_flat)
        if M < 1e-9:
            return -1e9
        return H_d / M

    def log_softmax(z):
        z_shifted = z - np.max(z)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z)

    def objective_smooth(z, temperature=0.1):
        """Objective using smooth max surrogate and log-softmax parameterization"""
        p_flat = log_softmax(z)
        p_d = np.bincount(d_inv, weights=p_flat, minlength=num_d)
        p_m0 = np.bincount(m0_inv, weights=p_flat, minlength=num_m0)
        p_m1 = np.bincount(m1_inv, weights=p_flat, minlength=num_m1)
        p_m2 = np.bincount(m2_inv, weights=p_flat, minlength=num_m2)

        H_d = shannon(p_d)
        H_m0 = shannon(p_m0)
        H_m1 = shannon(p_m1)
        H_m2 = shannon(p_m2)

        # Smooth max using log-sum-exp
        entropies = np.array([H_m0, H_m1, H_m2])
        M_smooth = temperature * np.log(np.sum(np.exp(entropies / temperature)))
        
        if M_smooth < 1e-9:
            return 1e9
        return -H_d / M_smooth

    def objective(z):
        """Exact objective with log-softmax parameterization"""
        p_flat = log_softmax(z)
        H_d, M = compute_entropies_from_p(p_flat)
        if M < 1e-9:
            return 1e9
        return -H_d / M

    # --- Build structured starting points ---
    starts = []

    # 1. Uniform
    starts.append(np.zeros(100))  # log-softmax of zeros = uniform

    # 2. Single x-line
    for x0 in range(n):
        z = np.full(100, -100.0)
        z[x0 * 10:(x0 + 1) * 10] = 0.0
        starts.append(z)

    # 3. Single y-line
    for y0 in range(n):
        z = np.full(100, -100.0)
        z[y0 * 10:(y0 + 1) * 10] = 0.0
        starts.append(z)

    # 4. Main diagonal
    z_diag = np.full(100, -100.0)
    for i in range(n):
        z_diag[i * 10 + i] = 0.0
    starts.append(z_diag)

    # 5. Anti-diagonal
    z_adiag = np.full(100, -100.0)
    for i in range(n):
        z_adiag[i * 10 + (9 - i)] = 0.0
    starts.append(z_adiag)

    # 6. Concentrated on two x-values
    for x0 in range(n):
        for x1 in range(x0 + 1, n):
            z = np.full(100, -100.0)
            z[x0 * 10:(x0 + 1) * 10] = 0.0
            z[x1 * 10:(x1 + 1) * 10] = 0.0
            starts.append(z)

    # 7. Random Dirichlet converted to log-softmax
    rng = np.random.default_rng(42)
    for trial in range(50):
        alpha = rng.uniform(0.05, 5.0, 100)
        p_flat = rng.dirichlet(alpha)
        # Convert to log-space for log-softmax parameterization
        z = np.log(p_flat + 1e-15)
        starts.append(z)

    # 8. Checkerboard patterns
    for offset in range(2):
        z = np.full(100, -100.0)
        for y in range(n):
            for x in range(n):
                if (x + y) % 2 == offset:
                    z[y * 10 + x] = 0.0
        starts.append(z)

    # 9. Slope patterns: x + r*y = const for various r
    for r in [-2, -1, 0, 1, 2, 3]:
        for c in range(-10, 20):
            z = np.full(100, -100.0)
            for y in range(n):
                for x in range(n):
                    if x + r * y == c:
                        z[y * 10 + x] = 0.0
            if np.any(z > -50):
                starts.append(z)

    # --- Evaluate all starts and find best ---
    best_score = -1e9
    z_best = starts[0].copy()

    for z_flat in starts:
        p_flat = log_softmax(z_flat)
        score = get_score(p_flat)
        if score > best_score:
            best_score = score
            z_best = z_flat.copy()

    # --- Multi-phase optimization ---
    # Phase 1: Smooth optimization with annealing
    temperatures = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    max_smooth_iters = 300

    for temp in temperatures:
        if time.time() > deadline - 60.0:
            break

        # Optimize from current best and a few diverse starts
        candidates = [z_best.copy()]
        # Add some diverse starts
        for i in range(min(10, len(starts))):
            candidates.append(starts[i].copy())

        for z_init in candidates:
            if time.time() > deadline - 60.0:
                break
            try:
                res = minimize(
                    lambda z: objective_smooth(z, temperature=temp),
                    z_init,
                    method='Nelder-Mead',
                    options={'maxiter': max_smooth_iters, 'fatol': 1e-12, 'xatol': 1e-14}
                )
                p_res = log_softmax(res.x)
                score = get_score(p_res)
                if score > best_score:
                    best_score = score
                    z_best = res.x.copy()
            except Exception:
                pass

    # Phase 2: Exact optimization with SLSQP (using probability space)
    bounds_p = [(0.0, 1.0) for _ in range(100)]
    constraints_p = [{'type': 'eq', 'fun': lambda p: np.sum(p) - 1.0}]

    max_slsqp = 30
    slsqp_count = 0

    # Convert best z to p
    p_best = log_softmax(z_best)

    # Also try from diverse starts in probability space
    p_starts = [log_softmax(z) for z in starts[:20]]
    p_starts.append(p_best)

    for p_init in p_starts:
        if time.time() > deadline - 30.0:
            break
        if slsqp_count >= max_slsqp:
            break

        try:
            res = minimize(
                lambda p: -get_score(p),
                p_init,
                method='SLSQP',
                bounds=bounds_p,
                constraints=constraints_p,
                options={'maxiter': 500, 'ftol': 1e-15}
            )
            p_res = np.clip(res.x, 0, None)
            s = np.sum(p_res)
            if s > 1e-9:
                p_res /= s
            score = get_score(p_res)
            if score > best_score:
                best_score = score
                p_best = p_res.copy()
                z_best = np.log(p_best + 1e-15)
            slsqp_count += 1
        except Exception:
            pass

    # Phase 3: High-precision Nelder-Mead polish
    if time.time() < deadline - 15.0:
        try:
            res_nm = minimize(
                lambda z: objective(z),
                z_best,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'fatol': 1e-15, 'xatol': 1e-16}
            )
            p_nm = log_softmax(res_nm.x)
            score_nm = get_score(p_nm)
            if score_nm > best_score:
                best_score = score_nm
                p_best = p_nm.copy()
        except Exception:
            pass

    # Phase 4: Final tight SLSQP polish
    if time.time() < deadline - 10.0:
        try:
            res_final = minimize(
                lambda p: -get_score(p),
                p_best,
                method='SLSQP',
                bounds=bounds_p,
                constraints=constraints_p,
                options={'maxiter': 1000, 'ftol': 1e-16}
            )
            p_final = np.clip(res_final.x, 0, None)
            s = np.sum(p_final)
            if s > 1e-9:
                p_final /= s
            score_final = get_score(p_final)
            if score_final > best_score:
                best_score = score_final
                p_best = p_final.copy()
        except Exception:
            pass

    # --- Final validation and normalization ---
    P = p_best.reshape((n, n))
    P = np.clip(P, 0.0, None)
    s = np.sum(P)
    if s > 1e-9:
        P = P / s
    else:
        P = np.ones((n, n)) / (n * n)

    return P
