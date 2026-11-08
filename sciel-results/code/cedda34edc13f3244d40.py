"""
DCT-basis representation with epigraph reformulation and diverse optimization.

Key strategy:
1. Represent h as a superposition of a small number of DCT basis functions
   (low-frequency components only) to enforce global smoothness.
2. Use an epigraph/penalty formulation to handle the norm_inf constraint.
3. Combine structured family starts, gradient-based optimization, and
   local search on the full 50-dim space.
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
M = 2 * N - 1  # 99
W = 1.0 / (M + 1)  # 1/100
TIME_LIMIT = 150.0
RNG = np.random.RandomState(50)


def _build_y(g):
    y = np.zeros(M + 2)
    y[1:-1] = g
    return y


def compute_value(h):
    h = np.asarray(h, dtype=float).ravel()
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if not np.all(np.isfinite(h)) or np.any(h < -1e-12):
        return 0.0
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)


def _value_neg(h):
    return -compute_value(h)


def _smooth_max(g, temperature=0.01):
    """Smooth approximation of max(g) using log-sum-exp."""
    g = np.asarray(g, dtype=float)
    if g.size == 0:
        return 0.0
    max_g = np.max(g)
    if max_g <= 0:
        return 0.0
    # Log-sum-exp for stability
    shifted = g / max_g
    log_sum_exp = np.log(np.sum(np.exp(shifted - 1.0))) + 1.0
    return max_g * np.exp(log_sum_exp - 1.0) / np.exp(0)  # This is just max_g
    # Actually, let's use a proper smooth max:
    # smooth_max = max_g * (1 + (1/temperature) * log(mean(exp((g - max_g)/temperature))))
    # But for our purposes, we can use a simpler approach


def _value_neg_smooth(h, temperature=0.05):
    """Smoothed version of the value function for gradient-based optimization."""
    h = np.maximum(0.0, np.asarray(h, dtype=float))
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(g) / (m + 1)
    
    # Smooth max using log-sum-exp
    if np.max(g) <= 0:
        return 0.0
    # Use a soft max: T * log(sum(exp(g/T)))
    T = temperature
    shifted = g / T
    max_shifted = np.max(shifted)
    log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    smooth_norm_inf = T * log_sum_exp
    
    if norm_1 <= 0 or smooth_norm_inf <= 0:
        return 0.0
    return -(l2 / (norm_1 * smooth_norm_inf))


def _dct_basis(N, K):
    """Generate K DCT-II basis vectors of length N."""
    basis = np.zeros((K, N))
    n = np.arange(N, dtype=float)
    for k in range(K):
        if k == 0:
            basis[k, :] = 1.0 / np.sqrt(N)
        else:
            basis[k, :] = np.sqrt(2.0 / N) * np.cos(np.pi * (2 * n + 1) * k / (2 * N))
    return basis


def _dct_to_h(coeffs, basis):
    """Reconstruct h from DCT coefficients, ensuring nonnegativity."""
    h = basis.T @ coeffs
    return np.maximum(0.0, h)


def construct():
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 20.0)  # 20s safety margin

    best_val = 0.0
    best_h = np.ones(N)

    def track(h):
        nonlocal best_val, best_h
        h = np.maximum(0.0, h)
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()

    x_grid = np.arange(N, dtype=float)

    # ===== Phase 1: Generate diverse structured starts =====
    structured_starts = []

    # Uniform
    structured_starts.append(np.ones(N))

    # Gaussian bumps
    for center in [10, 15, 20, 25, 30, 35, 40]:
        for width in [2, 3, 5, 8, 12, 20]:
            h = np.exp(-0.5 * ((x_grid - center) / width) ** 2)
            structured_starts.append(h)

    # Two Gaussian bumps
    for c1 in [8, 12, 15, 18]:
        for c2 in [32, 35, 38, 42]:
            for w in [3, 5, 8]:
                h = np.exp(-0.5 * ((x_grid - c1) / w) ** 2) + \
                    np.exp(-0.5 * ((x_grid - c2) / w) ** 2)
                structured_starts.append(h)

    # Cosine windows
    for center in [15, 25, 35]:
        for width in [10, 15, 20, 25]:
            arg = np.pi * (x_grid - center) / width
            h = np.cos(arg)
            structured_starts.append(np.maximum(0.0, h))

    # Triangle
    for center in [15, 25, 35]:
        for hw in [5, 10, 15, 20]:
            h = np.maximum(0.0, 1.0 - np.abs(x_grid - center) / hw)
            structured_starts.append(h)

    # Semicircle
    for center in [15, 25, 35]:
        for radius in [8, 12, 16, 20]:
            d = (x_grid - center) / radius
            h = np.sqrt(np.maximum(0.0, 1.0 - d ** 2))
            structured_starts.append(h)

    # Power-law
    for center in [15, 25, 35]:
        for r in [10, 15, 20]:
            for p in [0.5, 1.0, 1.5, 2.0, 3.0]:
                base = np.maximum(0.0, 1.0 - np.abs(x_grid - center) / r)
                h = base ** p
                structured_starts.append(h)

    # Sparse spikes
    for num_spikes in [2, 3, 4, 5]:
        for _ in range(5):
            positions = RNG.randint(0, N, size=num_spikes)
            h = np.zeros(N)
            for pos in positions:
                h[pos] = 1.0
                if pos > 0:
                    h[pos - 1] = 0.5
                if pos < N - 1:
                    h[pos + 1] = 0.5
            structured_starts.append(h)

    # Evaluate all structured starts
    for s in structured_starts:
        s = np.maximum(0.0, s)
        if np.sum(s) > 0:
            track(s)

    # ===== Phase 2: DCT-basis optimization =====
    # Use a small number of DCT basis functions to enforce smoothness
    K_values = [5, 8, 10, 12, 15]
    for K in K_values:
        if time.time() > deadline:
            break
        basis = _dct_basis(N, K)
        
        # Initialize coefficients from best_h
        coeffs_init = basis @ best_h
        
        def _dct_obj(coeffs):
            if time.time() > deadline:
                return 0.0
            h = _dct_to_h(coeffs, basis)
            return _value_neg(h)
        
        # Optimize DCT coefficients
        try:
            res = opt.minimize(_dct_obj, coeffs_init, method='L-BFGS-B',
                             bounds=[(0, None)] * K,
                             options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.x is not None:
                h = _dct_to_h(res.x, basis)
                track(h)
        except Exception:
            pass
        
        # Also try Nelder-Mead on DCT coefficients
        try:
            res = opt.minimize(_dct_obj, coeffs_init, method='Nelder-Mead',
                             options={'maxiter': 800, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.x is not None:
                h = _dct_to_h(res.x, basis)
                track(h)
        except Exception:
            pass

    # ===== Phase 3: Local search on full 50-dim h =====
    bounds = [(0, None)] * N

    def _local_optimize(h0, max_iter=300):
        if time.time() > deadline:
            return
        h0 = np.maximum(0.0, h0)
        try:
            res = opt.minimize(
                _value_neg, h0, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.x is not None:
                track(res.x)
        except Exception:
            pass

    # Local optimize from best structured starts
    scored = [(compute_value(np.maximum(0.0, s)), s) for s in structured_starts]
    scored.sort(key=lambda t: -t[0])
    top_starts = [s for _, s in scored[:20]]

    for s in top_starts:
        if time.time() > deadline:
            break
        _local_optimize(s)

    # Also local optimize from current best with perturbations
    for _ in range(60):
        if time.time() > deadline:
            break
        pert = best_h + RNG.normal(0, 0.15, N)
        pert = np.maximum(0.0, pert)
        _local_optimize(pert, max_iter=200)

    # ===== Phase 4: Nelder-Mead on full h =====
    def _nm_optimize(h0):
        if time.time() > deadline:
            return
        h0 = np.maximum(0.0, h0)

        def obj(h):
            h = np.maximum(0.0, h)
            return _value_neg(h)

        try:
            res = opt.minimize(obj, h0, method='Nelder-Mead',
                               options={'maxiter': 1500, 'xatol': 1e-9,
                                        'fatol': 1e-11, 'adaptive': True})
            if res.x is not None:
                track(res.x)
        except Exception:
            pass

    # Nelder-Mead from diverse starts
    nm_starts = [
        np.ones(N),
        np.exp(-0.5 * ((x_grid - 25) / 8) ** 2),
        np.exp(-0.5 * ((x_grid - 15) / 5) ** 2) + np.exp(-0.5 * ((x_grid - 35) / 5) ** 2),
        best_h.copy(),
    ]
    for s in nm_starts:
        if time.time() > deadline:
            break
        _nm_optimize(s)

    # ===== Phase 5: Fine-tuning with small perturbations =====
    for _ in range(80):
        if time.time() > deadline:
            break
        for scale in [0.005, 0.01, 0.02, 0.05]:
            if time.time() > deadline:
                break
            pert = best_h + RNG.normal(0, scale, N)
            pert = np.maximum(0.0, pert)
            v = compute_value(pert)
            if v > best_val:
                best_val = v
                best_h = pert.copy()
                _local_optimize(pert, max_iter=150)

    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    return best_h.tolist()


if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
