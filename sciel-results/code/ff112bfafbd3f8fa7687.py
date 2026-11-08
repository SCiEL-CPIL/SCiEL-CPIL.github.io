import numpy as np
import time
import scipy.optimize as opt

N = 50
M = 2 * N - 1  # 99
W = 1.0 / (M + 1)  # 1/100
TIME_LIMIT = 150.0
RNG = np.random.RandomState(50)


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
    y = np.zeros(m + 2)
    y[1:-1] = g
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(g) / (m + 1)
    norm_inf = np.max(g)
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)


def _value_neg(h):
    return -compute_value(h)


def _smooth_max(g, tau=0.01):
    g_max = np.max(g)
    if g_max <= 0:
        return 0.0
    shifted = g / g_max
    log_sum = np.log(np.sum(np.exp(shifted / tau - 1.0))) + np.log(g_max)
    return np.exp(log_sum)


def _value_neg_smooth(h, tau=0.01, peak_penalty=0.0):
    """Smooth objective. peak_penalty adds a term discouraging a sharp single peak."""
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(g) / (m + 1)
    norm_inf_smooth = _smooth_max(g, tau)
    if norm_1 <= 0 or norm_inf_smooth <= 0:
        return 0.0
    val = l2 / (norm_1 * norm_inf_smooth)
    # Peak-flattening penalty: reward mass spread across top-k of g
    if peak_penalty > 0:
        g_sorted = np.sort(g)[::-1]
        g_sum = np.sum(g)
        if g_sum > 0:
            top1 = g_sorted[0] / g_sum
            # penalty when top-1 dominates
            val = val - peak_penalty * (top1 - 1.0 / len(g)) ** 2
    return -val


def _generate_starts():
    x_grid = np.arange(N, dtype=float)
    starts = []

    # Uniform
    starts.append(np.ones(N))

    # Hann window
    hann = 0.5 * (1 + np.cos(2 * np.pi * x_grid / (N - 1)))
    starts.append(hann)

    # Cosine windows
    for width in [0.5, 0.7, 0.8, 0.9, 1.0]:
        arg = np.pi * width * (x_grid - (N - 1) / 2) / ((N - 1) / 2)
        h = np.cos(arg)
        starts.append(np.maximum(0.0, h))

    # Gaussian bumps
    for center in [10, 15, 20, 25, 30, 35, 40]:
        for width in [3, 5, 8, 12, 15, 20]:
            h = np.exp(-0.5 * ((x_grid - center) / width) ** 2)
            starts.append(h)

    # Two Gaussians
    for c1 in [8, 12, 15, 18]:
        for c2 in [32, 35, 38, 42]:
            for w in [3, 5, 8, 12]:
                h = np.exp(-0.5 * ((x_grid - c1) / w) ** 2) + \
                    np.exp(-0.5 * ((x_grid - c2) / w) ** 2)
                starts.append(h)

    # Power-law bumps
    for center in [15, 25, 35]:
        for r in [8, 12, 16, 20]:
            for p in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
                base = np.maximum(0.0, 1.0 - np.abs(x_grid - center) / r)
                h = base ** p
                starts.append(h)

    # Semicircle
    for center in [15, 25, 35]:
        for radius in [8, 12, 16, 20]:
            d = (x_grid - center) / radius
            h = np.sqrt(np.maximum(0.0, 1.0 - d ** 2))
            starts.append(h)

    # Triangle
    for center in [15, 25, 35]:
        for hw in [5, 10, 15, 20]:
            h = np.maximum(0.0, 1.0 - np.abs(x_grid - center) / hw)
            starts.append(h)

    # Sparse spikes (from findings)
    for k in [2, 3, 5]:
        for _ in range(3):
            idx = RNG.choice(N, size=k, replace=False)
            h = np.zeros(N)
            h[idx] = 1.0
            starts.append(h)

    # Comb / periodic
    for period in [4, 5, 7, 10]:
        h = (np.arange(N) % period < period // 2).astype(float)
        starts.append(h)

    return starts


def _coordinate_descent(h0, max_iter=2000):
    h = np.maximum(0.0, h0.copy())
    for _ in range(max_iter):
        if time.time() > _DEADLINE:
            break
        improved = False
        for i in range(N):
            current_val = h[i]
            candidates = [
                current_val * 0.5,
                current_val * 0.75,
                current_val * 1.25,
                current_val * 1.5,
                current_val + 0.05,
                current_val - 0.05,
                0.0,
                1.0,
            ]
            if current_val > 0:
                candidates.extend([current_val * (1 + RNG.normal(0, 0.1)) for _ in range(3)])
            candidates = [max(0.0, c) for c in candidates]

            best_cand = current_val
            best_cand_val = compute_value(h)

            for c in candidates:
                h[i] = c
                v = compute_value(h)
                if v > best_cand_val:
                    best_cand_val = v
                    best_cand = c

            h[i] = best_cand
            if best_cand != current_val:
                improved = True

        if not improved:
            break
    return h


def _simulated_annealing(h0, n_iter=5000):
    h = np.maximum(0.0, h0.copy())
    current_val = compute_value(h)
    best_local_h = h.copy()
    best_local_val = current_val
    T = 0.5
    cooling = 0.9995

    for _ in range(n_iter):
        if time.time() > _DEADLINE:
            break
        i = RNG.randint(N)
        delta = RNG.normal(0, 0.1)
        old_val = h[i]
        new_val = max(0.0, old_val + delta)
        h[i] = new_val

        new_val_score = compute_value(h)
        delta_val = new_val_score - current_val

        if delta_val > 0 or RNG.random() < np.exp(delta_val / max(T, 1e-10)):
            current_val = new_val_score
            if current_val > best_local_val:
                best_local_val = current_val
                best_local_h = h.copy()
        else:
            h[i] = old_val

        T *= cooling

    return best_local_h


_DEADLINE = 0.0


def construct():
    global _DEADLINE
    start_time = time.time()
    _DEADLINE = start_time + (TIME_LIMIT - 20.0)  # 20s safety margin

    best_val = 0.0
    best_h = np.ones(N)

    def track(h):
        nonlocal best_val, best_h
        h = np.maximum(0.0, h)
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()

    # ===== Phase 1: Generate and evaluate diverse structured starts =====
    starts = _generate_starts()
    for s in starts:
        s = np.maximum(0.0, s)
        if np.sum(s) > 0:
            track(s)

    # ===== Phase 2: Smooth gradient-based optimization from top starts =====
    scored = [(compute_value(np.maximum(0.0, s)), s) for s in starts]
    scored.sort(key=lambda t: -t[0])
    top_starts = [s for _, s in scored[:20]]

    bounds = [(0, None)] * N

    for s in top_starts:
        if time.time() > _DEADLINE:
            break
        s = np.maximum(0.0, s)
        for tau in [0.005, 0.01, 0.02, 0.05]:
            if time.time() > _DEADLINE:
                break
            for pp in [0.0, 0.5, 1.0, 2.0]:
                if time.time() > _DEADLINE:
                    break
                try:
                    res = opt.minimize(
                        lambda h: _value_neg_smooth(h, tau, pp),
                        s, method='L-BFGS-B', bounds=bounds,
                        options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10}
                    )
                    if res.x is not None:
                        track(res.x)
                except Exception:
                    pass

    # ===== Phase 3: Coordinate descent with peak-aware updates =====
    for s in top_starts[:10]:
        if time.time() > _DEADLINE:
            break
        s = np.maximum(0.0, s)
        h_cd = _coordinate_descent(s, max_iter=1000)
        track(h_cd)

    for _ in range(8):
        if time.time() > _DEADLINE:
            break
        pert = best_h + RNG.normal(0, 0.1, N)
        pert = np.maximum(0.0, pert)
        h_cd = _coordinate_descent(pert, max_iter=500)
        track(h_cd)

    # ===== Phase 4: Simulated annealing for global exploration =====
    for s in top_starts[:5]:
        if time.time() > _DEADLINE:
            break
        h_sa = _simulated_annealing(s, n_iter=5000)
        track(h_sa)

    # ===== Phase 5: Nelder-Mead polish on exact objective =====
    def _nm_optimize(h0):
        if time.time() > _DEADLINE:
            return
        h0 = np.maximum(0.0, h0)

        def obj(h):
            h = np.maximum(0.0, h)
            return _value_neg(h)

        try:
            res = opt.minimize(obj, h0, method='Nelder-Mead',
                               options={'maxiter': 800, 'xatol': 1e-8,
                                        'fatol': 1e-10, 'adaptive': True})
            if res.x is not None:
                track(res.x)
        except Exception:
            pass

    x_grid = np.arange(N, dtype=float)
    nm_starts = [
        np.ones(N),
        np.exp(-0.5 * ((x_grid - 25) / 8) ** 2),
        np.exp(-0.5 * ((x_grid - 15) / 5) ** 2) + np.exp(-0.5 * ((x_grid - 35) / 5) ** 2),
        best_h.copy(),
    ]
    for s in nm_starts:
        if time.time() > _DEADLINE:
            break
        _nm_optimize(s)

    # ===== Phase 6: Fine-tune with very small perturbations =====
    for _ in range(40):
        if time.time() > _DEADLINE:
            break
        for scale in [0.01, 0.05, 0.1]:
            if time.time() > _DEADLINE:
                break
            pert = best_h + RNG.normal(0, scale, N)
            pert = np.maximum(0.0, pert)
            v = compute_value(pert)
            if v > best_val:
                best_val = v
                best_h = pert.copy()
                # Local optimize from this improvement
                try:
                    res = opt.minimize(
                        _value_neg, best_h, method='L-BFGS-B', bounds=bounds,
                        options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
                    )
                    if res.x is not None:
                        track(res.x)
                except Exception:
                    pass

    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    return best_h.tolist()


if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
