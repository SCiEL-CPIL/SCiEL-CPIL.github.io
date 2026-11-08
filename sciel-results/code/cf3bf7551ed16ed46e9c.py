"""Combined approach: plateau structure + epigraph/L-BFGS-B smoothing + Nelder-Mead polish."""
import numpy as np
import time
import scipy.optimize as opt

N = 50
M = 2 * N - 1                # len(g) = 99
W = 1.0 / (M + 1)            # 1/100
TIME_LIMIT = 150.0
RNG = np.random.RandomState(50)


def _build_y(g):
    y = np.zeros(M + 2)
    y[1:-1] = g
    return y


def compute_value(h):
    """Exact evaluator metric for given heights."""
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


def _neg_exact_value(h):
    h = np.maximum(0.0, h)
    return -compute_value(h)


def _plateau(j, k):
    h = np.zeros(N)
    end = min(N, j + k)
    if end > j:
        h[j:end] = 1.0
    return h


def _smooth_value(h, beta=200.0):
    """Smooth surrogate using log-sum-exp for max(g)."""
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(g) / (m + 1)
    if norm_1 <= 0:
        return 0.0
    g_max = np.max(g)
    if g_max <= 0:
        return 0.0
    lse = g_max + (1.0 / beta) * np.log(np.sum(np.exp(beta * (g - g_max))))
    if lse <= 0:
        return 0.0
    return -(l2 / (norm_1 * lse))


def _epigraph_objective(params, beta=200.0):
    """
    Epigraph formulation: minimize -l2/(norm_1 * t) subject to t >= max(g).
    We use a penalty approach: objective = -l2/(norm_1 * t) + lambda * sum(relu(g_i - t))
    params = [h_0, ..., h_49, t]
    """
    h = params[:N]
    t = params[N]
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(g) / (m + 1)
    if norm_1 <= 0:
        return 1e6
    if t <= 0:
        return 1e6
    
    # Penalty for violation of t >= max(g)
    violation = np.maximum(0.0, g - t)
    penalty = 100.0 * np.sum(violation)
    
    val = l2 / (norm_1 * t)
    return -val + penalty


def construct():
    """Construct 50 nonnegative step heights to maximize the value."""
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 25.0)  # Safety margin

    best_val = 0.0
    best_h = np.ones(N)

    def track(h):
        nonlocal best_val, best_h
        h = np.maximum(0.0, h)
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()

    # --- Phase 1: Diversified initialization --------------------------------
    # Plateau sweep
    for k in range(1, N + 1):
        h = _plateau(0, k)
        track(h)

    # Shifted plateaus
    for k in [5, 10, 15, 20, 25, 30, 35, 40]:
        for j in range(0, N - k + 1, max(1, (N - k) // 8)):
            h = _plateau(j, k)
            track(h)

    # Random starts
    for _ in range(30):
        h = np.abs(RNG.normal(0.5, 0.3, N))
        track(h)

    # Structured patterns: cosine, linear, etc.
    x = np.linspace(0, 1, N)
    patterns = [
        np.cos(np.pi * x) ** 2,
        np.sin(np.pi * x),
        1.0 - x,
        x,
        np.exp(-10 * (x - 0.5) ** 2),
        np.exp(-20 * (x - 0.3) ** 2),
        np.exp(-20 * (x - 0.7) ** 2),
    ]
    for p in patterns:
        h = np.maximum(0.0, p)
        track(h)

    # --- Phase 2: Multi-start L-BFGS-B on smooth surrogate ------------------
    # Use multiple perturbation scales to escape local optima
    pert_scales = [0.02, 0.04, 0.06, 0.08, 0.10]
    num_starts = 40

    if time.time() < deadline:
        for scale in pert_scales:
            for _ in range(num_starts // len(pert_scales)):
                if time.time() > deadline - 30.0:
                    break
                pert = np.maximum(0.0, best_h + RNG.normal(0.0, scale, N))
                try:
                    res = opt.minimize(
                        lambda x: _smooth_value(x, 200.0),
                        pert,
                        method='L-BFGS-B',
                        bounds=[(0, None)] * N,
                        options={'maxiter': 600, 'ftol': 1e-15, 'gtol': 1e-12}
                    )
                    track(res.x)
                except Exception:
                    pass

    # --- Phase 2b: Epigraph formulation with L-BFGS-B -----------------------
    # Add an auxiliary variable t for the peak constraint
    if time.time() < deadline - 30.0:
        for _ in range(15):
            if time.time() > deadline - 30.0:
                break
            pert_h = np.maximum(0.0, best_h + RNG.normal(0.0, 0.05, N))
            g = np.convolve(pert_h, pert_h)
            t_init = np.max(g) * 1.1
            params = np.concatenate([pert_h, [t_init]])
            try:
                res = opt.minimize(
                    _epigraph_objective,
                    params,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N + [(1e-6, None)],
                    options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                track(res.x[:N])
            except Exception:
                pass

    # --- Phase 3: Nelder-Mead polish on exact objective ---------------------
    if time.time() < deadline:
        for _ in range(35):
            if time.time() > deadline - 20.0:
                break
            scale = 0.03 + 0.04 * RNG.rand()
            pert = best_h + RNG.normal(0.0, scale, N)
            pert = np.maximum(0.0, pert)
            try:
                res = opt.minimize(
                    _neg_exact_value,
                    pert,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                track(res.x)
            except Exception:
                pass

    # --- Phase 4: High-precision final polish --------------------------------
    # Use very tight tolerances and more iterations for the final few candidates
    if time.time() < deadline - 10.0:
        for _ in range(8):
            if time.time() > deadline - 10.0:
                break
            pert = best_h + RNG.normal(0.0, 0.01, N)
            pert = np.maximum(0.0, pert)
            try:
                res = opt.minimize(
                    _neg_exact_value,
                    pert,
                    method='Nelder-Mead',
                    options={'maxiter': 8000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True}
                )
                track(res.x)
            except Exception:
                pass

    # --- Phase 5: Coordinate descent refinement ------------------------------
    # Try small coordinate-wise adjustments for final polish
    if time.time() < deadline - 5.0:
        for _ in range(200):
            if time.time() > deadline - 5.0:
                break
            idx = RNG.randint(N)
            delta = RNG.uniform(-0.02, 0.02)
            trial = best_h.copy()
            trial[idx] = max(0.0, trial[idx] + delta)
            v = compute_value(trial)
            if v > best_val:
                best_val = v
                best_h = trial.copy()
            # Try shrinking the peak
            g = np.convolve(best_h, best_h)
            peak_idx = np.argmax(g)
            # Find which h indices contribute to g[peak_idx]
            # g[k] = sum_i h[i]*h[i+k], so for peak at k, adjust h around that
            for offset in range(-3, 4):
                i = peak_idx + offset
                if 0 <= i < N:
                    trial = best_h.copy()
                    trial[i] = max(0.0, trial[i] * 0.99)
                    v = compute_value(trial)
                    if v > best_val:
                        best_val = v
                        best_h = trial.copy()

    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    return best_h.tolist()


if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
