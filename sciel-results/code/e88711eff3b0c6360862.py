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


def _smooth_value(h, beta=80.0):
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


def _epigraph_objective(params, beta=100.0):
    """
    params: [h_0, ..., h_49, t]
    h: 50 heights, t: upper bound on max(g)
    Objective: -l2/(norm_1 * t) + penalty * sum(max(0, g_i - t)^2)
    """
    h = np.maximum(0.0, params[:N])
    t = params[N]
    if t <= 0:
        t = 1e-6
    
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(g) / (m + 1)
    if norm_1 <= 0:
        return 1e10
    
    # Penalty for g_i > t
    violation = np.maximum(0.0, g - t)
    penalty = 100.0 * np.sum(violation ** 2)
    
    return -(l2 / (norm_1 * t)) + penalty


def _generate_starts(best_h, n_starts=10):
    """Generate diverse starting points for optimization."""
    starts = [best_h.copy()]
    
    # Plateau-based starts with varying widths and positions
    for k in [5, 10, 15, 20, 25, 30, 40]:
        for j in [0, N//4, N//2, 3*N//4]:
            if j + k <= N:
                h = _plateau(j, k)
                # Add small noise to break symmetry
                h = np.maximum(0.0, h + RNG.normal(0, 0.02, N))
                starts.append(h)
    
    # Bimodal plateaus
    for k1 in [8, 12]:
        for k2 in [8, 12]:
            for gap in [5, 10]:
                h = np.zeros(N)
                h[:k1] = 1.0
                h[N-k2:] = 1.0
                if gap > 0 and k1 + gap < N - k2:
                    h[k1:N-k2] = 0.3
                h = np.maximum(0.0, h + RNG.normal(0, 0.02, N))
                starts.append(h)
    
    # Smooth bumps (Gaussian-like)
    for center in [N//4, N//2, 3*N//4]:
        for width in [5, 10, 15]:
            x = np.arange(N)
            h = np.exp(-((x - center)**2) / (2*width**2))
            h = np.maximum(0.0, h + RNG.normal(0, 0.01, N))
            starts.append(h)
    
    # Random smooth profiles
    for _ in range(5):
        h = np.abs(RNG.randn(N))
        h = h / np.max(h) if np.max(h) > 0 else h
        h = np.maximum(0.0, h)
        starts.append(h)
    
    return starts


def construct():
    """Construct 50 nonnegative step heights to maximize the value."""
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 20.0)

    best_val = 0.0
    best_h = np.ones(N)

    def track(h):
        nonlocal best_val, best_h
        h = np.maximum(0.0, h)
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()

    # --- Phase 1: exact analytic sweep over plateau widths -----------------
    for k in range(1, N + 1):
        h = _plateau(0, k)
        track(h)

    # Also try some shifted plateaus for diversity
    for k in [10, 15, 20, 25, 30]:
        for j in [0, N//4, N//2, 3*N//4]:
            if j + k <= N:
                h = _plateau(j, k)
                track(h)

    # --- Phase 2: Epigraph formulation with penalty for peak constraint ----
    # We use an auxiliary variable t to represent max(g), with a penalty
    # for violating g <= t, combined with the smooth surrogate.
    
    if time.time() < deadline:
        # Generate diverse starts
        starts = _generate_starts(best_h, n_starts=20)
        
        for h_start in starts[:15]:  # Limit to 15 starts to save time
            if time.time() > deadline:
                break
            
            g_start = np.convolve(h_start, h_start)
            t_start = np.max(g_start) + 0.1
            
            x0 = np.concatenate([h_start, [t_start]])
            bounds = [(0, None)] * N + [(1e-4, None)]
            
            try:
                res = opt.minimize(
                    lambda x: _epigraph_objective(x, beta=100.0),
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                res_h = np.maximum(0.0, res.x[:N])
                track(res_h)
            except Exception:
                pass

    # --- Phase 3: L-BFGS-B on smooth surrogate from best -------------------
    if time.time() < deadline:
        # Use progressive beta for better convergence
        for beta in [50.0, 100.0, 200.0, 400.0]:
            if time.time() > deadline:
                break
            
            for _ in range(5):
                if time.time() > deadline:
                    break
                pert = np.maximum(0.0, best_h + RNG.normal(0.0, 0.04, N))
                try:
                    res = opt.minimize(
                        lambda x, b=beta: _smooth_value(x, b),
                        pert,
                        method='L-BFGS-B',
                        bounds=[(0, None)] * N,
                        options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10}
                    )
                    track(res.x)
                except Exception:
                    pass

    # --- Phase 4: Nelder-Mead polish on exact objective ---------------------
    if time.time() < deadline:
        # Use multiple restarts with varying perturbation scales
        for scale in [0.08, 0.05, 0.03]:
            for _ in range(8):
                if time.time() > deadline:
                    break
                pert = best_h + RNG.normal(0.0, scale, N)
                pert = np.maximum(0.0, pert)
                try:
                    res = opt.minimize(
                        _neg_exact_value,
                        pert,
                        method='Nelder-Mead',
                        options={'maxiter': 2000, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True}
                    )
                    track(res.x)
                except Exception:
                    pass

    # --- Phase 5: High-precision final polish --------------------------------
    if time.time() < deadline - 5.0:
        for _ in range(8):
            if time.time() > deadline - 5.0:
                break
            pert = best_h + RNG.normal(0.0, 0.01, N)
            pert = np.maximum(0.0, pert)
            try:
                res = opt.minimize(
                    _neg_exact_value,
                    pert,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
                )
                track(res.x)
            except Exception:
                pass

    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    return best_h.tolist()


if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
