"""
Optimization of the second autocorrelation inequality constant C2.
Uses L-BFGS-B with a log-sum-exp smoothed objective for local refinement,
progressive beta strategy, and extensive polish phase.
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
M = 2 * N - 1          # len(g) = 99
W = 1.0 / (M + 1)      # 1/100
TIME_LIMIT = 149.0     # safety margin before 150s kill
RNG = np.random.RandomState(50)

def _build_y(g):
    y = np.zeros(M + 2)
    y[1:-1] = g
    return y

def compute_value(h):
    """Exact evaluator metric for given heights (defensive)."""
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

def _smooth_value(h, beta=100.0):
    """
    Smooth surrogate for the value function using log-sum-exp for norm_inf.
    Returns negative value for minimization.
    """
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(g) / (m + 1)  # g >= 0 since h >= 0
    
    # Smooth max via log-sum-exp
    g_max = np.max(g)
    if g_max <= 0:
        return 0.0
    lse = g_max + (1.0 / beta) * np.log(np.sum(np.exp(beta * (g - g_max))))
    
    if norm_1 <= 0 or lse <= 0:
        return 0.0
        
    val = l2 / (norm_1 * lse)
    return -val

def _generate_starts():
    x = np.linspace(0, N - 1, N)
    starts = []
    # Uniform
    starts.append(np.ones(N))
    # Gaussian centered
    starts.append(np.exp(-0.5 * ((x - N / 2) / (N / 4)) ** 2))
    # Two peaks
    p1 = np.exp(-0.5 * ((x - N * 0.25) / (N / 8)) ** 2)
    p2 = np.exp(-0.5 * ((x - N * 0.75) / (N / 8)) ** 2)
    starts.append(p1 + p2)
    # Semicircle
    starts.append(np.sqrt(np.maximum(0.0, 1 - ((2 * x / (N - 1) - 1)) ** 2)))
    # Cosine bump
    starts.append(0.5 * (1 + np.cos(2 * np.pi * x / N)))
    # Triangle
    starts.append(1 - 2 * np.abs(x - (N - 1) / 2) / (N - 1))
    # Sharp peak in center
    starts.append(np.exp(-0.5 * ((x - N / 2) / (N / 10)) ** 2))
    # Flat top with sharp edges
    t = np.abs(x - (N - 1) / 2) / (N / 2)
    starts.append(np.maximum(0.0, 1.0 - t * t))
    # Ramp up then down
    starts.append(np.minimum(x / (N / 2), (N - 1 - x) / (N / 2)))
    # Random smooth profiles
    for _ in range(30):
        freqs = RNG.randint(1, 15, size=12)
        phases = RNG.uniform(0, 2 * np.pi, size=12)
        amps = RNG.uniform(0.1, 1.2, size=12)
        prof = np.zeros(N)
        for f_, p_, a_ in zip(freqs, phases, amps):
            prof += a_ * np.cos(2 * np.pi * f_ * x / N + p_)
        starts.append(np.maximum(0.01, prof))
    # Additional structured starts
    # Off-center Gaussian
    starts.append(np.exp(-0.5 * ((x - N * 0.35) / (N / 5)) ** 2))
    # Wide plateau
    t = np.abs(x - (N - 1) / 2) / (N / 2)
    starts.append(np.where(t < 0.6, 1.0, np.maximum(0.0, 1.5 - 1.5 * t)))
    # Asymmetric bump
    starts.append(np.exp(-0.5 * ((x - N * 0.4) / (N / 6)) ** 2) + 0.5 * np.exp(-0.5 * ((x - N * 0.6) / (N / 8)) ** 2))
    return starts

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_val = 0.0
    best_h = np.ones(N)
    
    def track(h):
        nonlocal best_val, best_h
        h = np.maximum(0.0, h)
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()
    
    starts = _generate_starts()
    
    # Initial evaluation
    for s in starts:
        track(s)
    
    bounds = [(0, None)] * N
    
    # Phase 1: L-BFGS-B with low beta for exploration
    betas_low = [20.0, 50.0, 100.0]
    for s in starts:
        if time.time() > deadline - 65.0:
            break
        x0 = np.maximum(0.0, np.asarray(s, dtype=float))
        
        for beta in betas_low:
            if time.time() > deadline - 35.0:
                break
            try:
                res = opt.minimize(
                    lambda x: _smooth_value(x, beta),
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < 0:
                    track(res.x)
            except Exception:
                pass
    
    # Phase 2: L-BFGS-B with high beta for refinement from best
    betas_high = [100.0, 200.0, 300.0, 500.0]
    for beta in betas_high:
        if time.time() > deadline - 25.0:
            break
        try:
            res = opt.minimize(
                lambda x: _smooth_value(x, beta),
                best_h,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.fun < 0:
                track(res.x)
        except Exception:
            pass
    
    # Phase 3: Extensive polish with perturbations
    for i in range(100):
        if time.time() > deadline - 8.0:
            break
        # Vary perturbation scale: start larger, decrease
        base_scale = max(0.005, 0.10 * np.std(best_h)) * (1.0 - i / 120.0)
        scale = max(0.002, base_scale)
        pert = best_h + RNG.normal(0.0, scale, N)
        pert = np.maximum(0.0, pert)
        
        for beta in [200.0, 300.0, 500.0]:
            if time.time() > deadline - 4.0:
                break
            try:
                res = opt.minimize(
                    lambda x: _smooth_value(x, beta),
                    pert,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-11}
                )
                if res.fun < 0:
                    track(res.x)
            except Exception:
                pass
    
    # Phase 4: Final Nelder-Mead polish
    if time.time() < deadline - 3.0:
        try:
            nm_res = opt.minimize(
                lambda x: _smooth_value(x, 500.0),
                best_h,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-14, 'adaptive': True}
            )
            if nm_res.fun < 0:
                track(nm_res.x)
        except Exception:
            pass
    
    # Phase 5: Very tight final L-BFGS-B polish
    if time.time() < deadline - 1.0:
        try:
            res = opt.minimize(
                lambda x: _smooth_value(x, 500.0),
                best_h,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.fun < 0:
                track(res.x)
        except Exception:
            pass
    
    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    return best_h.tolist()

if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
