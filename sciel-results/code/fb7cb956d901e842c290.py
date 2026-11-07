"""
Model-and-optimize approach with smooth surrogate and gradient-based search.

Key insight from cell findings: a smooth log-sum-exp surrogate for the
non-differentiable norm_inf lets L-BFGS-B take meaningful gradient steps
across the kink, and diverse centered/multi-bump initializations give more
entry points into high-value basins.

Strategy:
1. Define a smooth surrogate: value_smooth = l2 / (norm_1 * soft_max(g))
   where soft_max uses log-sum-exp with temperature parameter.
2. Multi-start with diverse structured initializations.
3. L-BFGS-B optimization on the smooth surrogate.
4. Final coordinate-descent polish on the exact value function.
5. Deadline guard with safety margin.
"""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50
M = 2 * N - 1  # 99
W = 1.0 / (M + 1)  # 1/100
TIME_LIMIT = 145.0  # safety margin before 150s kill


def compute_value(h):
    """Exact objective value."""
    h = np.asarray(h, dtype=float).ravel()
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if not np.all(np.isfinite(h)):
        return 0.0
    h = np.maximum(0.0, h)
    if np.all(h == 0):
        return 0.0
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)


def _smooth_neg_value(h, temp=1.0):
    """
    Smooth negative surrogate for optimization.
    Uses log-sum-exp as a smooth approximation to max(g) for the norm_inf term.
    Returns negative so we minimize.
    """
    h = np.maximum(0.0, np.asarray(h, dtype=float).ravel())
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if not np.all(np.isfinite(h)):
        return 1e10
    if np.all(h == 0):
        return 1e10
    
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    
    # Log-sum-exp smooth max of g
    g_max = np.max(g)
    if g_max <= 0:
        return 1e10
    # Numerically stable log-sum-exp
    shifted = g / temp - g_max / temp
    log_sum_exp = temp * (g_max / temp + np.log(np.sum(np.exp(shifted))))
    soft_max = log_sum_exp
    
    if l2 <= 0 or norm_1 <= 0 or soft_max <= 0:
        return 1e10
    
    return -(l2 / (norm_1 * soft_max))


def _smooth_neg_value_grad(h, temp=1.0):
    """Numerical gradient of smooth negative surrogate using central differences."""
    eps = 1e-5
    grad = np.zeros(N)
    h = np.maximum(0.0, np.asarray(h, dtype=float).ravel())
    for i in range(N):
        h_plus = h.copy()
        h_minus = h.copy()
        h_plus[i] = h[i] + eps
        h_minus[i] = max(0.0, h[i] - eps)
        grad[i] = (_smooth_neg_value(h_plus, temp) - _smooth_neg_value(h_minus, temp)) / (2 * eps)
    return grad


def _coordinate_descent(h, max_iter=200, step_init=0.1):
    """Local coordinate descent on exact value with adaptive step and line search."""
    h = np.maximum(0.0, np.asarray(h, dtype=float).copy())
    best_val = compute_value(h)
    step = step_init
    for it in range(max_iter):
        improved = False
        for i in range(N):
            orig = h[i]
            # Try multiple step sizes with geometric progression
            for mult in [1.0, 0.5, 2.0, 0.25, 4.0, 0.125]:
                for sign in [1, -1]:
                    delta = sign * step * mult
                    trial = h.copy()
                    trial[i] = max(0.0, orig + delta)
                    v = compute_value(trial)
                    if v > best_val + 1e-10:
                        best_val = v
                        h = trial
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            step *= 0.7
            if step < 1e-9:
                break
    return h, best_val


def _generate_diverse_starts(n_starts=40):
    """Generate diverse structured initializations."""
    starts = []
    x = np.linspace(0, N - 1, N)
    rng = np.random.RandomState(42)
    
    # Uniform
    starts.append(np.ones(N))
    
    # Single Gaussian centered
    starts.append(np.exp(-0.5 * ((x - N / 2) / (N / 6)) ** 2))
    
    # Single Gaussian off-center
    starts.append(np.exp(-0.5 * ((x - N * 0.3) / (N / 5)) ** 2))
    starts.append(np.exp(-0.5 * ((x - N * 0.7) / (N / 5)) ** 2))
    
    # Narrow centered Gaussian
    starts.append(np.exp(-0.5 * ((x - N / 2) / (N / 10)) ** 2))
    
    # Two-bump
    starts.append(np.exp(-0.5 * ((x - N * 0.25) / (N / 7)) ** 2) + np.exp(-0.5 * ((x - N * 0.75) / (N / 7)) ** 2))
    starts.append(np.exp(-0.5 * ((x - N * 0.35) / (N / 8)) ** 2) + np.exp(-0.5 * ((x - N * 0.65) / (N / 8)) ** 2))
    
    # Three-bump
    starts.append(np.exp(-0.5 * ((x - N * 0.2) / (N / 6)) ** 2) + np.exp(-0.5 * ((x - N * 0.5) / (N / 6)) ** 2) + np.exp(-0.5 * ((x - N * 0.8) / (N / 6)) ** 2))
    
    # Triangular
    starts.append(1 - 2 * np.abs(x - (N - 1) / 2) / (N - 1))
    
    # Cosine bump
    starts.append(0.5 * (1 + np.cos(2 * np.pi * x / N)))
    
    # Ramp
    starts.append(np.linspace(1, 0.1, N))
    starts.append(np.linspace(0.1, 1, N))
    
    # Plateau
    starts.append(np.where(np.abs(x - (N-1)/2) < N/4, 1.0, 0.1))
    
    # Random smooth profiles
    for _ in range(n_starts - len(starts)):
        n_basis = rng.randint(1, 5)
        centers = rng.uniform(0, N, n_basis)
        widths = rng.uniform(N / 6, N / 2.5, n_basis)
        amps = rng.uniform(0.2, 2.0, n_basis)
        prof = np.zeros(N)
        for c, w, a in zip(centers, widths, amps):
            prof += a * np.exp(-0.5 * ((x - c) / w) ** 2)
        prof = np.maximum(0.01, prof)
        starts.append(prof)
    
    return starts


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_val = 0.0
    best_h = np.ones(N)
    
    def track(h):
        nonlocal best_val, best_h
        h = np.maximum(0.0, np.asarray(h, dtype=float).ravel())
        if h.size != N:
            h = h[:N]
            if h.size < N:
                h = np.pad(h, (0, N - h.size), 'constant')
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()
        return v
    
    # Phase 1: Evaluate diverse starts
    starts = _generate_diverse_starts(40)
    for s in starts:
        track(s)
    
    # Sort starts by value
    start_vals = [compute_value(s) for s in starts]
    order = np.argsort(start_vals)[::-1]
    
    # Phase 2: L-BFGS-B optimization from top starts with smooth surrogate
    # Use multiple temperature values for balance
    temps = [3.0, 1.5, 0.8]
    
    n_optimize = min(15, len(starts))
    for idx in order[:n_optimize]:
        if time.time() > deadline - 40:
            break
        
        h0 = starts[idx].copy()
        
        for temp in temps:
            if time.time() > deadline - 35:
                break
            
            try:
                res = minimize(
                    _smooth_neg_value,
                    h0,
                    args=(temp,),
                    jac=_smooth_neg_value_grad,
                    method='L-BFGS-B',
                    bounds=[(0, 10)] * N,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                h_opt = np.maximum(0.0, res.x)
                track(h_opt)
            except Exception:
                pass
    
    # Phase 3: Coordinate descent polish from best and diverse points
    if time.time() < deadline - 20:
        h_pol, v_pol = _coordinate_descent(best_h, max_iter=400, step_init=0.08)
        track(h_pol)
    
    # Polish from several other promising starts
    polish_candidates = []
    for idx in order[:12]:
        if time.time() > deadline - 15:
            break
        h0 = starts[idx].copy()
        h_pol, v_pol = _coordinate_descent(h0, max_iter=150, step_init=0.1)
        polish_candidates.append((v_pol, h_pol))
    
    polish_candidates.sort(key=lambda x: -x[0])
    for v, h in polish_candidates[:6]:
        if time.time() > deadline - 10:
            break
        h_pol, v_pol = _coordinate_descent(h, max_iter=300, step_init=0.05)
        track(h_pol)
    
    # Phase 4: Fine polish on the best
    if time.time() < deadline - 5:
        h_fin, v_fin = _coordinate_descent(best_h, max_iter=500, step_init=0.02)
        track(h_fin)
    
    # Final very fine polish
    if time.time() < deadline - 2:
        h_fin, v_fin = _coordinate_descent(best_h, max_iter=300, step_init=0.005)
        track(h_fin)
    
    # Phase 5: One more ultra-fine pass if time remains
    if time.time() < deadline - 1:
        h_fin, v_fin = _coordinate_descent(best_h, max_iter=200, step_init=0.001)
        track(h_fin)
    
    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    return best_h.tolist()


if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
