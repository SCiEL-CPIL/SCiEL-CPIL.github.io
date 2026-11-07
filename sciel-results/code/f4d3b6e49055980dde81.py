"""
Structural change: Replace Newton's method with L-BFGS-B using a smooth
log-sum-exp surrogate for the non-smooth norm_inf (peak autocorrelation).
This decouples the optimization from the kink at the peak, allowing
gradient-based convergence to sharper optima.

Improvements:
  * Add spike and asymmetric initializations to explore sharper peaks.
  * Time-adaptive beta: start loose for global search, tighten for polish.
  * Final coordinate-descent polish on the EXACT objective to exploit
    the non-smooth max at high precision.
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
M = 2 * N - 1          # len(g) = 99
W = 1.0 / (M + 1)      # 1/100
TIME_LIMIT = 150.0
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

def _smooth_value(h, beta=50.0):
    """
    Smooth surrogate for the value function.
    Uses Log-Sum-Exp for norm_inf to provide differentiability.
    Returns negative value for minimization.
    """
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(g) / (m + 1) # g >= 0
    
    # Smooth max: (1/beta) * log(sum(exp(beta * g)))
    # Shift g to prevent overflow
    g_max = np.max(g)
    lse = g_max + (1.0 / beta) * np.log(np.sum(np.exp(beta * (g - g_max))))
    
    if norm_1 <= 0 or lse <= 0:
        return 0.0
        
    val = l2 / (norm_1 * lse)
    return -val

def _smooth_value_log(h_log, beta=50.0):
    """
    Smooth surrogate in log-space for better conditioning.
    h = exp(h_log), so h >= 0 automatically.
    Returns negative value for minimization.
    """
    h = np.exp(h_log)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(g) / (m + 1)
    
    g_max = np.max(g)
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
    # Gaussian (centered)
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
    # Spike (sharp peak at center)
    spike = np.zeros(N)
    spike[N // 2] = 1.0
    starts.append(spike)
    # Spike (sharp peak slightly off-center)
    spike2 = np.zeros(N)
    spike2[N // 3] = 1.0
    starts.append(spike2)
    # Asymmetric Gaussian (wider on one side)
    asym = np.exp(-0.5 * ((x - N / 2) / (N / 4)) ** 2)
    asym[x < N / 2] *= np.exp(0.5 * (x[x < N / 2] - N / 2) / (N / 8))
    starts.append(asym)
    # Random smooth profiles
    for _ in range(20):
        freqs = RNG.randint(1, 8, size=6)
        phases = RNG.uniform(0, 2 * np.pi, size=6)
        amps = RNG.uniform(0.5, 1.0, size=6)
        prof = np.zeros(N)
        for f_, p_, a_ in zip(freqs, phases, amps):
            prof += a_ * np.cos(2 * np.pi * f_ * x / N + p_)
        starts.append(np.maximum(0.1, prof))
    # Random sparse spikes
    for _ in range(5):
        prof = np.zeros(N)
        k = RNG.randint(2, 6)
        idx = RNG.choice(N, size=k, replace=False)
        prof[idx] = RNG.uniform(0.5, 1.5, size=k)
        starts.append(prof)
        
    return starts

def construct():
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 20.0) # 20s safety margin

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

    # Optimization Phase
    # We use L-BFGS-B with bounds [0, inf)
    bounds = [(0, None)] * N
    
    # Time-adaptive beta schedule: start loose, tighten over time
    # Phase 1: global search with moderate beta
    # Phase 2: local refinement with high beta
    # Phase 3: coordinate descent on exact objective
    
    elapsed_frac = 0.0
    
    # Phase 1: Global search in log-space with moderate beta
    beta_global = 30.0
    for s in starts:
        if time.time() > deadline:
            break
        
        x0 = np.maximum(1e-6, np.asarray(s, dtype=float))
        x0_log = np.log(x0)
        
        try:
            res = opt.minimize(
                lambda x: _smooth_value_log(x, beta_global),
                x0_log,
                method='L-BFGS-B',
                bounds=[(-10, 10)] * N,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            if res.success or res.fun < 0:
                track(np.exp(res.x))
        except Exception:
            pass

    # Phase 2: Local refinement in log-space with higher beta
    beta_local = 100.0
    for _ in range(20):
        if time.time() > deadline:
            break
        pert = best_h + RNG.normal(0.0, 0.05, N)
        pert = np.maximum(1e-6, pert)
        pert_log = np.log(pert)
        
        try:
            res = opt.minimize(
                lambda x: _smooth_value_log(x, beta_local),
                pert_log,
                method='L-BFGS-B',
                bounds=[(-10, 10)] * N,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            track(np.exp(res.x))
        except Exception:
            pass

    # Phase 3: Coordinate descent with golden-section 1D search on EXACT objective
    # This exploits the non-smooth max at high precision
    phi = (1 + np.sqrt(5)) / 2
    inv_phi = 1 / phi

    def _golden_section_1d(i, lo, hi, max_iter=40):
        """Golden-section search for best value of h[i] in [lo, hi]."""
        a = lo
        b = hi
        if a >= b:
            return best_h[i]
        c = b - inv_phi * (b - a)
        d = a + inv_phi * (b - a)
        best_h[i] = c
        fc = compute_value(best_h)
        best_h[i] = d
        fd = compute_value(best_h)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - inv_phi * (b - a)
                best_h[i] = c
                fc = compute_value(best_h)
            else:
                a = c
                c = d
                fc = fd
                d = a + inv_phi * (b - a)
                best_h[i] = d
                fd = compute_value(best_h)
        
        # Return the better of the two
        best_h[i] = c
        fc = compute_value(best_h)
        best_h[i] = d
        fd = compute_value(best_h)
        best_h[i] = c if fc >= fd else d
        return best_h[i]

    for _ in range(50):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            
            base_val = best_h[i]
            # Determine search range based on current value
            if base_val < 1e-4:
                lo, hi = 0.0, max(1.0, 2 * base_val + 0.1)
            else:
                lo, hi = max(0.0, base_val - 0.5), base_val + 0.5
            
            old_val = best_h[i]
            best_h[i] = _golden_section_1d(i, lo, hi, max_iter=30)
            new_val = best_h[i]
            
            if abs(new_val - old_val) > 1e-8:
                improved = True
        
        if not improved:
            break

    # Final polish: one more L-BFGS-B pass in log-space with very high beta
    if time.time() < deadline - 5:
        try:
            best_h_safe = np.maximum(1e-6, best_h)
            best_h_log = np.log(best_h_safe)
            res = opt.minimize(
                lambda x: _smooth_value_log(x, 500.0),
                best_h_log,
                method='L-BFGS-B',
                bounds=[(-10, 10)] * N,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.success or res.fun < 0:
                track(np.exp(res.x))
        except Exception:
            pass

    # Final coordinate descent polish with tighter golden-section
    for _ in range(20):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            
            base_val = best_h[i]
            if base_val < 1e-4:
                lo, hi = 0.0, max(0.1, 2 * base_val + 0.01)
            else:
                lo, hi = max(0.0, base_val - 0.1), base_val + 0.1
            
            old_val = best_h[i]
            best_h[i] = _golden_section_1d(i, lo, hi, max_iter=40)
            new_val = best_h[i]
            
            if abs(new_val - old_val) > 1e-10:
                improved = True
        
        if not improved:
            break

    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    return best_h.tolist()

if __name__ == "__main__":
    t0 = time.time()
    h = construct()
    print("Score:", compute_value(h))
    print("Time: %.1f s" % (time.time() - t0))
