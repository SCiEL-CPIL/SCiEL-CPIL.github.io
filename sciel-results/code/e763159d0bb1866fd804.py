"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Combines FFT-accelerated overlap computation with Nesterov accelerated projected
gradient on a log-sum-exp smooth surrogate. Multi-phase beta schedule for
coarse-to-fine optimization with exact projection onto the feasible set.
Adds coordinate-descent polish targeting active constraints.
"""
import time
import numpy as np

N = 100
_TARGET = N / 2.0
_NFFT = 256  # next power of 2 >= 2*N
_DEADLINE = time.time() + 145  # Safety margin before 150s kill


def _autocorr(h):
    """Linear autocorrelation r[k] = sum_{i=k}^{N-1} h[i]*h[i-k] for k=0..N-1 via FFT."""
    h_pad = np.zeros(_NFFT)
    h_pad[:N] = h
    H = np.fft.rfft(h_pad)
    corr = np.fft.irfft(H * np.conj(H))
    return corr[:N]


def _overlap_values(h):
    """C_k = (2/N)*(N/2 - r_k) for k=0..N-1."""
    n = len(h)
    r = _autocorr(h)
    return (2.0 / n) * (n / 2.0 - r)


def _score(h):
    """True (hard) maximum overlap value."""
    cs = _overlap_values(h)
    return float(np.max(cs))


def _project(h):
    """Exact Euclidean projection onto {h in [0,1]^N : sum(h) = N/2}."""
    h = np.array(h, dtype=np.float64, copy=True)
    target = _TARGET

    # Quick check if already feasible
    if abs(np.sum(h) - target) < 1e-13 and np.min(h) >= -1e-13 and np.max(h) <= 1 + 1e-13:
        return np.clip(h, 0.0, 1.0)

    lo = np.min(h) - 2.0
    hi = np.max(h) + 2.0

    def calc_sum(tau):
        return np.sum(np.clip(h - tau, 0.0, 1.0))

    # Ensure bracket
    if calc_sum(lo) < target:
        lo -= 10.0
    if calc_sum(hi) > target:
        hi += 10.0

    # Bisection for tau
    for _ in range(60):
        mid = (lo + hi) / 2.0
        s = calc_sum(mid)
        if abs(s - target) < 1e-14:
            lo = hi = mid
            break
        if s > target:
            lo = mid
        else:
            hi = mid

    tau = (lo + hi) / 2.0
    h_proj = np.clip(h - tau, 0.0, 1.0)

    # Final exact correction
    diff = target - np.sum(h_proj)
    if abs(diff) > 1e-15:
        mask = (h_proj > 1e-10) & (h_proj < 1.0 - 1e-10)
        if np.sum(mask) > 0:
            h_proj[mask] += diff / np.sum(mask)
            h_proj = np.clip(h_proj, 0.0, 1.0)
        else:
            if diff > 0:
                idx = np.argmin(h_proj)
            else:
                idx = np.argmax(h_proj)
            h_proj[idx] += diff
            h_proj = np.clip(h_proj, 0.0, 1.0)

    return h_proj


def _smooth_max_and_grad(x, beta):
    """Log-sum-exp soft-max of x and its gradient (softmax(beta*x))."""
    m = np.max(x)
    e = np.exp(beta * (x - m))
    s = np.sum(e)
    val = (m + np.log(s)) / beta
    grad = e / s
    return val, grad


def _smooth_obj_and_grad(h, beta):
    """f(h) = S_beta(C(h)); returns (f, df/dh)."""
    cs = _overlap_values(h)
    val, g_c = _smooth_max_and_grad(cs, beta)
    n = N
    g_h = np.zeros(n)
    factor = -(2.0 / n)
    for k in range(n):
        gc = g_c[k]
        if gc < 1e-15:
            continue
        if k == 0:
            g_h += factor * gc * (2.0 * h)
        else:
            g_h[k:] += factor * gc * h[:n - k]
            g_h[:n - k] += factor * gc * h[k:]
    return val, g_h


def _coord_descent_polish(h, n_rounds=50, time_budget=8.0):
    """Coordinate descent targeting the maximum C_k constraints.
    For each coordinate i, try small adjustments that reduce the max overlap."""
    t_start = time.time()
    h = _project(h.copy())
    best_score = _score(h)
    best_h = h.copy()
    
    # Identify which k values are near the maximum
    cs = _overlap_values(h)
    max_c = np.max(cs)
    # Active set: k where C_k is within 10% of max
    active_k = np.where(cs > 0.9 * max_c)[0]
    
    for round_idx in range(n_rounds):
        if time.time() - t_start > time_budget:
            break
        
        # For each coordinate, try small perturbations
        for i in range(N):
            if time.time() - t_start > time_budget:
                break
            
            # Try several step sizes
            for delta in [0.01, -0.01, 0.005, -0.005, 0.002, -0.002]:
                h_trial = h.copy()
                h_trial[i] += delta
                h_trial = _project(h_trial)
                
                s_trial = _score(h_trial)
                if s_trial < best_score - 1e-12:
                    best_score = s_trial
                    best_h = h_trial.copy()
                    h = h_trial.copy()
                    # Update active set occasionally
                    if round_idx % 5 == 0:
                        cs = _overlap_values(h)
                        max_c = np.max(cs)
                        active_k = np.where(cs > 0.9 * max_c)[0]
    
    return best_h, best_score


def _nesterov_projected(h0, beta, n_steps=5000, lr=None, time_budget=10.0):
    """Nesterov accelerated projected gradient on the smooth surrogate."""
    t_start = time.time()
    h = _project(h0.copy())
    y = h.copy()
    best_h = h.copy()
    best_val = _smooth_obj_and_grad(h, beta)[0]

    if lr is None:
        lr = 1.0 / (beta * 12.0)

    t_k = 1.0
    stall_count = 0

    for step in range(n_steps):
        if time.time() - t_start > time_budget:
            break

        val_y, grad_y = _smooth_obj_and_grad(y, beta)
        h_new = _project(y - lr * grad_y)
        val_new = _smooth_obj_and_grad(h_new, beta)[0]

        # Backtracking
        for _ in range(8):
            if val_new <= best_val + 1e-13:
                break
            lr *= 0.7
            h_new = _project(y - lr * grad_y)
            val_new = _smooth_obj_and_grad(h_new, beta)[0]
        else:
            lr *= 0.5

        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k ** 2)) / 2.0
        y = h_new + ((t_k - 1.0) / t_new) * (h_new - h)
        y = _project(y)
        h = h_new
        t_k = t_new

        if val_new < best_val - 1e-14:
            best_val = val_new
            best_h = h.copy()
            stall_count = 0
        else:
            stall_count += 1
            # If stalled, reduce learning rate
            if stall_count % 100 == 0:
                lr *= 0.8

    return best_h, _score(best_h)


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _project(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # 1. Initial Candidates
    i = np.arange(N)
    
    # Cosine wave
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    _update_best(h_cos)
    
    # Constant
    h_const = np.full(N, 0.5)
    _update_best(h_const)
    
    # Sine
    h_sin = 0.5 + 0.4 * np.sin(2.0 * np.pi * i / N)
    _update_best(h_sin)
    
    # Alternating
    h_alt = np.where(i % 2 == 0, 0.8, 0.2)
    _update_best(h_alt)
    
    # Random
    h_rand = np.random.uniform(0.1, 0.9, N)
    _update_best(h_rand)
    
    # More diverse candidates
    h_cos2 = 0.5 + 0.3 * np.cos(4.0 * np.pi * i / N)
    _update_best(h_cos2)
    
    h_saw = 0.5 + 0.4 * (2.0 * (i / N - 0.5))
    _update_best(h_saw)
    
    # Additional structured candidates
    # Triangular wave
    h_tri = 0.5 + 0.3 * np.abs(2.0 * (i / N % 1.0) - 1.0) - 0.3
    _update_best(h_tri)
    
    # Two-frequency mix
    h_mix = 0.5 + 0.2 * np.cos(2.0 * np.pi * i / N) + 0.15 * np.cos(4.0 * np.pi * i / N)
    _update_best(h_mix)
    
    # Phase-shifted sine
    h_sin_phase = 0.5 + 0.4 * np.sin(2.0 * np.pi * i / N + np.pi / 4)
    _update_best(h_sin_phase)
    
    # 2. Optimization Phases
    
    # Phase 1: Broad search with lower beta for smoother landscape
    candidates = [h_cos, h_sin, h_alt, h_rand, h_cos2, h_saw, h_tri, h_mix, h_sin_phase]
    
    for h0 in candidates:
        if time.time() - t0 > _DEADLINE - 35:
            break
        # Lower beta for broader exploration
        h_opt, s = _nesterov_projected(h0, beta=15.0, n_steps=2500, time_budget=7.0)
        _update_best(h_opt)
        
    # Phase 2: Refinement with moderate beta from best candidate
    if best_h is not None and time.time() - t0 < _DEADLINE - 25:
        h_opt, s = _nesterov_projected(best_h, beta=50.0, n_steps=3500, time_budget=9.0)
        _update_best(h_opt)
        
    # Phase 3: Coordinate descent polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 12:
        h_cd, s_cd = _coord_descent_polish(best_h, n_rounds=80, time_budget=10.0)
        _update_best(h_cd)
        
    # Phase 4: Final Nesterov with high beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        h_opt, s = _nesterov_projected(best_h, beta=200.0, n_steps=3000, time_budget=7.0)
        _update_best(h_opt)
        
    # Phase 5: Final coordinate descent polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 2:
        h_cd, s_cd = _coord_descent_polish(best_h, n_rounds=50, time_budget=5.0)
        _update_best(h_cd)

    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _project(best_h)
    return best_h
