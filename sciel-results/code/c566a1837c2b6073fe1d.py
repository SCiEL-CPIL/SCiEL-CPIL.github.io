"""
Optimized program for the second autocorrelation inequality constant.
Strategy:
1. Fast exact evaluation.
2. Epigraph reformulation: minimize max(g) subject to l2/(norm_1 * max(g)) >= target.
   Equivalently, maximize l2 / (norm_1 * max(g)).
   Use a penalty/epigraph approach with L-BFGS-B.
3. Diverse starts including power-law, sparse spikes, and structured profiles.
4. Nelder-Mead polish on the exact objective for the kink in max.
5. Time-bounded search with safety margin.
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
M = 2 * N - 1          # len(g) = 99
W = 1.0 / (M + 1)      # 1/100
TIME_LIMIT = 145.0
RNG = np.random.RandomState(42)

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
    if np.all(h == 0):
        return 0.0
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
    Smooth surrogate: -l2 / (norm_1 * lse(g))
    """
    h = np.maximum(0.0, h)
    if np.all(h == 0):
        return 0.0
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
    return -l2 / (norm_1 * lse)

def _epigraph_objective(h, t, penalty=100.0):
    """
    Epigraph formulation: minimize t + penalty * max(0, l2/(norm_1*t) - target)^2
    But simpler: just minimize t subject to l2 >= target * norm_1 * t.
    We use a soft constraint: minimize t + penalty * max(0, target * norm_1 - l2/t)^2
    Actually, let's use: minimize t + penalty * max(0, target * norm_1 * t - l2)^2 / (norm_1^2 * t^2 + eps)
    """
    h = np.maximum(0.0, h)
    if np.all(h == 0):
        return t + 1e10
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(g) / (m + 1)
    
    if norm_1 <= 0 or t <= 0:
        return t + 1e10
    
    # We want l2 / (norm_1 * t) >= target, i.e., l2 >= target * norm_1 * t
    deficit = target * norm_1 * t - l2
    if deficit > 0:
        penalty_term = penalty * deficit * deficit / (norm_1 * norm_1 * t * t + 1e-10)
    else:
        penalty_term = 0.0
    return t + penalty_term

def _neg_exact_value(h):
    return -compute_value(h)

def _generate_starts():
    x = np.linspace(0, N - 1, N)
    starts = []
    
    # 1. Uniform (Seed)
    starts.append(np.ones(N))
    
    # 2. Gaussian centered
    starts.append(np.exp(-0.5 * ((x - N / 2) / (N / 4)) ** 2))
    
    # 3. Two Peaks
    p1 = np.exp(-0.5 * ((x - N * 0.25) / (N / 8)) ** 2)
    p2 = np.exp(-0.5 * ((x - N * 0.75) / (N / 8)) ** 2)
    starts.append(p1 + p2)
    
    # 4. Semicircle
    starts.append(np.sqrt(np.maximum(0.0, 1 - ((2 * x / (N - 1) - 1)) ** 2)))
    
    # 5. Cosine bump
    starts.append(0.5 * (1 + np.cos(2 * np.pi * x / N)))
    
    # 6. Triangle
    starts.append(1 - 2 * np.abs(x - (N - 1) / 2) / (N - 1))
    
    # 7. Power-law profiles
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        starts.append((1 + np.cos(np.pi * x / (N - 1))) ** alpha)
    
    # 8. Sparse spikes
    for center in [N // 4, N // 2, 3 * N // 4]:
        spike = np.zeros(N)
        spike[center] = 1.0
        spike[max(0, center - 1)] = 0.5
        spike[min(N - 1, center + 1)] = 0.5
        starts.append(spike)
    
    # 9. Comb patterns
    for k in [2, 3, 5, 8]:
        comb = 1.0 + 0.8 * np.cos(2 * np.pi * k * x / N)
        starts.append(np.maximum(0.1, comb))
        
    # 10. Random smooth profiles
    for _ in range(20):
        freqs = RNG.randint(1, 10, size=6)
        phases = RNG.uniform(0, 2 * np.pi, size=6)
        amps = RNG.uniform(0.5, 1.0, size=6)
        prof = np.zeros(N)
        for f_, p_, a_ in zip(freqs, phases, amps):
            prof += a_ * np.cos(2 * np.pi * f_ * x / N + p_)
        starts.append(np.maximum(0.1, prof))
        
    # 11. Random binary-like sequences (smoothed)
    for _ in range(15):
        binary = (RNG.rand(N) > 0.5).astype(float)
        kernel = np.ones(3) / 3.0
        smooth_binary = np.convolve(binary, kernel, mode='same')
        starts.append(np.maximum(0.1, smooth_binary))
    
    # 12. Flat-top with tapered edges
    for frac in [0.3, 0.5, 0.7]:
        width = int(N * frac)
        flat = np.zeros(N)
        start_idx = (N - width) // 2
        flat[start_idx:start_idx + width] = 1.0
        # Taper edges
        taper = 3
        if start_idx >= taper:
            flat[start_idx - taper:start_idx] = np.linspace(0, 1, taper)
        if start_idx + width + taper <= N:
            flat[start_idx + width:start_idx + width + taper] = np.linspace(1, 0, taper)
        starts.append(flat)
    
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
    
    # Initial evaluation of starts
    for s in starts:
        track(s)

    bounds = [(0, None)] * N
    betas = [10.0, 20.0, 50.0, 100.0]
    
    optimized_candidates = []
    
    # Phase 1: L-BFGS-B with smooth surrogate
    for s in starts:
        if time.time() > deadline - 40:
            break
            
        x0 = np.maximum(0.0, np.asarray(s, dtype=float))
        
        for beta in betas:
            if time.time() > deadline - 40:
                break
            
            try:
                res = opt.minimize(
                    lambda x, b=beta: _smooth_value(x, b),
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.success or res.fun < 0:
                    x_opt = np.maximum(0.0, res.x)
                    v_opt = compute_value(x_opt)
                    if v_opt > 0.1 * best_val:
                        optimized_candidates.append(x_opt)
            except Exception:
                pass

    # Phase 2: Epigraph-based optimization on top candidates
    unique_candidates = []
    for c in optimized_candidates:
        if not any(np.allclose(c, uc, atol=1e-4) for uc in unique_candidates):
            unique_candidates.append(c)
    
    scored_candidates = [(compute_value(c), c) for c in unique_candidates]
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = [c for _, c in scored_candidates[:8]]
    
    # Add the best start if not already in top
    for s in starts[:3]:
        s = np.maximum(0.0, s)
        if not any(np.allclose(s, tc, atol=1e-4) for tc in top_candidates):
            top_candidates.append(s)
    
    # Epigraph optimization: try to improve by minimizing peak
    for c in top_candidates:
        if time.time() > deadline - 15:
            break
        
        c = np.maximum(0.0, c)
        v_c = compute_value(c)
        if v_c <= 0:
            continue
        
        # Try to find a lower peak with same or better l2/norm_1 ratio
        # Use Nelder-Mead on exact value for polish
        try:
            res_nm = opt.minimize(
                _neg_exact_value,
                c,
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-7, 'fatol': 1e-12, 'adaptive': True}
            )
            track(res_nm.x)
        except Exception:
            pass

    # Phase 3: Local refinement with small perturbations + L-BFGS-B
    if time.time() < deadline - 5:
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            pert = best_h + RNG.normal(0.0, 0.03, N)
            pert = np.maximum(0.0, pert)
            try:
                res = opt.minimize(
                    lambda x: _smooth_value(x, 100.0),
                    pert,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                track(res.x)
            except Exception:
                pass
    
    # Final Nelder-Mead polish on best
    if time.time() < deadline - 2:
        try:
            res_final = opt.minimize(
                _neg_exact_value,
                best_h,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-14, 'adaptive': True}
            )
            track(res_final.x)
        except Exception:
            pass

    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    return best_h.tolist()

if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
