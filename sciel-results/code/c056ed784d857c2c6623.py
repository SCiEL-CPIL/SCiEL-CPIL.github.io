import time
import math
import numpy as np
from scipy.optimize import minimize

N = 50
DH = 1.0 / 20.0
OFFSET = 10

_DEADLINE = 150.0
_SAFETY = 10.0
_start = time.time()


def _elapsed():
    return time.time() - _start


def _time_left():
    return _DEADLINE - _SAFETY - _elapsed()


def _autoconv(h):
    """Discrete autoconvolution sampled on the grid defined by OFFSET."""
    h = np.asarray(h, dtype=float)
    full = np.convolve(h, h, mode='full')
    return full[OFFSET:OFFSET + N]


def _ratio_exact(h):
    """Compute the exact ratio R = ||h||^2_2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=float)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 0.0 or l2sq <= 0.0:
        return 0.0
    ac = _autoconv(h)
    acmax = float(np.max(ac))
    if acmax <= 0.0:
        return 0.0
    return l2sq / (l1 * acmax)


def _log_ratio_smooth(h, beta=100.0):
    """Compute log R using log-sum-exp for the max term."""
    h = np.asarray(h, dtype=float)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 0.0 or l2sq <= 0.0:
        return -1e12
    ac = _autoconv(h)
    ac_max = float(np.max(ac))
    if ac_max <= 0.0:
        return -1e12
    ac_shifted = ac - ac_max
    exp_ac = np.exp(beta * ac_shifted)
    sum_exp = float(np.sum(exp_ac))
    if sum_exp <= 0.0:
        return -1e12
    logsumexp = ac_max + (1.0 / beta) * math.log(sum_exp)
    if logsumexp <= 0.0:
        return -1e12
    return math.log(l2sq) - math.log(l1) - math.log(logsumexp)


def _grad_log_ratio_smooth(h, beta=100.0):
    """Compute gradient of log R with respect to h."""
    h = np.asarray(h, dtype=float)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 0.0 or l2sq <= 0.0:
        return np.zeros(N)
    
    ac = _autoconv(h)
    ac_max = float(np.max(ac))
    if ac_max <= 0.0:
        return np.zeros(N)
    
    ac_shifted = ac - ac_max
    exp_ac = np.exp(beta * ac_shifted)
    sum_exp = float(np.sum(exp_ac))
    if sum_exp <= 0.0:
        return np.zeros(N)
    
    logsumexp = ac_max + (1.0 / beta) * math.log(sum_exp)
    if logsumexp <= 0.0:
        return np.zeros(N)
    
    w = exp_ac / sum_exp
    
    d_logsumexp_dh = np.zeros(N)
    for j in range(N):
        s = 0.0
        for k in range(N):
            idx = OFFSET + k - j
            if 0 <= idx < N:
                s += w[k] * h[idx]
        d_logsumexp_dh[j] = 2.0 * s / logsumexp
        
    g = np.zeros(N)
    g += 2.0 * h / l2sq
    g -= 1.0 / l1
    g -= d_logsumexp_dh
    
    return g


def _neg_log_ratio_smooth(h, beta=100.0):
    val = _log_ratio_smooth(h, beta)
    return -val


def _neg_log_ratio_smooth_jac(h, beta=100.0):
    val = _log_ratio_smooth(h, beta)
    grad = _grad_log_ratio_smooth(h, beta)
    return -val, -grad


def _neg_log_ratio_exact(h):
    val = _ratio_exact(h)
    if val <= 0.0:
        return 1e12
    return -math.log(val)


def _initializations():
    rng = np.random.default_rng(42)
    inits = []
    
    # 1. Random positive
    for _ in range(20):
        h = rng.random(N) + 0.1
        inits.append(h)
        
    # 2. Sparse spikes
    for k in [1, 2, 3, 5]:
        for _ in range(5):
            h = np.zeros(N)
            idx = rng.choice(N, size=k, replace=False)
            h[idx] = 1.0
            inits.append(h)
            
    # 3. Bumps
    for _ in range(10):
        center = rng.uniform(0, N)
        width = rng.uniform(1.0, 5.0)
        x = np.arange(N)
        h = np.exp(-0.5 * ((x - center) / width) ** 2)
        inits.append(h)
        
    # 4. Cosine window
    for _ in range(5):
        x = np.arange(N)
        h = 0.5 * (1 + np.cos(2 * np.pi * x / N))
        inits.append(h)
        
    # 5. Alternating
    for _ in range(3):
        h = np.ones(N)
        h[1::2] = 0.5
        inits.append(h)
        
    # 6. Blocky
    for _ in range(5):
        h = np.zeros(N)
        start = rng.integers(0, N-5)
        end = start + rng.integers(5, 15)
        h[start:end] = 1.0
        inits.append(h)
        
    # 7. New: Disjoint Gaussian Bumps (Structural change based on directive)
    # h_k = |a * w1(k) + b * w2(k)|^2 where w1, w2 are non-overlapping
    # We parametrize by centers c1, c2, widths s1, s2, amplitudes a, b
    for _ in range(20):
        c1 = rng.uniform(0, N/4)
        c2 = rng.uniform(3*N/4, N)
        s1 = rng.uniform(0.5, 2.0)
        s2 = rng.uniform(0.5, 2.0)
        a = rng.uniform(0.5, 2.0)
        b = rng.uniform(0.5, 2.0)
        
        x = np.arange(N)
        w1 = np.exp(-0.5 * ((x - c1) / s1) ** 2)
        w2 = np.exp(-0.5 * ((x - c2) / s2) ** 2)
        
        # Combine linearly then square? Or just sum of squares?
        # The directive says h_k = |a w1 + b w2|^2.
        # Since w1, w2 are disjoint (mostly), this is approx a^2 w1^2 + b^2 w2^2 + 2ab w1w2.
        # If they are truly disjoint, cross term is 0.
        # Let's try h = (a*w1 + b*w2)**2 to respect the modulus structure, 
        # though for real positive disjoint functions it's just (sum)^2.
        # Actually, if they are disjoint, (a w1 + b w2)^2 = a^2 w1^2 + b^2 w2^2.
        # This creates a "double well" potential.
        
        h = (a * w1 + b * w2) ** 2
        # Ensure non-negative
        h = np.maximum(h, 0.0)
        
        # Normalize to unit L1 to avoid scale issues during init
        if np.sum(h) > 0:
            h = h / np.sum(h) * 1.0 # Keep scale roughly 1 for optimizer
        inits.append(h)
        
    # 8. New: Tapered Plateau / Raised Cosine variants
    for _ in range(5):
        x = np.arange(N)
        # Raised cosine window
        h = 0.5 * (1 + np.cos(2 * np.pi * x / N))
        # Apply a taper to make it more compact
        taper = np.exp(-0.1 * (x - N/2)**2)
        h = h * taper
        h = np.maximum(h, 0.0)
        if np.sum(h) > 0:
            h = h / np.sum(h) * 1.0
        inits.append(h)

    return inits


def construct():
    global _start
    _start = time.time()
    
    best_h = None
    best_v = 0.0
    
    inits = _initializations()
    
    # Score initializations
    scored = []
    for h in inits:
        v = _ratio_exact(h)
        scored.append((v, h))
        if v > best_v:
            best_v = v
            best_h = h.copy()
            
    scored.sort(key=lambda x: -x[0])
    
    # Optimization loop
    beta = 100.0
    bounds = [(0.0, None)] * N
    
    # We will iterate through the top initializations
    # Allocate time based on number of remaining starts
    n_starts = len(scored)
    
    for i, (v0, h0) in enumerate(scored):
        if _time_left() <= 5.0:
            break
            
        # Skip if this start is much worse than current best and we have many starts left
        if i > 5 and v0 < best_v * 0.8 and best_v > 15.0:
            continue
            
        t0 = time.time()
        time_budget = min(_time_left() * 0.8, 15.0) # Cap per start to ensure diversity
        
        # Phase 1: L-BFGS-B on smooth objective
        try:
            res = minimize(
                _neg_log_ratio_smooth,
                h0,
                method='L-BFGS-B',
                jac=lambda x: _neg_log_ratio_smooth_jac(x, beta),
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False},
                callback=lambda x: (np.nan if (time.time() - t0) > time_budget * 0.7 else x)
            )
            h_opt = np.maximum(res.x, 0.0)
            v_opt = _ratio_exact(h_opt)
            if v_opt > best_v:
                best_v = v_opt
                best_h = h_opt.copy()
        except Exception:
            h_opt = h0.copy()
            
        # Phase 2: Nelder-Mead polish on exact objective
        if _time_left() > 5.0:
            try:
                t1 = time.time()
                time_budget2 = min(_time_left() * 0.5, 8.0)
                res2 = minimize(
                    _neg_log_ratio_exact,
                    h_opt,
                    method='Nelder-Mead',
                    options={'maxiter': 150, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True, 'disp': False},
                    callback=lambda x: (np.nan if (time.time() - t1) > time_budget2 else x)
                )
                h_pol = np.maximum(res2.x, 0.0)
                v_pol = _ratio_exact(h_pol)
                if v_pol > best_v:
                    best_v = v_pol
                    best_h = h_pol.copy()
            except Exception:
                pass
                
    # Final polish on the best solution found
    if best_h is not None and _time_left() > 10.0:
        try:
            t_final = time.time()
            time_budget_final = _time_left() * 0.8
            res_final = minimize(
                _neg_log_ratio_exact,
                best_h,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True, 'disp': False},
                callback=lambda x: (np.nan if (time.time() - t_final) > time_budget_final else x)
            )
            h_final = np.maximum(res_final.x, 0.0)
            v_final = _ratio_exact(h_final)
            if v_final > best_v:
                best_v = v_final
                best_h = h_final.copy()
        except Exception:
            pass

    if best_h is None:
        best_h = np.ones(N)
        
    best_h = np.maximum(best_h, 0.0)
    return [float(x) for x in best_h]


if __name__ == '__main__':
    t0 = time.time()
    h = construct()
    print("N =", len(h))
    print("ratio =", _ratio_exact(h))
    print("time =", time.time() - t0)
