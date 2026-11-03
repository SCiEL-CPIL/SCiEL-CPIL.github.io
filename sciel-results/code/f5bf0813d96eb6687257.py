"""Erdos minimum overlap: bisection on t with vectorized smooth-surrogate Nesterov
projected gradient, enhanced with 2D basis-mixing initialization and adaptive beta scheduling."""
import time
import numpy as np

N = 100
_TARGET = N / 2.0
_NFFT = 256
_DEADLINE = time.time() + 142


def _autocorr(h):
    h_pad = np.zeros(_NFFT)
    h_pad[:N] = h
    H = np.fft.rfft(h_pad)
    corr = np.fft.irfft(H * np.conj(H))
    return corr[:N]


def _overlap_values(h):
    r = _autocorr(h)
    return (2.0 / N) * (N / 2.0 - r)


def _score(h):
    cs = _overlap_values(h)
    return float(np.max(cs))


def _project(h):
    h = np.array(h, dtype=np.float64, copy=True)
    target = _TARGET
    if abs(np.sum(h) - target) < 1e-13 and np.min(h) >= -1e-13 and np.max(h) <= 1 + 1e-13:
        return np.clip(h, 0.0, 1.0)
    lo = np.min(h) - 2.0
    hi = np.max(h) + 2.0
    def calc_sum(tau):
        return np.sum(np.clip(h - tau, 0.0, 1.0))
    if calc_sum(lo) < target:
        lo -= 10.0
    if calc_sum(hi) > target:
        hi += 10.0
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
    diff = target - np.sum(h_proj)
    if abs(diff) > 1e-15:
        mask = (h_proj > 1e-10) & (h_proj < 1.0 - 1e-10)
        if np.sum(mask) > 0:
            h_proj[mask] += diff / np.sum(mask)
            h_proj = np.clip(h_proj, 0.0, 1.0)
        else:
            idx = np.argmin(h_proj) if diff > 0 else np.argmax(h_proj)
            h_proj[idx] += diff
            h_proj = np.clip(h_proj, 0.0, 1.0)
    return h_proj


def _smooth_max_and_grad(x, beta):
    m = np.max(x)
    e = np.exp(beta * (x - m))
    s = np.sum(e)
    val = (m + np.log(s)) / beta
    grad = e / s
    return val, grad


def _smooth_obj_and_grad_vec(h, beta):
    cs = _overlap_values(h)
    val, g_c = _smooth_max_and_grad(cs, beta)
    n = N
    factor = -(2.0 / n)
    i_idx = np.arange(n)
    k_idx = np.arange(n)
    I, K = np.meshgrid(i_idx, k_idx, indexing='ij')
    fwd_mask = (I >= K) & (K >= 1)
    fwd_vals = np.where(fwd_mask, g_c[K] * h[np.where(fwd_mask, I - K, 0)], 0.0)
    fwd_sum = np.sum(fwd_vals, axis=1)
    bwd_mask = (I + K < n) & (K >= 1)
    bwd_vals = np.where(bwd_mask, g_c[K] * h[np.where(bwd_mask, I + K, 0)], 0.0)
    bwd_sum = np.sum(bwd_vals, axis=1)
    g_h = factor * (g_c[0] * 2.0 * h + fwd_sum + bwd_sum)
    return val, g_h


def _nesterov_projected(h0, beta, n_steps=5000, lr=None, time_budget=10.0):
    t_start = time.time()
    h = _project(h0.copy())
    y = h.copy()
    best_h = h.copy()
    best_val = _smooth_obj_and_grad_vec(h, beta)[0]
    if lr is None:
        lr = 1.0 / (beta * 12.0)
    t_k = 1.0
    for step in range(n_steps):
        if time.time() - t_start > time_budget:
            break
        val_y, grad_y = _smooth_obj_and_grad_vec(y, beta)
        h_new = _project(y - lr * grad_y)
        val_new = _smooth_obj_and_grad_vec(h_new, beta)[0]
        for _ in range(8):
            if val_new <= best_val + 1e-13:
                break
            lr *= 0.7
            h_new = _project(y - lr * grad_y)
            val_new = _smooth_obj_and_grad_vec(h_new, beta)[0]
        else:
            lr *= 0.5
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k ** 2)) / 2.0
        y = h_new + ((t_k - 1.0) / t_new) * (h_new - h)
        y = _project(y)
        h = h_new
        t_k = t_new
        if val_new < best_val:
            best_val = val_new
            best_h = h.copy()
    return best_h, _score(best_h)


def _basis_mixed_init(rng, n_patterns=8):
    """Generate diverse initializations using convex combinations of sign-alternating basis patterns."""
    i = np.arange(N)
    candidates = []
    
    # Pattern 1: Balanced alternating (period 2)
    p1 = np.where(i % 2 == 0, 0.6, 0.4)
    
    # Pattern 2: High-contrast alternating (period 2, stronger)
    p2 = np.where(i % 2 == 0, 0.8, 0.2)
    
    # Pattern 3: Period 4 balanced
    p3 = np.array([0.6, 0.4, 0.4, 0.6]) * np.ones(N // 4 + 1)
    p3 = p3[:N]
    
    # Pattern 4: Period 4 high-contrast
    p4 = np.array([0.8, 0.2, 0.2, 0.8]) * np.ones(N // 4 + 1)
    p4 = p4[:N]
    
    # Pattern 5: Period 3 balanced
    p5 = np.array([0.6, 0.4, 0.4]) * np.ones(N // 3 + 1)
    p5 = p5[:N]
    
    # Pattern 6: Cosine fundamental
    p6 = 0.5 + 0.3 * np.cos(2.0 * np.pi * i / N)
    
    # Pattern 7: Cosine + 2nd harmonic
    p7 = 0.5 + 0.25 * np.cos(2.0 * np.pi * i / N) + 0.15 * np.cos(4.0 * np.pi * i / N)
    
    # Pattern 8: Multi-frequency
    p8 = 0.5
    for f, a in [(1, 0.15), (2, 0.10), (3, 0.08), (5, 0.06)]:
        p8 += a * np.cos(2.0 * np.pi * f * i / N)
    
    patterns = [p1, p2, p3, p4, p5, p6, p7, p8]
    
    # Mix pairs of patterns with random weights
    for _ in range(n_patterns):
        i1 = rng.integers(0, len(patterns))
        i2 = rng.integers(0, len(patterns))
        w = rng.uniform(0.2, 0.8)
        h = w * patterns[i1] + (1 - w) * patterns[i2]
        h = _project(h)
        candidates.append(h)
    
    # Also add some pure patterns
    for p in patterns[:4]:
        candidates.append(_project(p.copy()))
    
    return candidates


def _coord_descent_polish(h0, max_iters=20000, time_budget=15.0, tol=1e-12):
    t_start = time.time()
    h = _project(h0.copy())
    n = N
    best_score = _score(h)
    for iteration in range(max_iters):
        if time.time() - t_start > time_budget:
            break
        cs = _overlap_values(h)
        current_max = np.max(cs)
        active_mask = cs >= current_max - 1e-8
        active_k = np.where(active_mask)[0]
        if len(active_k) == 0:
            break
        best_improvement = 0
        best_move = None
        for k in active_k[:10]:
            if k == 0:
                continue
            num_pairs = n - k
            if num_pairs > 50:
                candidates = np.random.choice(num_pairs, 50, replace=False)
            else:
                candidates = np.arange(num_pairs)
            for i in candidates:
                j = i + k
                h_i = h[i]
                h_j = h[j]
                if (h_i < 1e-10 and h_j < 1e-10) or (h_i > 1-1e-10 and h_j > 1-1e-10):
                    continue
                d_min = max(-h_i, -(1-h_j))
                d_max = min(1-h_i, h_j)
                d_min = max(d_min, -0.15)
                d_max = min(d_max, 0.15)
                if d_max - d_min < 1e-6:
                    continue
                gr = (np.sqrt(5) - 1) / 2
                a = d_min
                b = d_max
                c = b - gr * (b - a)
                d = a + gr * (b - a)
                def eval_d(dd):
                    h_test = h.copy()
                    h_test[i] += dd
                    h_test[j] -= dd
                    h_test = _project(h_test)
                    return _score(h_test)
                fc = eval_d(c)
                fd = eval_d(d)
                for _ in range(20):
                    if b - a < 1e-8:
                        break
                    if fc < fd:
                        b = d
                        d = c
                        fd = fc
                        c = b - gr * (b - a)
                        fc = eval_d(c)
                    else:
                        a = c
                        c = d
                        fc = fd
                        d = a + gr * (b - a)
                        fd = eval_d(d)
                d_opt = (a + b) / 2
                score_opt = eval_d(d_opt)
                if score_opt < best_score - 1e-10:
                    improvement = best_score - score_opt
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_move = (i, j, d_opt, score_opt)
        if best_move is None:
            for _ in range(5):
                h_pert = h + np.random.normal(0, 0.01, n)
                h_pert = _project(h_pert)
                s_pert = _score(h_pert)
                if s_pert < best_score:
                    h = h_pert
                    best_score = s_pert
                    break
            else:
                break
        i, j, d_opt, score_opt = best_move
        h[i] += d_opt
        h[j] -= d_opt
        h = _project(h)
        best_score = _score(h)
        if current_max - best_score < 1e-14:
            break
    return h, best_score


def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)
    best_h = None
    best_score = float('inf')

    def _update(h):
        nonlocal best_h, best_score
        h = _project(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    # Phase 1: Basis-mixed initialization (novel 2D mixing strategy)
    if time.time() - t0 < _DEADLINE - 80:
        candidates = _basis_mixed_init(rng, n_patterns=12)
        for h_cand in candidates:
            _update(h_cand)

    # Phase 2: Additional frequency-based starts for diversity
    if time.time() - t0 < _DEADLINE - 60:
        i = np.arange(N)
        freq_sets = [
            ([1, 2, 3, 5], [0.15, 0.10, 0.08, 0.06]),
            ([1, 2, 4, 7], [0.14, 0.10, 0.07, 0.05]),
            ([1, 3, 4, 6, 9], [0.12, 0.10, 0.08, 0.06, 0.04]),
            ([2, 3, 5, 8, 13], [0.12, 0.10, 0.08, 0.06, 0.04]),
            ([1, 2, 3, 4, 5, 6], [0.10, 0.08, 0.06, 0.05, 0.04, 0.03]),
        ]
        for freqs, amps in freq_sets:
            if time.time() - t0 > _DEADLINE - 60:
                break
            h = np.full(N, 0.5)
            for f, a in zip(freqs, amps):
                h += a * np.cos(2.0 * np.pi * f * i / N)
            _update(h)

    # Phase 3: Bisection on t with feasibility checking
    # Use the bisection structure to guide beta selection
    t_low = 0.0
    t_high = best_score
    bisect_iters = 0
    
    while time.time() - t0 < _DEADLINE - 40 and t_high - t_low > 1e-6:
        t_mid = (t_low + t_high) / 2.0
        bisect_iters += 1
        
        # Adaptive beta: increase as we narrow the gap
        beta = 100.0 + bisect_iters * 100.0
        beta = min(beta, 5000.0)
        
        remaining = _DEADLINE - time.time() - 30.0
        if remaining < 5.0:
            break
        
        time_per_attempt = min(10.0, remaining / 2.0)
        if time_per_attempt < 3.0:
            break
        
        # Try to find feasible h with smooth_max <= t_mid
        h_ref, s_ref = _nesterov_projected(best_h, beta=beta, n_steps=6000,
                                             time_budget=time_per_attempt)
        _update(h_ref)
        
        # Perturbation restart
        if best_h is not None and time.time() - t0 < _DEADLINE - 25:
            h_pert = best_h + rng.normal(0, 0.01, N)
            h_pert = _project(h_pert)
            h_ref2, s_ref2 = _nesterov_projected(h_pert, beta=beta, n_steps=3000,
                                                   time_budget=min(5.0, _DEADLINE - time.time() - 20))
            _update(h_ref2)
        
        # Bisection update
        if best_score <= t_mid + 1e-5:
            t_high = t_mid
        else:
            t_low = t_mid

    # Phase 4: High-beta polishing
    if best_h is not None and time.time() - t0 < _DEADLINE - 15:
        for beta in [2000.0, 5000.0, 10000.0]:
            remaining = _DEADLINE - time.time() - 10.0
            if remaining < 5.0:
                break
            time_budget = min(8.0, remaining / 1.5)
            if time_budget < 2.0:
                break
            h_polish, s_polish = _nesterov_projected(best_h, beta=beta, n_steps=4000,
                                                       time_budget=time_budget)
            _update(h_polish)

    # Phase 5: Coordinate descent polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 8:
        remaining = _DEADLINE - time.time() - 5.0
        if remaining > 5.0:
            h_cd, s_cd = _coord_descent_polish(best_h, max_iters=30000,
                                                 time_budget=min(15.0, remaining))
            _update(h_cd)

    # Phase 6: Final ultra-high-precision polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 3:
        remaining = _DEADLINE - time.time() - 1.0
        if remaining > 2.0:
            h_final, s_final = _nesterov_projected(best_h, beta=20000.0, n_steps=2000,
                                                     time_budget=min(3.0, remaining))
            _update(h_final)
            if time.time() - t0 < _DEADLINE - 1.5:
                remaining2 = _DEADLINE - time.time() - 0.5
                if remaining2 > 1.0:
                    h_cd2, s_cd2 = _coord_descent_polish(best_h, max_iters=10000,
                                                           time_budget=min(2.0, remaining2))
                    _update(h_cd2)

    if best_h is None:
        best_h = np.full(N, 0.5)

    return _project(best_h)
