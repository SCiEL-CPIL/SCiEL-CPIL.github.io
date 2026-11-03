"""Erdos minimum overlap: smooth-surrogate optimization with epigraph bisection.
Uses a custom projected-gradient solver on the smooth log-sum-exp surrogate
with a variable t (epigraph), enabling fast and accurate optimization.
"""
import time
import numpy as np

N = 100
_DEADLINE = time.time() + 145  # Safety margin before 150s kill
_TARGET = N / 2.0
_NFFT = 256  # next power of 2 >= 2*N


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
    """Project h to sum=N/2, values in [0,1]."""
    h = np.array(h, dtype=np.float64, copy=True)
    h = np.clip(h, 0.0, 1.0)
    diff = _TARGET - np.sum(h)
    if abs(diff) > 1e-14:
        if diff > 0:
            mask = h < 1.0
        else:
            mask = h > 0.0
        cnt = np.sum(mask)
        if cnt > 0:
            h[mask] += diff / cnt
            h = np.clip(h, 0.0, 1.0)
    # Final exact correction
    diff = _TARGET - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
    return h


def _smooth_max_and_grad(x, beta):
    """Log-sum-exp soft-max of x and its gradient (softmax(beta*x))."""
    m = np.max(x)
    e = np.exp(beta * (x - m))
    s = np.sum(e)
    val = (m + np.log(s)) / beta
    grad = e / s  # gradient w.r.t. x
    return val, grad


def _epigraph_obj_and_grad(h, t, beta):
    """Compute objective and gradient for epigraph formulation.
    obj = t + S_beta(C(h) - t)
    Returns (obj, grad_h, grad_t)."""
    cs = _overlap_values(h)
    shifted = cs - t
    sm, g_c = _smooth_max_and_grad(shifted, beta)
    obj = t + sm
    g_h = np.zeros(N)
    factor = -(2.0 / N)
    for k in range(N):
        gc = g_c[k]
        if gc == 0.0:
            continue
        if k == 0:
            g_h += factor * gc * (2.0 * h)
        else:
            g_h[k:] += factor * gc * h[:N - k]
            g_h[:N - k] += factor * gc * h[k:]
    g_t = 1.0 - np.sum(g_c)
    return obj, g_h, g_t


def _projected_gradient_step(h, t, g_h, g_t, beta, lr):
    """One projected gradient step for the epigraph problem.
    We minimize t + S_beta(C(h)-t) subject to h in [0,1]^N, sum(h)=N/2.
    We use a simple approach: update h with projected gradient, then optimize t analytically.
    """
    # Update h: h_new = project(h - lr * g_h)
    h_new = h - lr * g_h
    h_new = _project(h_new)
    
    # For t: the objective is t + S_beta(C(h_new) - t).
    # Let cs = C(h_new). We minimize over t: t + (1/beta)*log(sum(exp(beta*(cs_j - t))))
    # = t + (1/beta)*log(sum(exp(beta*cs_j))) - t
    # = (1/beta)*log(sum(exp(beta*cs_j)))
    # This is independent of t! So the minimum over t is achieved at any t,
    # but we should set t to the max of cs for the epigraph interpretation.
    # Actually, the smooth max is minimized when t equals the soft-max value.
    # Let's just set t to the current max overlap.
    cs_new = _overlap_values(h_new)
    t_new = float(np.max(cs_new))
    
    return h_new, t_new


def _refine_epigraph_pgd(h0, t0, beta, n_iter=2000, time_budget=25.0):
    """Refine using projected gradient descent on the epigraph smooth surrogate."""
    h = h0.copy()
    t = t0
    t_start = time.time()
    
    # Adaptive learning rate
    lr = 0.01
    best_h = h.copy()
    best_score = _score(h)
    
    for it in range(n_iter):
        if time.time() - t_start > time_budget:
            break
        
        obj, g_h, g_t = _epigraph_obj_and_grad(h, t, beta)
        
        # Gradient step
        h_new = h - lr * g_h
        h_new = _project(h_new)
        
        # Update t to be the max overlap of h_new
        cs_new = _overlap_values(h_new)
        t_new = float(np.max(cs_new))
        
        # Check if we improved
        s_new = float(np.max(cs_new))
        if s_new < best_score:
            best_score = s_new
            best_h = h_new.copy()
        
        # Adaptive learning rate: decrease if objective increased
        obj_new, _, _ = _epigraph_obj_and_grad(h_new, t_new, beta)
        if obj_new > obj + 1e-10:
            lr *= 0.8
        else:
            lr = min(lr * 1.05, 0.05)
        
        h = h_new
        t = t_new
    
    return best_h, best_score


def _construct_direct():
    """Direct multi-frequency construction to flatten autocorrelation."""
    i = np.arange(N)
    h = np.full(N, 0.5)
    freqs = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23]
    amplitudes = [0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
    phases = [0.0, 0.7, 1.3, 2.1, 0.4, 1.8, 2.5, 0.9, 1.6, 0.3]
    for f, a, p in zip(freqs, amplitudes, phases):
        h += a * np.cos(2.0 * np.pi * f * i / N + p)
    return _project(h)


def _random_start(rng):
    """Random feasible start."""
    h = rng.uniform(0.0, 1.0, N)
    return _project(h)


def _perturb(h, rng, scale=0.05):
    """Add small random perturbation and re-project."""
    h_p = h + rng.normal(0.0, scale, N)
    return _project(h_p)


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
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

    # Phase 1: Direct construction
    h_direct = _construct_direct()
    _update(h_direct)

    # Phase 2: Diverse frequency-based starts
    if time.time() - t0 < _DEADLINE - 60:
        i = np.arange(N)
        freq_sets = [
            ([1, 2, 3, 5], [0.15, 0.10, 0.08, 0.06]),
            ([1, 2, 4, 7], [0.14, 0.10, 0.07, 0.05]),
            ([1, 3, 4, 6, 9], [0.12, 0.10, 0.08, 0.06, 0.04]),
            ([2, 3, 5, 8, 13], [0.12, 0.10, 0.08, 0.06, 0.04]),
            ([1, 2, 3, 4, 5, 6], [0.10, 0.08, 0.06, 0.05, 0.04, 0.03]),
            ([3, 5, 7, 11], [0.12, 0.10, 0.08, 0.06]),
            ([1, 4, 6, 9, 12], [0.11, 0.09, 0.07, 0.05, 0.03]),
            ([2, 5, 8, 13, 21], [0.10, 0.08, 0.06, 0.04, 0.02]),
        ]
        for freqs, amps in freq_sets:
            if time.time() - t0 > _DEADLINE - 60:
                break
            h = np.full(N, 0.5)
            for f, a in zip(freqs, amps):
                h += a * np.cos(2.0 * np.pi * f * i / N)
            _update(h)

    # Phase 3: Random starts
    if time.time() - t0 < _DEADLINE - 50:
        for _ in range(8):
            if time.time() - t0 > _DEADLINE - 50:
                break
            h_rand = _random_start(rng)
            _update(h_rand)

    # Phase 4: Smooth surrogate refinement with annealed beta using PGD
    beta_list = [15.0, 40.0, 100.0, 250.0, 500.0, 800.0]
    
    starts_to_refine = []
    if best_h is not None:
        starts_to_refine.append(best_h.copy())
    
    # Add perturbed versions of best
    if best_h is not None and time.time() - t0 < _DEADLINE - 80:
        for _ in range(3):
            if time.time() - t0 > _DEADLINE - 80:
                break
            starts_to_refine.append(_perturb(best_h, rng, scale=0.03))
    
    # Add a couple random starts
    if best_h is not None and time.time() - t0 < _DEADLINE - 70:
        for _ in range(2):
            if time.time() - t0 > _DEADLINE - 70:
                break
            starts_to_refine.append(_random_start(rng))

    for beta in beta_list:
        remaining = _DEADLINE - time.time()
        if remaining < 15:
            break
        n_starts = len(starts_to_refine)
        time_per_start = min(20.0, (remaining - 8.0) / max(n_starts, 1))
        if time_per_start < 3.0:
            break
        
        for h_start in starts_to_refine:
            if time.time() - t0 > _DEADLINE - 12:
                break
            t0_start = _score(h_start)
            h_ref, s_ref = _refine_epigraph_pgd(h_start, t0_start, beta=beta,
                                                n_iter=3000, time_budget=time_per_start)
            _update(h_ref)
        
        # Update starts: keep best and add perturbation
        if best_h is not None:
            starts_to_refine[0] = best_h.copy()
            if len(starts_to_refine) > 1:
                starts_to_refine[-1] = _perturb(best_h, rng, scale=0.02)

    # Phase 5: Final high-precision polish with very high beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 10:
        remaining = _DEADLINE - time.time() - 5.0
        if remaining > 5.0:
            t0_start = _score(best_h)
            h_polish, s_polish = _refine_epigraph_pgd(best_h, t0_start, beta=1000.0,
                                                      n_iter=5000, time_budget=min(20.0, remaining))
            _update(h_polish)
        
        # One more polish with even higher beta if time allows
        if best_h is not None and time.time() - t0 < _DEADLINE - 8:
            remaining = _DEADLINE - time.time() - 4.0
            if remaining > 4.0:
                t0_start = _score(best_h)
                h_polish, s_polish = _refine_epigraph_pgd(best_h, t0_start, beta=1500.0,
                                                          n_iter=3000, time_budget=min(12.0, remaining))
                _update(h_polish)

    # Phase 6: Local perturbation search around best
    if best_h is not None and time.time() - t0 < _DEADLINE - 15:
        for scale in [0.02, 0.01, 0.005]:
            if time.time() - t0 > _DEADLINE - 15:
                break
            for _ in range(5):
                if time.time() - t0 > _DEADLINE - 15:
                    break
                h_pert = _perturb(best_h, rng, scale=scale)
                _update(h_pert)
            if best_h is not None:
                # Quick refine of perturbed best
                remaining = _DEADLINE - time.time() - 8.0
                if remaining > 5.0:
                    t0_start = _score(best_h)
                    h_ref, s_ref = _refine_epigraph_pgd(best_h, t0_start, beta=800.0,
                                                        n_iter=2000, time_budget=min(10.0, remaining))
                    _update(h_ref)

    if best_h is None:
        best_h = np.full(N, 0.5)

    return _project(best_h)
