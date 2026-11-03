import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 140  # Safety margin before 150s kill


def _linear_overlap_values(h):
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _smooth_max_grad(h, beta):
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    val = m + np.log(sum_exp) / beta
    grad = np.zeros(n)
    for k in range(n):
        wk = w[k]
        if wk < 1e-15:
            continue
        factor = -(2.0 / n) * wk
        if k > 0:
            grad[k:] += factor * h[:n - k]
        grad[:n - k] += factor * h[k:]
    return val, grad


def _fix_sum(h):
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 1e-10) & (h < 1.0 - 1e-10)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
    diff = target - np.sum(h)
    if abs(diff) > 1e-14:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
    return h


def _accelerated_smooth_max(h0, beta, max_iter=500):
    """Nesterov accelerated projected gradient on smooth max with box+sum constraints."""
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _fix_sum(h)

    h_prev = h.copy()
    t_k = 1.0
    lr = 0.01

    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        val, grad = _smooth_max_grad(h, beta)

        # Gradient step
        h_new = h - lr * grad

        # Project onto {h: 0 <= h_i <= 1, sum(h) = N/2}
        h_new = np.clip(h_new, 0.0, 1.0)
        h_new = _fix_sum(h_new)

        # Nesterov update
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        h_next = h_new + ((t_k - 1.0) / t_new) * (h_new - h_prev)
        h_next = np.clip(h_next, 0.0, 1.0)
        h_next = _fix_sum(h_next)

        h_prev = h
        h = h_next
        t_k = t_new

        # Adaptive learning rate
        if it % 50 == 49:
            lr = max(0.001, min(0.1, lr * 0.95))

    return h, _score(h)


def construct():
    t0 = time.time()
    np.random.seed(42)

    best_h = None
    best_score = float('inf')

    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    i = np.arange(N)

    # Diverse structured starts
    starts = []

    # Pure random
    for _ in range(20):
        h_rand = np.random.rand(N)
        starts.append(_fix_sum(h_rand))

    # Random sine waves
    for _ in range(20):
        f = np.random.uniform(1, 5)
        phase = np.random.uniform(0, 2 * np.pi)
        h_sine = 0.5 + 0.5 * np.sin(2 * np.pi * f * i / N + phase)
        starts.append(_fix_sum(h_sine))

    # Cosine fundamental
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    starts.append(_fix_sum(h_cos))

    # Sine fundamental
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    starts.append(_fix_sum(h_sin))

    # Two-frequency
    h_2f = 0.5 + 0.3 * np.cos(2.0 * np.pi * i / N) + 0.2 * np.cos(4.0 * np.pi * i / N)
    starts.append(_fix_sum(h_2f))

    # Evaluate all starts
    for h0 in starts:
        _update_best(h0)

    # Phase 2: Nesterov accelerated gradient refinement
    if best_h is not None and time.time() < _DEADLINE - 10:
        # First pass with moderate beta
        h_candidate, s_candidate = _accelerated_smooth_max(best_h, beta=50.0, max_iter=400)
        if s_candidate < best_score:
            _update_best(h_candidate)

        # Second pass with higher beta
        if time.time() < _DEADLINE - 10:
            h_candidate2, s_candidate2 = _accelerated_smooth_max(best_h, beta=100.0, max_iter=400)
            if s_candidate2 < best_score:
                _update_best(h_candidate2)

        # Try from a perturbed start
        if time.time() < _DEADLINE - 10:
            h_pert = best_h + 0.05 * np.random.randn(N)
            h_pert = _fix_sum(h_pert)
            h_candidate3, s_candidate3 = _accelerated_smooth_max(h_pert, beta=80.0, max_iter=300)
            if s_candidate3 < best_score:
                _update_best(h_candidate3)

    # Phase 3: SLSQP polish
    if best_h is not None and time.time() < _DEADLINE - 5:
        def _run_optimization(h0, beta, maxiter, ftol):
            nonlocal best_h, best_score
            cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0,
                     'jac': lambda x: np.ones(N)}]
            bounds = [(0.0, 1.0)] * N

            def obj_func(x):
                val, _ = _smooth_max_grad(x, beta)
                return val

            def jac_func(x):
                _, grad = _smooth_max_grad(x, beta)
                return grad

            try:
                res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                               bounds=bounds, constraints=cons,
                               options={'maxiter': maxiter, 'ftol': ftol})
                if res.success or res.fun < best_score:
                    _update_best(res.x)
            except Exception:
                pass

        _run_optimization(best_h, beta=100.0, maxiter=200, ftol=1e-10)

        # Final high-precision polish
        if time.time() < _DEADLINE - 3:
            _run_optimization(best_h, beta=200.0, maxiter=300, ftol=1e-12)

    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _fix_sum(best_h)
    return best_h
