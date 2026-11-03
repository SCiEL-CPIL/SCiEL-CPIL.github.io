import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 145  # Safety margin before 150s kill


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


def _structured_h(freq, phase, amp, ceil_val):
    """h[i] = min(ceil_val, 0.5 + amp * cos(2*pi*freq*i/N + phase)), then fix sum to N/2."""
    i = np.arange(N)
    raw = 0.5 + amp * np.cos(2.0 * np.pi * freq * i / N + phase)
    h = np.minimum(ceil_val, raw)
    h = _fix_sum(h)
    return h


def _structured_score(params):
    """Score for structured parameters: [freq, phase, amp, ceil_val]."""
    freq, phase, amp, ceil_val = params
    h = _structured_h(freq, phase, amp, ceil_val)
    return _score(h)


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


def _structured_search():
    """Search over 4D parameter space: freq, phase, amp, ceil_val."""
    best_score = float('inf')
    best_params = None
    best_h = None

    # Grid search over frequencies
    freqs = np.arange(1, 11)  # 1 to 10
    phases = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    amps = np.linspace(0.1, 0.49, 20)
    ceils = np.linspace(0.5, 1.0, 25)

    t_start = time.time()
    total_grid = len(freqs) * len(phases) * len(amps) * len(ceils)
    count = 0

    for f in freqs:
        for ph in phases:
            for a in amps:
                for c in ceils:
                    count += 1
                    if time.time() > _DEADLINE - 60:
                        break
                    h = _structured_h(f, ph, a, c)
                    s = _score(h)
                    if s < best_score:
                        best_score = s
                        best_params = np.array([f, ph, a, c])
                        best_h = h.copy()
                if time.time() > _DEADLINE - 60:
                    break
            if time.time() > _DEADLINE - 60:
                break
        if time.time() > _DEADLINE - 60:
            break

    # Local refinement using scipy minimize on the 4D parameter space
    if best_params is not None and time.time() < _DEADLINE - 15:
        bounds = [(0.5, 10.0), (0, 2 * np.pi), (0.05, 0.49), (0.5, 1.0)]
        try:
            res = minimize(_structured_score, best_params, method='Nelder-Mead',
                           bounds=bounds, options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
                best_h = _structured_h(*res.x)
        except Exception:
            pass

    # Try multiple random initializations for local refinement
    if time.time() < _DEADLINE - 10:
        np.random.seed(123)
        for trial in range(10):
            if time.time() > _DEADLINE - 10:
                break
            f0 = np.random.uniform(1, 8)
            ph0 = np.random.uniform(0, 2 * np.pi)
            a0 = np.random.uniform(0.1, 0.45)
            c0 = np.random.uniform(0.6, 1.0)
            x0 = np.array([f0, ph0, a0, c0])
            bounds = [(0.5, 10.0), (0, 2 * np.pi), (0.05, 0.49), (0.5, 1.0)]
            try:
                res = minimize(_structured_score, x0, method='Nelder-Mead',
                               bounds=bounds, options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10})
                if res.fun < best_score:
                    best_score = res.fun
                    best_params = res.x
                    best_h = _structured_h(*res.x)
            except Exception:
                pass

    if best_h is None:
        best_h = _structured_h(1.0, 0.0, 0.5, 1.0)

    return best_h, best_score


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

    best_h = None
    best_score = float('inf')

    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    # Phase 1: Structured sinusoidal manifold search (primary strategy)
    if time.time() < _DEADLINE - 50:
        h_struct, s_struct = _structured_search()
        if s_struct < best_score:
            best_score = s_struct
            best_h = h_struct.copy()

    # Phase 2: General optimization from the best structured solution
    if best_h is None:
        best_h = np.full(N, 0.5)
        _update_best(best_h)

    # Nesterov accelerated gradient refinement
    if best_h is not None and time.time() < _DEADLINE - 20:
        # First pass with moderate beta
        h_candidate, s_candidate = _accelerated_smooth_max(best_h, beta=50.0, max_iter=300)
        if s_candidate < best_score:
            _update_best(h_candidate)

        # Second pass with higher beta
        if time.time() < _DEADLINE - 15:
            h_candidate2, s_candidate2 = _accelerated_smooth_max(best_h, beta=100.0, max_iter=300)
            if s_candidate2 < best_score:
                _update_best(h_candidate2)

        # Try from a perturbed start
        if time.time() < _DEADLINE - 10:
            np.random.seed(42)
            h_pert = best_h + 0.05 * np.random.randn(N)
            h_pert = _fix_sum(h_pert)
            h_candidate3, s_candidate3 = _accelerated_smooth_max(h_pert, beta=80.0, max_iter=200)
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

        _run_optimization(best_h, beta=100.0, maxiter=150, ftol=1e-10)

        # Final high-precision polish
        if time.time() < _DEADLINE - 3:
            _run_optimization(best_h, beta=200.0, maxiter=200, ftol=1e-12)

    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _fix_sum(best_h)
    return best_h
