"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses Nesterov accelerated projected gradient on smooth-max surrogate with
adaptive temperature, multiple restarts, and a final high-precision polish."""
import time
import numpy as np

N = 100
_T_START = time.time()
_DEADLINE = _T_START + 145


def _linear_overlap_values(h):
    """Compute C_k = (2/N)*(N/2 - r_k) where r_k = sum_{i=k}^{N-1} h_i h_{i-k}."""
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _smooth_max_val_grad(h, beta):
    """Return (smooth_max_value, gradient) of max(C_k) using log-sum-exp."""
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
    """Adjust h to have sum exactly N/2, keeping values in [0, 1]."""
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


def _project(h):
    """Project onto {h: 0 <= h_i <= 1, sum(h) = N/2}."""
    h = np.clip(h, 0.0, 1.0)
    return _fix_sum(h)


def _nesterov_smooth_max(h0, beta, max_iter=400, lr_init=0.02):
    """Nesterov accelerated projected gradient on smooth max with box+sum constraints."""
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _fix_sum(h)

    h_prev = h.copy()
    t_k = 1.0
    lr = lr_init

    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        val, grad = _smooth_max_val_grad(h, beta)

        # Gradient step
        h_new = h - lr * grad

        # Project
        h_new = _project(h_new)

        # Nesterov update
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        h_next = h_new + ((t_k - 1.0) / t_new) * (h_new - h_prev)
        h_next = _project(h_next)

        h_prev = h
        h = h_next
        t_k = t_new

        # Adaptive learning rate: decrease slowly
        if it % 30 == 29:
            lr = max(0.0005, lr * 0.92)

    return h, _score(h)


def _param_h(a, b, c, d):
    """Parameterized family: h_i = 0.5 + a*cos(2*pi*i/N) + b*cos(4*pi*i/N) + c*sin(2*pi*i/N) + d*sin(4*pi*i/N)"""
    i = np.arange(N)
    h = 0.5 + a * np.cos(2.0 * np.pi * i / N) + b * np.cos(4.0 * np.pi * i / N) \
        + c * np.sin(2.0 * np.pi * i / N) + d * np.sin(4.0 * np.pi * i / N)
    return _fix_sum(h)


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
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

    # Diverse initial guesses
    starts = []

    # Pure cosine
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    starts.append(_fix_sum(h_cos))

    # Pure sine
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    starts.append(_fix_sum(h_sin))

    # Two-frequency cosine
    h_2f = 0.5 + 0.35 * np.cos(2.0 * np.pi * i / N) + 0.25 * np.cos(4.0 * np.pi * i / N)
    starts.append(_fix_sum(h_2f))

    # Two-frequency sine
    h_2fs = 0.5 + 0.35 * np.sin(2.0 * np.pi * i / N) + 0.25 * np.sin(4.0 * np.pi * i / N)
    starts.append(_fix_sum(h_2fs))

    # Mixed
    h_mix = 0.5 + 0.3 * np.cos(2.0 * np.pi * i / N) + 0.2 * np.sin(2.0 * np.pi * i / N) \
        + 0.15 * np.cos(4.0 * np.pi * i / N) + 0.1 * np.sin(4.0 * np.pi * i / N)
    starts.append(_fix_sum(h_mix))

    # Sawtooth-like
    h_saw = 0.5 + 0.4 * np.sin(2.0 * np.pi * i / N) * np.sign(np.sin(4.0 * np.pi * i / N))
    starts.append(_fix_sum(h_saw))

    # Random starts
    for _ in range(8):
        h_r = np.random.uniform(0, 1, N)
        starts.append(_fix_sum(h_r))

    # Parameterized family with various coefficients
    for a in [0.3, 0.4, 0.5]:
        for b in [0.1, 0.2, 0.3]:
            for c in [0.0, 0.1, 0.2]:
                for d in [0.0, 0.1]:
                    h_p = _param_h(a, b, c, d)
                    starts.append(h_p)

    # Evaluate all starts
    for h0 in starts:
        _update_best(h0)

    # Phase 1: Nesterov with moderate beta from best and perturbed starts
    beta_list = [30.0, 50.0, 80.0, 120.0]
    for beta in beta_list:
        if time.time() > _DEADLINE - 15:
            break

        # From current best
        if best_h is not None:
            h_cand, s_cand = _nesterov_smooth_max(best_h, beta, max_iter=300, lr_init=0.015)
            if s_cand < best_score:
                _update_best(h_cand)

        # From perturbed best
        if best_h is not None:
            for _ in range(3):
                h_pert = best_h + 0.03 * np.random.randn(N)
                h_pert = _fix_sum(h_pert)
                h_cand2, s_cand2 = _nesterov_smooth_max(h_pert, beta, max_iter=200, lr_init=0.015)
                if s_cand2 < best_score:
                    _update_best(h_cand2)

        # From some random starts
        for _ in range(2):
            h_r = np.random.uniform(0, 1, N)
            h_r = _fix_sum(h_r)
            h_cand3, s_cand3 = _nesterov_smooth_max(h_r, beta, max_iter=200, lr_init=0.02)
            if s_cand3 < best_score:
                _update_best(h_cand3)

    # Phase 2: Bisection on t with feasibility check
    t_low = 0.0
    t_high = best_score

    for bisect_iter in range(20):
        if time.time() > _DEADLINE - 10:
            break

        t_mid = (t_low + t_high) / 2.0
        beta = 100.0 + bisect_iter * 20

        # Try to find h with score <= t_mid
        if best_h is not None:
            h_cand, s_cand = _nesterov_smooth_max(best_h, beta, max_iter=250, lr_init=0.01)
            if s_cand < best_score:
                _update_best(h_cand)

            # Perturbed
            h_pert = best_h + 0.02 * np.random.randn(N)
            h_pert = _fix_sum(h_pert)
            h_cand2, s_cand2 = _nesterov_smooth_max(h_pert, beta, max_iter=150, lr_init=0.01)
            if s_cand2 < best_score:
                _update_best(h_cand2)

        if best_score <= t_mid + 1e-7:
            t_high = t_mid
        else:
            t_low = t_mid

        if t_high - t_low < 1e-9:
            break

    # Phase 3: Final high-precision polish
    if best_h is not None and time.time() < _DEADLINE - 5:
        for beta_final in [200.0, 300.0, 500.0]:
            if time.time() > _DEADLINE - 3:
                break
            h_final, s_final = _nesterov_smooth_max(best_h, beta_final, max_iter=400, lr_init=0.005)
            if s_final < best_score:
                _update_best(h_final)

    # Phase 4: Coordinate descent polish on the raw max
    if best_h is not None and time.time() < _DEADLINE - 2:
        h_cd = best_h.copy()
        for _ in range(50):
            if time.time() > _DEADLINE - 2:
                break
            improved = False
            cs = _linear_overlap_values(h_cd)
            k_max = np.argmax(cs)
            # Try small perturbations around the worst-shift direction
            for delta in [0.001, 0.002, 0.005]:
                for sign in [1, -1]:
                    h_try = h_cd.copy()
                    # Perturb in direction that reduces C_{k_max}
                    # C_k = (2/N)(N/2 - sum h_i h_{i-k}), so to reduce C_k, increase sum h_i h_{i-k}
                    # Gradient of C_k w.r.t. h: factor * (h shifted)
                    factor = -(2.0 / N)
                    grad_k = np.zeros(N)
                    if k_max > 0:
                        grad_k[k_max:] += factor * h_cd[:N - k_max]
                    grad_k[:N - k_max] += factor * h_cd[k_max:]
                    h_try -= sign * delta * grad_k
                    h_try = _project(h_try)
                    s_try = _score(h_try)
                    if s_try < _score(h_cd):
                        h_cd = h_try
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        if _score(h_cd) < best_score:
            _update_best(h_cd)

    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _fix_sum(best_h)
    return best_h
