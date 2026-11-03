import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 135  # Safety margin before 150s kill


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


def _spectral_to_h(coeffs):
    n = len(coeffs)
    i = np.arange(N)
    h = np.full(N, coeffs[0])
    num_modes = (n - 1) // 2
    for k in range(1, num_modes + 1):
        a = coeffs[2 * k - 1]
        b = coeffs[2 * k]
        h += a * np.cos(2.0 * np.pi * k * i / N) + b * np.sin(2.0 * np.pi * k * i / N)
    return h


def _h_to_spectral(h):
    H = np.fft.rfft(h)
    coeffs = [H[0].real / N]
    for k in range(1, len(H)):
        coeffs.append(2.0 * H[k].real / N)
        coeffs.append(-2.0 * H[k].imag / N)
    return np.array(coeffs)


def _fix_sum(h):
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0

    if np.all(h >= 0.0) and np.all(h <= 1.0) and abs(np.sum(h) - target) < 1e-14:
        return h

    alpha_low = -max(h) - 1.0
    alpha_high = 1.0 - min(h) + 1.0

    for _ in range(60):
        alpha_mid = (alpha_low + alpha_high) / 2.0
        h_new = np.clip(h + alpha_mid, 0.0, 1.0)
        s = np.sum(h_new)
        if s < target:
            alpha_low = alpha_mid
        else:
            alpha_high = alpha_mid

    h_final = np.clip(h + (alpha_low + alpha_high) / 2.0, 0.0, 1.0)

    diff = target - np.sum(h_final)
    if abs(diff) > 1e-14:
        mask = (h_final > 1e-10) & (h_final < 1.0 - 1e-10)
        if np.any(mask):
            h_final[mask] += diff / np.sum(mask)
            h_final = np.clip(h_final, 0.0, 1.0)

    diff = target - np.sum(h_final)
    if abs(diff) > 1e-14:
        if diff > 0:
            idx = np.argmin(h_final)
        else:
            idx = np.argmax(h_final)
        h_final[idx] += diff
        h_final = np.clip(h_final, 0.0, 1.0)

    return h_final


def _smooth_max_val_grad(h, beta):
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    val = m + np.log(sum_exp) / beta

    # FFT-based gradient: grad = -(2/n) * (w*h + w*h_flip) circular conv
    n_fft = 2 * n
    w_pad = np.zeros(n_fft)
    w_pad[:n] = w
    W = np.fft.rfft(w_pad, n=n_fft)
    H = np.fft.rfft(h, n=n_fft)

    conv_wh = np.fft.irfft(W * H, n=n_fft)[:n]

    h_flip = np.zeros(n)
    h_flip[0] = h[0]
    h_flip[1:] = h[1:][::-1]
    h_flip_pad = np.zeros(n_fft)
    h_flip_pad[:n] = h_flip
    HF = np.fft.rfft(h_flip_pad, n=n_fft)
    conv_whf = np.fft.irfft(W * HF, n=n_fft)[:n]

    grad = -(2.0 / n) * (conv_wh + conv_whf)
    return val, grad


def _accelerated_smooth_max(h0, beta, max_iter=1000, lr_init=0.05):
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _fix_sum(h)
    h_prev = h.copy()
    t_k = 1.0
    lr = lr_init

    best_val = _score(h)
    best_h = h.copy()

    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break

        val, grad = _smooth_max_val_grad(h, beta)
        h_new = h - lr * grad
        h_new = _fix_sum(h_new)

        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        h_next = h_new + ((t_k - 1.0) / t_new) * (h_new - h_prev)
        h_next = _fix_sum(h_next)

        h_prev = h
        h = h_next
        t_k = t_new

        if it % 100 == 99:
            lr = max(0.001, min(0.1, lr * 0.95))

        if it % 50 == 49:
            current_score = _score(h)
            if current_score < best_val:
                best_val = current_score
                best_h = h.copy()

    return best_h, best_val


def _spectral_optimize(num_modes, h0, beta, maxiter=500, ftol=1e-12):
    coeffs0 = _h_to_spectral(h0)
    n_coeff = 2 * num_modes + 1
    coeffs0 = coeffs0[:n_coeff]

    def obj(c):
        h = _spectral_to_h(c)
        h = np.clip(h, 0.0, 1.0)
        h = _fix_sum(h)
        val, _ = _smooth_max_val_grad(h, beta)
        return val

    try:
        res = minimize(obj, coeffs0, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': ftol, 'adaptive': True})
        h_opt = _spectral_to_h(res.x)
        h_opt = np.clip(h_opt, 0.0, 1.0)
        h_opt = _fix_sum(h_opt)
        return h_opt, _score(h_opt)
    except Exception:
        return h0, _score(h0)


def _epigraph_feasibility(h0, t_target, beta, max_iter=300, lr_init=0.03):
    """Try to find h with smooth_max(h, beta) <= t_target.
    Uses accelerated gradient on the objective f(h) = max(0, smooth_max(h) - t_target).
    Returns (h, score) of best found."""
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _fix_sum(h)
    h_prev = h.copy()
    t_k = 1.0
    lr = lr_init

    best_val = _score(h)
    best_h = h.copy()

    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break

        val, grad = _smooth_max_val_grad(h, beta)
        # Only descend if val > t_target (violation)
        if val > t_target:
            h_new = h - lr * grad
        else:
            h_new = h - lr * 0.1 * grad  # gentle refinement

        h_new = _fix_sum(h_new)

        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        h_next = h_new + ((t_k - 1.0) / t_new) * (h_new - h_prev)
        h_next = _fix_sum(h_next)

        h_prev = h
        h = h_next
        t_k = t_new

        if it % 100 == 99:
            lr = max(0.001, min(0.05, lr * 0.95))

        if it % 50 == 49:
            current_score = _score(h)
            if current_score < best_val:
                best_val = current_score
                best_h = h.copy()

    return best_h, best_val


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

    # Phase 1: Generate diverse structured starts
    starts = []

    for _ in range(8):
        h_rand = np.random.rand(N)
        starts.append(_fix_sum(h_rand))

    for f in [1, 2, 3, 4, 5, 6, 7, 8]:
        for phase in [0, np.pi/2, np.pi, 3*np.pi/2]:
            h_sine = 0.5 + 0.3 * np.sin(2 * np.pi * f * i / N + phase)
            starts.append(_fix_sum(h_sine))

    for f in [1, 2, 3]:
        for amp in [0.2, 0.3, 0.4]:
            h_cos = 0.5 + amp * np.cos(2 * np.pi * f * i / N)
            starts.append(_fix_sum(h_cos))

    h_2f = 0.5 + 0.3 * np.cos(2 * np.pi * i / N) + 0.2 * np.cos(4 * np.pi * i / N)
    starts.append(_fix_sum(h_2f))
    h_2f2 = 0.5 + 0.25 * np.sin(2 * np.pi * i / N) + 0.25 * np.sin(3 * np.pi * i / N)
    starts.append(_fix_sum(h_2f2))

    # Sawtooth-like
    h_saw = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N) * np.sign(np.sin(4.0 * np.pi * i / N))
    starts.append(_fix_sum(h_saw))

    for h0 in starts:
        _update_best(h0)

    # Phase 2: Spectral optimization from best starts
    if best_h is not None:
        for num_modes in [2, 3, 4, 5]:
            if time.time() > _DEADLINE - 30:
                break
            h_candidate, s_candidate = _spectral_optimize(num_modes, best_h, beta=50.0, maxiter=200)
            if s_candidate < best_score:
                _update_best(h_candidate)

    # Phase 3: Bisection on t with epigraph feasibility
    # This is the structural idea from inspiration 1: binary search on the
    # maximum overlap value, using accelerated gradient to check feasibility.
    if best_h is not None:
        t_low = 0.0
        t_high = best_score

        for bisect_iter in range(20):
            if time.time() > _DEADLINE - 20:
                break

            t_mid = (t_low + t_high) / 2.0
            # Increase beta as we tighten for sharper max approximation
            beta = 50.0 + bisect_iter * 15

            # Try to find h with smooth_max(h, beta) <= t_mid
            h_candidate, s_candidate = _epigraph_feasibility(
                best_h, t_mid, beta, max_iter=250, lr_init=0.03
            )
            if s_candidate < best_score:
                _update_best(h_candidate)

            # Also try from a perturbed start
            h_pert = best_h + 0.04 * np.random.randn(N)
            h_pert = _fix_sum(h_pert)
            h_candidate2, s_candidate2 = _epigraph_feasibility(
                h_pert, t_mid, beta, max_iter=150, lr_init=0.03
            )
            if s_candidate2 < best_score:
                _update_best(h_candidate2)

            # Check feasibility: if best_score <= t_mid, go lower
            if best_score <= t_mid + 1e-6:
                t_high = t_mid
            else:
                t_low = t_mid

            if t_high - t_low < 1e-8:
                break

    # Phase 4: Full-space Nesterov accelerated gradient with multiple betas
    if best_h is not None:
        for beta in [50.0, 100.0, 200.0]:
            if time.time() > _DEADLINE - 15:
                break
            h_candidate, s_candidate = _accelerated_smooth_max(best_h, beta, max_iter=600, lr_init=0.05)
            if s_candidate < best_score:
                _update_best(h_candidate)

    # Phase 5: Final high-beta polish
    if best_h is not None and time.time() < _DEADLINE - 5:
        h_candidate, s_candidate = _accelerated_smooth_max(best_h, beta=400.0, max_iter=400, lr_init=0.02)
        if s_candidate < best_score:
            _update_best(h_candidate)

    # Phase 6: Ultra-high-beta final polish with tiny steps
    if best_h is not None and time.time() < _DEADLINE - 3:
        h_candidate, s_candidate = _accelerated_smooth_max(best_h, beta=800.0, max_iter=200, lr_init=0.005)
        if s_candidate < best_score:
            _update_best(h_candidate)

    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _fix_sum(best_h)
    return best_h
