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


def _spectral_to_h(coeffs, num_modes):
    """
    coeffs: [dc, a1, b1, a2, b2, ...]
    num_modes: number of cosine/sine pairs to include (excluding DC)
    """
    i = np.arange(N)
    h = np.full(N, coeffs[0])
    for k in range(1, num_modes + 1):
        a = coeffs[2 * k - 1]
        b = coeffs[2 * k]
        h += a * np.cos(2.0 * np.pi * k * i / N) + b * np.sin(2.0 * np.pi * k * i / N)
    return h


def _h_to_spectral(h, num_modes):
    H = np.fft.rfft(h)
    coeffs = [H[0].real / N]
    for k in range(1, num_modes + 1):
        coeffs.append(2.0 * H[k].real / N)
        coeffs.append(-2.0 * H[k].imag / N)
    return np.array(coeffs)


def _fix_sum(h):
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0

    if np.all(h >= 0.0) and np.all(h <= 1.0) and abs(np.sum(h) - target) < 1e-14:
        return h

    # Binary search for shift alpha such that sum(clip(h + alpha)) = target
    # h + alpha is monotonic in alpha for the sum, provided we are not saturated everywhere.
    # Range of alpha: [-max(h)-1, 1-min(h)+1]
    alpha_low = -max(h) - 1.0
    alpha_high = 1.0 - min(h) + 1.0

    for _ in range(50):
        alpha_mid = (alpha_low + alpha_high) / 2.0
        h_new = np.clip(h + alpha_mid, 0.0, 1.0)
        s = np.sum(h_new)
        if s < target:
            alpha_low = alpha_mid
        else:
            alpha_high = alpha_mid

    h_final = np.clip(h + (alpha_low + alpha_high) / 2.0, 0.0, 1.0)

    # Fine tuning for exact sum
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


def _spectral_objective(coeffs, num_modes, beta):
    h = _spectral_to_h(coeffs, num_modes)
    h = np.clip(h, 0.0, 1.0)
    h = _fix_sum(h)
    val, _ = _smooth_max_val_grad(h, beta)
    return val


def _optimize_spectral(h0, num_modes, beta, maxiter=200):
    coeffs0 = _h_to_spectral(h0, num_modes)
    
    # L-BFGS-B is robust for low-dim problems
    try:
        res = minimize(
            _spectral_objective,
            coeffs0,
            args=(num_modes, beta),
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        h_opt = _spectral_to_h(res.x, num_modes)
        h_opt = np.clip(h_opt, 0.0, 1.0)
        h_opt = _fix_sum(h_opt)
        return h_opt, _score(h_opt)
    except Exception:
        return h0, _score(h0)


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

    # Random starts
    for _ in range(10):
        h_rand = np.random.rand(N)
        starts.append(_fix_sum(h_rand))

    # Sinusoidal starts (low frequency)
    for f in [1, 2, 3, 4, 5]:
        for phase in [0, np.pi/2, np.pi, 3*np.pi/2]:
            h_sine = 0.5 + 0.3 * np.sin(2 * np.pi * f * i / N + phase)
            starts.append(_fix_sum(h_sine))
            h_cos = 0.5 + 0.3 * np.cos(2 * np.pi * f * i / N + phase)
            starts.append(_fix_sum(h_cos))

    # Two-frequency starts
    for f1 in [1, 2]:
        for f2 in [3, 4, 5]:
            for amp1 in [0.2, 0.3]:
                for amp2 in [0.2, 0.3]:
                    h_2f = 0.5 + amp1 * np.cos(2 * np.pi * f1 * i / N) + amp2 * np.cos(2 * np.pi * f2 * i / N)
                    starts.append(_fix_sum(h_2f))

    # Sawtooth-like
    h_saw = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N) * np.sign(np.sin(4.0 * np.pi * i / N))
    starts.append(_fix_sum(h_saw))

    # Evaluate all starts
    for h0 in starts:
        _update_best(h0)

    # Phase 2: Spectral Manifold Optimization
    # This is the core structural change: optimize in low-dim Fourier space
    # from multiple diverse starts to find the best manifold basin.
    if best_h is not None:
        # Select a subset of diverse starts for spectral optimization
        # We pick the top 5 distinct starts based on score, plus some random ones
        sorted_starts = sorted(starts, key=lambda x: _score(x))
        diverse_starts = sorted_starts[:5]
        # Add a few random ones from the rest
        if len(sorted_starts) > 5:
            diverse_starts += sorted_starts[10:15]
        
        for num_modes in [2, 3, 4, 5, 6]:
            if time.time() > _DEADLINE - 20:
                break
            
            beta = 50.0 # Moderate beta for smoothness during manifold optimization
            
            for h0 in diverse_starts:
                if time.time() > _DEADLINE - 10:
                    break
                
                h_candidate, s_candidate = _optimize_spectral(h0, num_modes, beta, maxiter=150)
                if s_candidate < best_score:
                    _update_best(h_candidate)
                
                # Also try from the current best if it's different
                if not np.allclose(h0, best_h):
                    h_candidate2, s_candidate2 = _optimize_spectral(best_h, num_modes, beta, maxiter=100)
                    if s_candidate2 < best_score:
                        _update_best(h_candidate2)

    # Phase 3: Full-space Nesterov Polish
    # Use the best spectral solution to polish in full space
    if best_h is not None:
        for beta in [100.0, 200.0, 500.0]:
            if time.time() > _DEADLINE - 5:
                break
            h_candidate, s_candidate = _accelerated_smooth_max(best_h, beta, max_iter=300, lr_init=0.05)
            if s_candidate < best_score:
                _update_best(h_candidate)

    # Phase 4: Final high-beta polish
    if best_h is not None and time.time() < _DEADLINE - 2:
        h_candidate, s_candidate = _accelerated_smooth_max(best_h, beta=1000.0, max_iter=100, lr_init=0.01)
        if s_candidate < best_score:
            _update_best(h_candidate)

    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _fix_sum(best_h)
    return best_h
