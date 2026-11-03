import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 138  # Safety margin before 150s kill


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


def _smooth_max_val_grad(h, beta):
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    val = m + np.log(sum_exp) / beta

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


def _accelerated_smooth_max(h0, beta, max_iter=1500, lr_init=0.05):
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


def _polish_high_beta(h0, beta, max_iter=800, lr_init=0.01):
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _fix_sum(h)
    h_prev = h.copy()
    t_k = 1.0
    lr = lr_init

    best_val = _score(h)
    best_h = h.copy()

    for it in range(max_iter):
        if time.time() > _DEADLINE - 3:
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

        if it % 50 == 49:
            lr = max(0.0005, lr * 0.97)

        if it % 25 == 24:
            current_score = _score(h)
            if current_score < best_val:
                best_val = current_score
                best_h = h.copy()

    return best_h, best_val


def _local_refinement(h0, beta, max_iter=300, lr_init=0.005):
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _fix_sum(h)
    lr = lr_init

    best_val = _score(h)
    best_h = h.copy()

    for it in range(max_iter):
        if time.time() > _DEADLINE - 2:
            break

        val, grad = _smooth_max_val_grad(h, beta)
        h_new = h - lr * grad
        h_new = _fix_sum(h_new)

        h = h_new

        if it % 30 == 29:
            lr = max(0.0002, lr * 0.95)

        if it % 20 == 19:
            current_score = _score(h)
            if current_score < best_val:
                best_val = current_score
                best_h = h.copy()

    return best_h, best_val


def _psd_spectral_optimize(num_modes, h0, beta, maxiter=200):
    """
    Optimize in the frequency domain directly.
    We parameterize h via its Fourier coefficients (real & imaginary parts for each mode).
    The constraint is that h must be in [0,1] and sum to N/2.
    We use the PSD connection: overlap at shift k is related to the inverse FFT of PSD.
    Specifically, r_k = sum h_i h_{i-k} is the autocorrelation, which is ifft(|H|^2).
    So C_k = (2/N)(N/2 - r_k). Minimizing max C_k is like minimizing the peak of (N/2 - ifft(|H|^2)).
    
    This is equivalent to making the autocorrelation as flat as possible (close to N/2 everywhere).
    This is the "minimum autocorrelation" problem, related to Barker sequences / CAZAC.
    
    We optimize the spectral coefficients directly with Nelder-Mead, 
    projecting back to time domain for constraints.
    """
    coeffs0 = np.fft.rfft(h0)
    # We optimize the first num_modes+1 coefficients (DC + num_modes harmonics)
    # Represented as real and imaginary parts
    n_coeff = 2 * num_modes + 1
    # Initial coeffs: DC is real, others are complex
    c0 = np.zeros(n_coeff)
    c0[0] = coeffs0[0].real
    for k in range(1, num_modes + 1):
        c0[2*k-1] = coeffs0[k].real
        c0[2*k] = coeffs0[k].imag

    def obj(c):
        # Reconstruct h from spectral coeffs
        h = np.zeros(N)
        h += c[0] / N
        for k in range(1, num_modes + 1):
            re = c[2*k-1]
            im = c[2*k]
            # h_k = re * cos + im * sin (since ifft of re*e^{i...} + im*i*e^{i...})
            # Actually, let's just build the complex spectrum and ifft
            pass
        
        # Better: build complex spectrum
        H_spec = np.zeros(N, dtype=np.complex128)
        H_spec[0] = c[0]
        for k in range(1, num_modes + 1):
            H_spec[k] = c[2*k-1] + 1j * c[2*k]
            H_spec[N-k] = c[2*k-1] - 1j * c[2*k]  # Conjugate symmetry for real h
        
        h = np.fft.ifft(H_spec).real
        h = np.clip(h, 0.0, 1.0)
        h = _fix_sum(h)
        
        val, _ = _smooth_max_val_grad(h, beta)
        return val

    try:
        res = minimize(obj, c0, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        # Reconstruct
        H_spec = np.zeros(N, dtype=np.complex128)
        H_spec[0] = res.x[0]
        for k in range(1, num_modes + 1):
            H_spec[k] = res.x[2*k-1] + 1j * res.x[2*k]
            H_spec[N-k] = res.x[2*k-1] - 1j * res.x[2*k]
        h_opt = np.fft.ifft(H_spec).real
        h_opt = np.clip(h_opt, 0.0, 1.0)
        h_opt = _fix_sum(h_opt)
        return h_opt, _score(h_opt)
    except Exception:
        return h0, _score(h0)


def _bisection_feasibility(h0, t_target, beta, max_iter=200):
    """
    Try to find h such that max_overlap <= t_target.
    We minimize the smooth max, but stop early if we get below t_target.
    This is used in a bisection scheme on t.
    """
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _fix_sum(h)
    h_prev = h.copy()
    t_k = 1.0
    lr = 0.02

    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break

        val, grad = _smooth_max_val_grad(h, beta)
        
        # If we've already achieved the target, return
        if val <= t_target + 0.001:
            h = _fix_sum(h)
            return h, _score(h), True

        h_new = h - lr * grad
        h_new = _fix_sum(h_new)

        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        h_next = h_new + ((t_k - 1.0) / t_new) * (h_new - h_prev)
        h_next = _fix_sum(h_next)

        h_prev = h
        h = h_next
        t_k = t_new

        if it % 50 == 49:
            lr = max(0.005, lr * 0.95)

    h = _fix_sum(h)
    s = _score(h)
    return h, s, s <= t_target + 0.001


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

    # Sine waves
    for f in range(1, 12):
        for phase in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:
            h_sine = 0.5 + 0.3 * np.sin(2 * np.pi * f * i / N + phase)
            starts.append(_fix_sum(h_sine))

    # Cosine waves
    for f in range(1, 8):
        for amp in [0.2, 0.3, 0.4]:
            h_cos = 0.5 + amp * np.cos(2 * np.pi * f * i / N)
            starts.append(_fix_sum(h_cos))

    # Two-frequency
    for f1, f2 in [(1, 2), (1, 3), (2, 3), (1, 4), (2, 5)]:
        for a1, a2 in [(0.25, 0.2), (0.2, 0.25), (0.3, 0.15)]:
            h_2f = 0.5 + a1 * np.cos(2 * np.pi * f1 * i / N) + a2 * np.cos(2 * np.pi * f2 * i / N)
            starts.append(_fix_sum(h_2f))

    # Chirp-based starts (squared magnitude of complex chirp)
    for gamma in [1.0, 2.0, 3.0, 5.0, 8.0]:
        for f0 in [0, 1, 2]:
            phase = 2.0 * np.pi * (f0 * i / N + gamma * (i / N) ** 2)
            z = np.exp(1j * phase)
            h_chirp = np.abs(z) ** 2
            h_chirp = h_chirp / np.sum(h_chirp) * (N / 2.0)
            starts.append(_fix_sum(h_chirp))

    # Evaluate all starts
    for h0 in starts:
        _update_best(h0)

    # Phase 2: Spectral (PSD) optimization from best starts
    # This is the structural change: optimize in frequency domain
    if best_h is not None:
        for num_modes in [4, 6, 8, 10, 12]:
            if time.time() > _DEADLINE - 40:
                break
            h_candidate, s_candidate = _psd_spectral_optimize(num_modes, best_h, beta=50.0, maxiter=150)
            if s_candidate < best_score:
                _update_best(h_candidate)

    # Phase 3: Bisection on target overlap t
    # Structural idea from inspiration 1: bisection on t with feasibility check
    if best_h is not None and time.time() < _DEADLINE - 30:
        t_low = 0.0
        t_high = best_score
        
        for bisect_iter in range(15):
            if time.time() > _DEADLINE - 15:
                break
            
            t_mid = (t_low + t_high) / 2.0
            beta = 80.0 + bisect_iter * 15
            
            # Try to find feasible h
            h_candidate, s_candidate, feasible = _bisection_feasibility(
                best_h, t_mid, beta, max_iter=150
            )
            
            if s_candidate < best_score:
                _update_best(h_candidate)
            
            if feasible or s_candidate <= t_mid + 0.01:
                t_high = t_mid
            else:
                t_low = t_mid
            
            if t_high - t_low < 1e-6:
                break

    # Phase 4: Full-space Nesterov accelerated gradient
    if best_h is not None:
        for beta in [100.0, 200.0, 350.0]:
            if time.time() > _DEADLINE - 15:
                break
            h_candidate, s_candidate = _accelerated_smooth_max(best_h, beta, max_iter=800, lr_init=0.04)
            if s_candidate < best_score:
                _update_best(h_candidate)

    # Phase 5: Final high-beta polish
    if best_h is not None and time.time() < _DEADLINE - 8:
        h_candidate, s_candidate = _polish_high_beta(best_h, beta=400.0, max_iter=400, lr_init=0.01)
        if s_candidate < best_score:
            _update_best(h_candidate)

    if best_h is not None and time.time() < _DEADLINE - 4:
        h_candidate, s_candidate = _polish_high_beta(best_h, beta=700.0, max_iter=300, lr_init=0.008)
        if s_candidate < best_score:
            _update_best(h_candidate)

    # Phase 6: Ultra-fine local refinement
    if best_h is not None and time.time() < _DEADLINE - 2:
        h_candidate, s_candidate = _local_refinement(best_h, beta=1000.0, max_iter=150, lr_init=0.003)
        if s_candidate < best_score:
            _update_best(h_candidate)

    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _fix_sum(best_h)
    return best_h
