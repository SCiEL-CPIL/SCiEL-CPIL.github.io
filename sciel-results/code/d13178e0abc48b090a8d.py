import time
import numpy as np
from scipy.optimize import minimize

N = 400
_t0 = time.time()
_DEADLINE = 140.0  # safety margin under 150s

def _value(h):
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))

def _fft_convolve(h):
    n = len(h)
    size = 2 * n
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * n - 1]
    return conv

def _smooth_max_value(h, beta=50.0):
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    h_norm = h * (N / s)
    conv = _fft_convolve(h_norm)
    m = np.max(conv)
    scaled = beta * (conv - m)
    scaled = np.clip(scaled, -100, 100)
    log_sum_exp = m + np.log(np.sum(np.exp(scaled))) / beta
    return 2.0 * log_sum_exp / N

def _golay_h():
    """Construct a Golay-type sequence with low autocorrelation."""
    # Start with a m-sequence-like structure
    rng = np.random.default_rng(42)
    # Use a chirp + random perturbation
    i = np.arange(N, dtype=float)
    h = np.sin(2 * np.pi * (i * i / (2 * N)))  # quadratic chirp
    # Add small random noise to break symmetry
    h += 0.1 * rng.normal(0, 1, N)
    return h

def _zadoff_chu_h():
    """Zadoff-Chu sequence."""
    i = np.arange(N, dtype=float)
    u = 1
    h = np.exp(1j * np.pi * u * i * i / N).real
    return h

def construct():
    global _t0
    _t0 = time.time()
    
    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE

    best_h = np.ones(N)
    best_v = _value(best_h)  # 2.0

    def update_best(h):
        nonlocal best_h, best_v
        h = np.asarray(h, dtype=float)
        if not np.all(np.isfinite(h)):
            return best_v
        s = float(np.sum(h))
        if s == 0.0:
            return best_v
        v = _value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()
        return v

    # Try initial structures
    for init_h in [_golay_h(), _zadoff_chu_h(), np.ones(N)]:
        update_best(init_h)

    # Phase 1: DCT spectral shaping
    # Represent h = DCT(c) where c are DCT coefficients
    # Optimize c to minimize smooth max value
    K = 20  # number of DCT modes
    
    def dct_to_h(c):
        """Convert DCT coefficients to h, ensuring sum(h)=N."""
        i = np.arange(N, dtype=float)
        h = np.zeros(N)
        h += c[0] * np.sqrt(1.0/N)
        for k in range(1, K):
            h += c[k] * np.sqrt(2.0/N) * np.cos(np.pi * k * (i + 0.5) / N)
        # Add remaining modes as small random to allow flexibility
        # Actually, let's use full DCT
        return h

    def full_dct_to_h(c):
        """Full DCT inverse."""
        # Use scipy idct
        from scipy.fft import idct
        h = idct(c, type=2, norm='ortho')
        return h

    def dct_smooth_obj(c, beta=50.0):
        h = full_dct_to_h(c)
        return _smooth_max_value(h, beta)

    if deadline_ok():
        try:
            from scipy.fft import dct
            # Initialize c from best_h
            c0 = dct(best_h, type=2, norm='ortho')
            
            # L-BFGS-B on DCT coefficients
            res = minimize(
                lambda c: dct_smooth_obj(c, 50.0),
                c0,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.x is not None:
                h_candidate = full_dct_to_h(res.x)
                update_best(h_candidate)
        except Exception:
            pass

    # Phase 2: Full-dimensional L-BFGS-B with smooth surrogate
    if deadline_ok():
        try:
            s_best = float(np.sum(best_h))
            if s_best != 0:
                h_start = best_h * (N / s_best)
            else:
                h_start = np.ones(N)
            
            res = minimize(
                lambda h: _smooth_max_value(h, 50.0),
                h_start,
                method='L-BFGS-B',
                options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.x is not None:
                update_best(res.x)
        except Exception:
            pass

    # Phase 3: Coordinate descent polish
    if deadline_ok():
        try:
            s_best = float(np.sum(best_h))
            if s_best != 0:
                h_cd = best_h * (N / s_best)
            else:
                h_cd = np.ones(N)
            
            beta_cd = 100.0
            step_size = 0.5
            max_cd_rounds = 5
            
            for rnd in range(max_cd_rounds):
                if not deadline_ok():
                    break
                improved = False
                for i in range(N):
                    if not deadline_ok():
                        break
                    # Try increasing
                    h_trial = h_cd.copy()
                    h_trial[i] += step_size
                    s_t = float(np.sum(h_trial))
                    if s_t != 0:
                        h_trial = h_trial * (N / s_t)
                        v_trial = _smooth_max_value(h_trial, beta_cd)
                        v_curr = _smooth_max_value(h_cd, beta_cd)
                        if v_trial < v_curr:
                            h_cd = h_trial
                            improved = True
                            continue
                    # Try decreasing
                    h_trial = h_cd.copy()
                    h_trial[i] -= step_size
                    s_t = float(np.sum(h_trial))
                    if s_t != 0:
                        h_trial = h_trial * (N / s_t)
                        v_trial = _smooth_max_value(h_trial, beta_cd)
                        if v_trial < v_curr:
                            h_cd = h_trial
                            improved = True
                if not improved:
                    break
                step_size *= 0.6
            
            update_best(h_cd)
        except Exception:
            pass

    # Phase 4: Gradient descent with subgradient of max
    if deadline_ok():
        try:
            s_best = float(np.sum(best_h))
            if s_best != 0:
                h_gd = best_h * (N / s_best)
            else:
                h_gd = np.ones(N)
            
            lr = 0.01
            for step in range(200):
                if not deadline_ok():
                    break
                conv = _fft_convolve(h_gd)
                idx_max = np.argmax(conv)
                # Gradient of max(conv) w.r.t. h is h shifted by idx_max
                # conv[k] = sum_i h[i] * h[i+k-N] for k in [0, 2N-2]
                # d(conv[idx_max])/d(h[j]) = h[j + idx_max - N] if in range, or h[j - (N-1-idx_max)]
                # Actually: conv[k] = sum_{i=max(0,k-N+1)}^{min(N-1,k)} h[i]*h[i+k]
                # This is complex. Use finite difference or skip.
                # Instead, do a simple projected gradient step
                break  # Skip for now, coordinate descent should suffice
        except Exception:
            pass

    # Final validation
    s = float(np.sum(best_h))
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
