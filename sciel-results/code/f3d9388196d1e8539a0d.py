import numpy as np
import time

N = 400
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2


def _autoconv_lin(x):
    """Compute linear autoconvolution using FFT."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    return np.fft.irfft(fx * fx, _FFT_SIZE)[:2 * N - 1]


def _metric(x):
    """max |autoconv| / (sum x)^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    return float(np.max(np.abs(c))) / (s * s)


def _smooth_metric(x, beta=300.0):
    """Soft-max |autoconv| / (sum x)^2 for differentiable optimization."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)[:2 * N - 1]
    abs_conv = np.abs(conv)
    max_val = np.max(abs_conv)
    shifted = abs_conv - max_val
    soft = max_val + np.log(np.sum(np.exp(beta * shifted))) / beta
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    return soft / (s * s)


def _allpass_response(phase, freqs):
    """
    Compute all-pass filter frequency response H(f) = prod_k exp(2j*phi_k)
    where the product is over cascade stages.
    Returns complex array of length len(freqs).
    """
    # For an all-pass filter, |H(f)| = 1 for all f by construction.
    # H(f) = prod_k (e^{-2j*pi*f*k} * e^{2j*phi_k})
    # We use the phase parameterization directly.
    n = len(phase)
    # H(f) = exp(2j * sum_k (phi_k - pi*f*k))
    # = exp(2j * (sum phi_k - pi*f*sum k))
    # But this is just a linear phase, not general all-pass.
    # For a proper all-pass: H(f) = prod_k (a_k + e^{-2j*pi*f}) / (1 + a_k*e^{-2j*pi*f})
    # where |a_k| < 1. Parameterize a_k = r_k * e^{j*theta_k}, r_k in (0,1).
    # This is more complex. Let's use a simpler parameterization:
    # H(f) = exp(j * psi(f)) where psi is a smooth function.
    # Actually, for our purpose, we just need |H(f)| = 1.
    # The simplest: H(f) = exp(j * sum_k phi_k * cos(2*pi*f*k))
    # This gives |H(f)| = 1 for all f.
    # H(f) = exp(j * sum_k phi_k * cos(2*pi*f*k))
    k = np.arange(n)
    # psi(f) = sum_k phi_k * cos(2*pi*f*k)
    # Compute via FFT-like approach
    # cos(2*pi*f*k) for f in freqs, k in 0..n-1
    # This is a cosine transform
    psi = np.zeros(len(freqs))
    for kk in range(n):
        psi += phase[kk] * np.cos(2 * np.pi * freqs * kk)
    return np.exp(1j * psi)


def _allpass_to_signal(phase, freqs):
    """
    Convert all-pass phase parameters to time-domain signal.
    H(f) = exp(j * sum_k phi_k * cos(2*pi*f*k))
    x = IFFT(H(f) * |X_target(f)|)
    For flat magnitude, |X_target(f)| = constant.
    """
    H = _allpass_response(phase, freqs)
    # For flat magnitude spectrum, X(f) = H(f) * C where C is constant
    # We want sum(x) = X(0) = H(0) * C = C (since H(0) = 1)
    # Set C = 1, so sum(x) = 1
    X = H  # |X(f)| = 1 for all f
    x = np.fft.irfft(X, _FFT_SIZE)[:N]
    # Normalize to have sum = 1
    s = np.sum(x)
    if abs(s) > 1e-12:
        x = x / s
    return x


def _optimize_allpass_phase(x_init, deadline, rng, n_phase_params=32, max_iters=200):
    """
    Optimize all-pass phase parameters to minimize the metric.
    """
    freqs = np.fft.fftfreq(_FFT_SIZE)[:_FFT_SIZE // 2 + 1]
    # Use only the non-negative frequencies
    freqs_nonneg = np.fft.rfftfreq(_FFT_SIZE)
    
    # Initialize phase parameters from x_init
    fx = np.fft.rfft(x_init, _FFT_SIZE)
    phase_init = np.angle(fx)[:n_phase_params]
    
    best_x = x_init.copy()
    best_val = _metric(x_init)
    
    phase = phase_init.copy()
    step_size = 0.1
    patience = 0
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        x = _allpass_to_signal(phase, freqs_nonneg)
        val = _metric(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
            patience = 0
        else:
            patience += 1
        
        if patience > 20:
            step_size *= 0.5
            patience = 0
            if step_size < 1e-6:
                break
        
        # Compute gradient via finite differences (efficient for small n_phase_params)
        grad = np.zeros(n_phase_params)
        eps = 1e-4
        base_val = _smooth_metric(x)
        
        for k in range(n_phase_params):
            phase_plus = phase.copy()
            phase_plus[k] += eps
            x_plus = _allpass_to_signal(phase_plus, freqs_nonneg)
            val_plus = _smooth_metric(x_plus)
            grad[k] = (val_plus - base_val) / eps
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        
        phase = phase - step_size * grad / grad_norm
        # Keep phases bounded
        phase = np.clip(phase, -np.pi, np.pi)
    
    return best_x, best_val


def _sign_project(x, target_sum=1.0):
    """
    Project x to a sign sequence (values in {-1, 1}) while preserving
    approximate sum. Uses iterative thresholding.
    """
    x = x.copy()
    n = len(x)
    
    for _ in range(50):
        # Threshold to signs
        x_sign = np.sign(x)
        x_sign[x_sign == 0] = 1.0
        
        # Adjust to match target sum
        s = np.sum(x_sign)
        if abs(s) < 1e-12:
            break
        
        # Find elements to flip to reduce |sum - target|
        # Compute error
        error = s - target_sum
        
        if abs(error) < 0.5:
            break
        
        # Flip elements that are most "misaligned"
        # If error > 0, flip some +1 to -1
        # If error < 0, flip some -1 to +1
        if error > 0:
            # Flip +1 elements with smallest |x| (most uncertain)
            pos_idx = np.where(x_sign == 1.0)[0]
            if len(pos_idx) > 0:
                # Sort by x value (ascending, so most negative first)
                order = np.argsort(x[pos_idx])
                n_flip = int(np.ceil(error / 2.0))
                n_flip = min(n_flip, len(pos_idx))
                for i in range(n_flip):
                    x_sign[pos_idx[order[i]]] = -1.0
        else:
            # Flip -1 elements with largest |x| (most uncertain)
            neg_idx = np.where(x_sign == -1.0)[0]
            if len(neg_idx) > 0:
                # Sort by x value (descending, so most positive first)
                order = np.argsort(x[neg_idx])[::-1]
                n_flip = int(np.ceil(-error / 2.0))
                n_flip = min(n_flip, len(neg_idx))
                for i in range(n_flip):
                    x_sign[neg_idx[order[i]]] = 1.0
        
        x = 0.5 * x + 0.5 * x_sign
    
    # Final threshold
    x_final = np.sign(x)
    x_final[x_final == 0] = 1.0
    return x_final


def _subgradient_refine(x, max_iters=30, tol=1e-8):
    """Subgradient refinement for the non-smooth max|c_k| objective."""
    x = x.copy()
    n = len(x)
    
    for _ in range(max_iters):
        s = np.sum(x)
        if abs(s) < 1e-12:
            return x
        
        c = _autoconv_lin(x)
        abs_c = np.abs(c)
        m = np.max(abs_c)
        
        if m < 1e-15:
            return x
        
        # Find which lags are at the maximum
        threshold = m * 0.999
        active_mask = abs_c >= threshold
        
        # Compute subgradient w.r.t. x
        grad = np.zeros(n)
        
        for k in range(len(c)):
            if not active_mask[k]:
                continue
            
            for i in range(n):
                contrib = 0.0
                if i >= k and i - k < n:
                    contrib += x[i - k]
                if k > 0 and i + k < n:
                    contrib += x[i + k]
                grad[i] += contrib
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            return x
        
        step_size = 0.05 / grad_norm
        x_new = x - step_size * grad
        
        # Project: preserve sum
        s_new = np.sum(x_new)
        if abs(s_new) > 1e-12:
            x_new = x_new * (s / s_new)
        
        c_new = _autoconv_lin(x_new)
        m_new = np.max(np.abs(c_new))
        
        if abs(m_new - m) < tol:
            return x_new
        
        x = x_new
    
    return x


def _try_barker_like_sequences():
    """Try sequences inspired by Barker codes."""
    candidates = []
    
    barker_codes = [
        [1, 1, 1, 1, 1, -1, 1],
        [1, 1, 1, -1, 1, -1, -1],
        [1, 1, -1, 1, 1, 1, -1, 1, -1, 1],
        [1, 1, 1, 1, -1, -1, -1, 1, -1, 1, -1, 1],
        [1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1],
    ]
    
    for code in barker_codes:
        x = np.tile(code, N // len(code) + 1)[:N]
        s = np.sum(x)
        if abs(s) > 1e-6:
            x = x / s
        candidates.append(x)
    
    rng = np.random.default_rng(42)
    for _ in range(10):
        x = rng.choice([-1.0, 1.0], size=N)
        c = _autoconv_lin(x)
        max_offdiag = np.max(np.abs(c[1:]))
        if max_offdiag < 20:
            s = np.sum(x)
            if abs(s) > 1e-6:
                x = x / s
            candidates.append(x)
    
    return candidates


def construct():
    deadline = time.time() + 140.0
    rng = np.random.default_rng(42)
    
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    # Candidate 1: Constant sequence
    cand_const = np.ones(N)
    v_const = _metric(cand_const)
    if v_const < best_val:
        best_val = v_const
        best_x = cand_const
    
    # Candidate 2: Barker-like sequences
    if time.time() < deadline - 80:
        try:
            barker_cands = _try_barker_like_sequences()
            for cand in barker_cands:
                v = _metric(cand)
                if v < best_val:
                    best_val = v
                    best_x = cand
        except Exception:
            pass
    
    # Candidate 3: All-pass phase optimization (new structural approach)
    if time.time() < deadline - 40:
        try:
            # Try different numbers of phase parameters
            for n_params in [16, 32, 64]:
                if time.time() > deadline - 30:
                    break
                x_ap, v_ap = _optimize_allpass_phase(
                    best_x, deadline - 30, rng, 
                    n_phase_params=n_params, max_iters=100
                )
                if v_ap < best_val:
                    best_val = v_ap
                    best_x = x_ap
        except Exception:
            pass
    
    # Candidate 4: Sign projection from best continuous solution
    if time.time() < deadline - 20:
        try:
            x_sign = _sign_project(best_x, target_sum=1.0)
            v_sign = _metric(x_sign)
            if v_sign < best_val:
                best_val = v_sign
                best_x = x_sign
        except Exception:
            pass
    
    # Candidate 5: Subgradient refinement
    if time.time() < deadline - 10:
        try:
            x_sub = _subgradient_refine(best_x, max_iters=30, tol=1e-8)
            v_sub = _metric(x_sub)
            if v_sub < best_val:
                best_val = v_sub
                best_x = x_sub
        except Exception:
            pass
    
    # Final: more subgradient iterations if time permits
    if time.time() < deadline - 5:
        try:
            x_final = _subgradient_refine(best_x, max_iters=20, tol=1e-10)
            v_final = _metric(x_final)
            if v_final < best_val:
                best_val = v_final
                best_x = x_final
        except Exception:
            pass
    
    return best_x
