"""Spectral-design approach: optimize magnitude spectrum for flat autoconvolution."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, fmin_bfgs


_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2


def _autoconv_max_abs(x):
    """Max absolute value of the discrete autoconvolution of x."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))


def _autoconv_values(x):
    """Full autoconvolution array."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return conv[:2 * N - 1]


def _integral(x):
    """Integral of the equal-width step function = sum of the heights."""
    return float(np.sum(x))


def _metric(x):
    """max |autoconv| / (integral)^2. Lower is better."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)


def _soft_max_abs(conv, beta=1000.0):
    """Smooth approximation of max |conv| using log-sum-exp."""
    abs_conv = np.abs(conv)
    max_val = np.max(abs_conv)
    shifted = abs_conv - max_val
    return max_val + np.log(np.sum(np.exp(beta * shifted))) / beta


def _smooth_metric(x, beta=500.0):
    """Smooth surrogate: soft_max |autoconv| / (integral)^2."""
    conv = _autoconv_values(x)
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _soft_max_abs(conv, beta) / (s * s)


def _spectral_design_optimize(x_init, deadline, rng):
    """
    Optimize the signal in the frequency domain by controlling the magnitude spectrum.
    
    Key insight: If we can make |X(k)| approximately constant for all k, then the
    autoconvolution (which is |X(k)|^2 in frequency domain) will be flat, minimizing
    its peak. We optimize the log-magnitude of the FFT with phase fixed from init.
    """
    best_x = x_init.copy()
    best_val = _metric(best_x)
    
    # Get initial phase and magnitude
    fx = np.fft.rfft(best_x, _FFT_SIZE)
    phase = np.angle(fx)
    mag_init = np.abs(fx)
    
    # Work with log-magnitude to ensure positivity
    log_mag_init = np.log(mag_init + 1e-10)
    
    def _objective(log_mag):
        """Compute metric from log-magnitude parameters."""
        mag = np.exp(log_mag)
        # Reconstruct complex spectrum
        spec = mag * np.exp(1j * phase)
        # Enforce Hermitian symmetry for real signal
        x = np.fft.irfft(spec, _FFT_SIZE)
        x = x[:N]
        return _smooth_metric(x, beta=300.0)
    
    def _objective_gradient(log_mag):
        """Gradient of smooth metric w.r.t. log-magnitude."""
        mag = np.exp(log_mag)
        spec = mag * np.exp(1j * phase)
        x = np.fft.irfft(spec, _FFT_SIZE)[:N]
        
        conv = _autoconv_values(x)
        abs_conv = np.abs(conv)
        s = _integral(x)
        
        if abs(s) < 1e-12:
            return np.zeros_like(log_mag)
        
        # Gradient of soft_max_abs
        beta = 300.0
        shifted = abs_conv - np.max(abs_conv)
        weights = np.exp(beta * shifted)
        weights /= np.sum(weights)
        
        # Compute gradient w.r.t. x
        grad_x = np.zeros(N)
        for k in range(2 * N - 1):
            if weights[k] < 1e-12:
                continue
            sign_k = np.sign(conv[k]) if conv[k] != 0 else 1.0
            for j in range(N):
                contrib = 0.0
                if j + k < N:
                    contrib += x[j + k]
                if j - k >= 0:
                    contrib += x[j - k]
                grad_x[j] += 2.0 * sign_k * weights[k] * contrib
        
        # Chain rule: d/dlog_mag = d/dx * dx/dspec * dspec/dlog_mag
        # dspec/dlog_mag[j] = spec[j] (since spec = exp(log_mag) * exp(i*phase))
        # dx/dspec involves inverse FFT
        soft_max_val = _soft_max_abs(conv, beta)
        grad_x *= -2.0 * soft_max_val / (s ** 3)
        
        # Transform gradient back to log_mag space
        # dx/dspec = IFFT matrix, so grad_spec = FFT(grad_x padded)
        grad_x_padded = np.zeros(_FFT_SIZE)
        grad_x_padded[:N] = grad_x
        grad_spec = np.fft.rfft(grad_x_padded)
        
        # dlog_mag_j = Re(grad_spec_j / spec_j) * mag_j = Re(grad_spec_j * exp(-i*phase_j))
        grad_log_mag = np.real(grad_spec * np.exp(-1j * phase))
        
        return grad_log_mag
    
    # Phase 1: L-BFGS-B on log-magnitude
    x0 = log_mag_init.copy()
    
    try:
        res = minimize(
            _objective,
            x0,
            jac=_objective_gradient,
            method='L-BFGS-B',
            options={'maxiter': 200, 'gtol': 1e-5, 'ftol': 1e-7}
        )
        if res.fun < best_val:
            # Convert back to time domain
            mag = np.exp(res.x)
            spec = mag * np.exp(1j * phase)
            x_new = np.fft.irfft(spec, _FFT_SIZE)[:N]
            v_new = _metric(x_new)
            if v_new < best_val:
                best_val = v_new
                best_x = x_new.copy()
    except Exception:
        pass
    
    # Phase 2: Fine-tune with smaller beta for sharper max approximation
    beta_fine = 800.0
    
    def _objective_fine(log_mag):
        mag = np.exp(log_mag)
        spec = mag * np.exp(1j * phase)
        x = np.fft.irfft(spec, _FFT_SIZE)[:N]
        return _smooth_metric(x, beta=beta_fine)
    
    x_fine_init = np.exp(np.log(np.abs(np.fft.rfft(best_x, _FFT_SIZE)) + 1e-10))
    phase_fine = np.angle(np.fft.rfft(best_x, _FFT_SIZE))
    
    def _objective_fine_v2(log_mag):
        mag = np.exp(log_mag)
        spec = mag * np.exp(1j * phase_fine)
        x = np.fft.irfft(spec, _FFT_SIZE)[:N]
        return _smooth_metric(x, beta=beta_fine)
    
    def _grad_fine(log_mag):
        mag = np.exp(log_mag)
        spec = mag * np.exp(1j * phase_fine)
        x = np.fft.irfft(spec, _FFT_SIZE)[:N]
        
        conv = _autoconv_values(x)
        abs_conv = np.abs(conv)
        s = _integral(x)
        
        if abs(s) < 1e-12:
            return np.zeros_like(log_mag)
        
        beta = beta_fine
        shifted = abs_conv - np.max(abs_conv)
        weights = np.exp(beta * shifted)
        weights /= np.sum(weights)
        
        grad_x = np.zeros(N)
        for k in range(2 * N - 1):
            if weights[k] < 1e-12:
                continue
            sign_k = np.sign(conv[k]) if conv[k] != 0 else 1.0
            for j in range(N):
                contrib = 0.0
                if j + k < N:
                    contrib += x[j + k]
                if j - k >= 0:
                    contrib += x[j - k]
                grad_x[j] += 2.0 * sign_k * weights[k] * contrib
        
        soft_max_val = _soft_max_abs(conv, beta)
        grad_x *= -2.0 * soft_max_val / (s ** 3)
        
        grad_x_padded = np.zeros(_FFT_SIZE)
        grad_x_padded[:N] = grad_x
        grad_spec = np.fft.rfft(grad_x_padded)
        grad_log_mag = np.real(grad_spec * np.exp(-1j * phase_fine))
        
        return grad_log_mag
    
    try:
        res2 = minimize(
            _objective_fine_v2,
            x_fine_init,
            jac=_grad_fine,
            method='L-BFGS-B',
            options={'maxiter': 150, 'gtol': 1e-6, 'ftol': 1e-8}
        )
        mag2 = np.exp(res2.x)
        spec2 = mag2 * np.exp(1j * phase_fine)
        x_new2 = np.fft.irfft(spec2, _FFT_SIZE)[:N]
        v_new2 = _metric(x_new2)
        if v_new2 < best_val:
            best_val = v_new2
            best_x = x_new2.copy()
    except Exception:
        pass
    
    return best_x, best_val


def _try_flat_spectrum_random(deadline, rng):
    """Try random flat-spectrum signals with random phases."""
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    # Try several random phase configurations with near-flat magnitude
    for trial in range(30):
        if time.time() > deadline:
            break
        
        # Near-flat magnitude with small random variation
        n_bins = len(np.fft.rfft(np.ones(N), _FFT_SIZE))
        log_mag = np.zeros(n_bins) + rng.normal(0, 0.1, n_bins)
        # Force DC component to keep sum positive
        log_mag[0] = np.log(N) + rng.normal(0, 0.05)
        
        phase = rng.uniform(-np.pi, np.pi, n_bins)
        # Phase of DC must be 0 for real signal with positive sum
        phase[0] = 0.0
        
        spec = np.exp(log_mag) * np.exp(1j * phase)
        x = np.fft.irfft(spec, _FFT_SIZE)[:N]
        
        # Ensure positive sum
        s = np.sum(x)
        if s < 1:
            x = x + (1 - s) / N
        
        v = _metric(x)
        if v < best_val:
            best_val = v
            best_x = x.copy()
    
    return best_x, best_val


def _try_local_search_fast(x, deadline, rng, max_rounds=100000):
    """Fast local search with in-place modifications."""
    best_x = x.copy()
    best_val = _metric(best_x)
    no_improve = 0
    step_scale = 0.2
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        if no_improve > 100:
            step_scale *= 0.9
            no_improve = 0
            if step_scale < 0.0005:
                break
        
        k = int(rng.integers(1, 4))
        indices = rng.choice(N, size=k, replace=False)
        
        best_delta = 0
        best_idx = None
        for trial in range(3):
            x_test = best_x.copy()
            for idx in indices:
                x_test[idx] += rng.normal(0, step_scale)
            v = _metric(x_test)
            if v < best_val + best_delta:
                best_delta = v - best_val
                best_idx = x_test
        
        if best_idx is not None and best_delta < 0:
            best_val += best_delta
            best_x = best_idx
            no_improve = 0
        else:
            no_improve += 1
    
    return best_x, best_val


def _final_polish(x, deadline, rng):
    """Coordinate descent polish."""
    best_x = x.copy()
    best_val = _metric(best_x)
    
    for sweep in range(30):
        if time.time() > deadline:
            break
        
        improved = False
        for idx in range(N):
            if time.time() > deadline:
                break
            
            lo, hi = -0.05, 0.05
            best_delta = 0.0
            best_v = best_val
            
            for _ in range(12):
                if time.time() > deadline:
                    break
                m1 = lo + (hi - lo) / 3
                m2 = hi - (hi - lo) / 3
                
                x1 = best_x.copy()
                x1[idx] += m1
                v1 = _metric(x1)
                
                x2 = best_x.copy()
                x2[idx] += m2
                v2 = _metric(x2)
                
                if v1 < v2:
                    hi = m2
                else:
                    lo = m1
                
                if v1 < best_v:
                    best_v = v1
                    best_delta = m1
                if v2 < best_v:
                    best_v = v2
                    best_delta = m2
            
            if best_delta != 0.0 and best_v < best_val:
                best_x[idx] += best_delta
                best_val = best_v
                improved = True
        
        if not improved:
            break
    
    return best_x, best_val


def construct():
    deadline = time.time() + 145.0
    
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    rng = np.random.default_rng(2024)
    
    # Phase 1: Spectral design optimization (main approach)
    if time.time() < deadline - 60:
        x_spec, val_spec = _spectral_design_optimize(best_x, deadline - 60, rng)
        if val_spec < best_val:
            best_val = val_spec
            best_x = x_spec
    
    # Phase 2: Random flat-spectrum exploration
    if time.time() < deadline - 30:
        x_rand, val_rand = _try_flat_spectrum_random(deadline - 30, rng)
        if val_rand < best_val:
            best_val = val_rand
            best_x = x_rand
    
    # Phase 3: Fine-tune with spectral design from current best
    if time.time() < deadline - 20:
        x_spec2, val_spec2 = _spectral_design_optimize(best_x, deadline - 20, rng)
        if val_spec2 < best_val:
            best_val = val_spec2
            best_x = x_spec2
    
    # Phase 4: Local search
    if time.time() < deadline - 10:
        best_x, best_val = _try_local_search_fast(best_x, deadline - 10, rng, max_rounds=50000)
    
    # Phase 5: Final polish
    if time.time() < deadline - 1:
        best_x, best_val = _final_polish(best_x, deadline - 1, rng)
    
    return best_x
# EVOLVE-BLOCK-END
