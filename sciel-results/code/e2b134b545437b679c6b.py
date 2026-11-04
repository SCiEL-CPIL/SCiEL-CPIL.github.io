"""Second autocorrelation inequality: spectral phase optimization."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0  # safety margin before 150s kill


def _ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-12:
        return 0.0
    l2_sq = np.sum(h * h)
    conv = np.convolve(h, h, mode='full')
    sup = np.max(conv)
    if sup < 1e-12:
        return 0.0
    return l2_sq / (l1 * sup)


def _smooth_max(v, alpha=50.0):
    """Smooth approximation of max(v)."""
    return np.log(np.sum(np.exp(alpha * v)) / len(v)) / alpha


def _autoconv_peak(h, alpha=50.0):
    """Smooth max of autoconvolution."""
    conv = np.convolve(h, h, mode='full')
    return _smooth_max(conv, alpha)


def _objective_complex(theta, mags, alpha=50.0):
    """Objective: minimize -ratio using h_k = mags_k^2 * cos(theta_k + phase_k)."""
    # Build h from magnitudes and phases
    # h_k = |z_k|^2 where z_k = mags_k * exp(i*theta_k)
    # But we need h real and nonnegative, so h_k = mags_k^2
    # The phases affect the autoconvolution structure through the DFT
    # Actually, let's parameterize differently:
    # h = |FFT^{-1}(spectrum)|^2 where spectrum has fixed magnitudes and variable phases
    
    # Simpler: h_k = mags_k^2, and we optimize mags directly
    # The autoconvolution peak depends on the shape of h
    h = mags ** 2
    l1 = np.sum(h)
    if l1 < 1e-12:
        return 1e10
    l2_sq = np.sum(h * h)
    peak = _autoconv_peak(h, alpha)
    if peak < 1e-12:
        return 1e10
    ratio = l2_sq / (l1 * peak)
    return -ratio


def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    # Initial guess: uniform
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    # Strategy: optimize h directly using L-BFGS-B with smooth max
    def objective(h_flat):
        h = np.maximum(h_flat, 0.0)
        l1 = np.sum(h)
        if l1 < 1e-12:
            return 1e10
        l2_sq = np.sum(h * h)
        peak = _autoconv_peak(h, alpha=50.0)
        if peak < 1e-12:
            return 1e10
        ratio = l2_sq / (l1 * peak)
        return -ratio
    
    # Try multiple initializations
    initializations = [
        np.ones(N),
        np.zeros(N),  # will be clipped
        np.exp(-0.5 * np.linspace(-3, 3, N)**2),  # Gaussian
        np.exp(-np.linspace(-2, 2, N)**2),  # Narrower Gaussian
        np.ones(N) * (1 + 0.1 * np.sin(2*np.pi*np.arange(N)/N)),  # Perturbed uniform
    ]
    
    for init in initializations:
        if time_left() < 5:
            break
        init = np.maximum(init, 0.0)
        if np.sum(init) < 1e-12:
            init = np.ones(N)
        
        try:
            res = minimize(objective, init, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            h_opt = np.maximum(res.x, 0.0)
            s_opt = _ratio(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt
        except Exception:
            pass
    
    # Try spectral phase optimization: h = |IFFT(S)|^2 where S has fixed magnitude profile
    if time_left() > 10:
        # Try different magnitude profiles for the spectrum
        for profile_type in ['flat', 'gaussian', 'lorentzian']:
            if time_left() < 5:
                break
            
            if profile_type == 'flat':
                spec_mags = np.ones(N)
            elif profile_type == 'gaussian':
                freqs = np.fft.fftfreq(N)
                spec_mags = np.exp(-5.0 * freqs**2)
            else:  # lorentzian
                freqs = np.fft.fftfreq(N)
                spec_mags = 1.0 / (1.0 + 5.0 * freqs**2)
            
            # Optimize phases of spectrum
            def phase_objective(phases):
                spectrum = spec_mags * np.exp(1j * phases)
                # Enforce conjugate symmetry for real h
                # For real-valued IFFT, need S[k] = conj(S[N-k])
                # We'll just take real part
                h_complex = np.fft.ifft(spectrum)
                h = np.abs(h_complex)**2
                h = np.maximum(h, 0.0)
                l1 = np.sum(h)
                if l1 < 1e-12:
                    return 1e10
                l2_sq = np.sum(h * h)
                peak = _autoconv_peak(h, alpha=50.0)
                if peak < 1e-12:
                    return 1e10
                ratio = l2_sq / (l1 * peak)
                return -ratio
            
            rng = np.random.default_rng(42)
            for trial in range(3):
                if time_left() < 3:
                    break
                phases0 = rng.uniform(-np.pi, np.pi, size=N)
                try:
                    res = minimize(phase_objective, phases0, method='L-BFGS-B',
                                  options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
                    h_opt = np.abs(np.fft.ifft(spec_mags * np.exp(1j * res.x)))**2
                    h_opt = np.maximum(h_opt, 0.0)
                    s_opt = _ratio(h_opt)
                    if s_opt > best_score:
                        best_score = s_opt
                        best_h = h_opt
                except Exception:
                    pass
    
    # Final: try a few more random restarts if time permits
    if time_left() > 15:
        rng = np.random.default_rng(123)
        for trial in range(20):
            if time_left() < 5:
                break
            h0 = rng.uniform(0, 1, size=N)
            try:
                res = minimize(objective, h0, method='L-BFGS-B',
                              options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10})
                h_opt = np.maximum(res.x, 0.0)
                s_opt = _ratio(h_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_h = h_opt
            except Exception:
                pass
    
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)
