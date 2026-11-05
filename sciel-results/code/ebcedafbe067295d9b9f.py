import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

N = 400
DEADLINE = 140.0  # safety margin before 150s kill


def _value(h):
    """Compute the exact task value."""
    if not np.all(np.isfinite(h)):
        return 1e18
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return 1e18
    conv = np.convolve(h, h)
    return abs(2.0 * N * float(np.max(conv)) / (s * s))


def _bandlimited_h(cutoff_frac, n_harmonics=200, phase=0.0):
    """
    Generate a band-limited signal in frequency domain.
    The signal has zero energy in the central frequency band [0, cutoff_frac]
    and unity elsewhere, then inverse FFT to time domain.
    
    cutoff_frac: fraction of the frequency axis that is suppressed (0 to 0.5)
    n_harmonics: number of frequency bins to use
    phase: additional phase shift
    """
    # Frequency indices for N-point DFT
    # We want to suppress low frequencies (near DC) to reduce the peak in autocorrelation
    # The autocorrelation peak is related to the integral of |H(f)|^2
    # By suppressing low frequencies, we spread the energy more evenly
    
    # Create frequency response
    freqs = np.fft.fftfreq(N, d=1.0/N)  # frequencies in cycles per sample
    # Normalize to [0, 0.5]
    freq_norm = np.abs(freqs) / 0.5  # now in [0, 1]
    
    # Suppress frequencies below cutoff_frac
    mask = np.ones(N)
    mask[freq_norm < cutoff_frac] = 0.0
    
    # Apply a smooth transition to avoid Gibbs phenomena
    transition_width = 0.05
    low_edge = cutoff_frac - transition_width
    high_edge = cutoff_frac + transition_width
    smooth_mask = mask.copy()
    for i in range(N):
        if low_edge < freq_norm[i] < high_edge:
            # Smoothstep transition
            t = (freq_norm[i] - low_edge) / (high_edge - low_edge)
            smooth_mask[i] = t * t * (3 - 2 * t)
    
    # Add a DC component to ensure sum(h) != 0
    # We need sum(h) to be nonzero, so we add a small DC offset
    dc_offset = 0.1
    
    # Inverse FFT to get time-domain signal
    h_freq = smooth_mask * np.exp(1j * phase)
    h_freq[0] += dc_offset  # Add DC component
    
    h = np.real(np.fft.ifft(h_freq))
    
    # Normalize so that sum(h) = 1 (to simplify the objective)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return None
    h = h / s
    
    return h


def _value_bandlimited(cutoff_frac):
    """Objective function for optimizing the cutoff frequency."""
    h = _bandlimited_h(cutoff_frac)
    if h is None:
        return 1e18
    return _value(h)


def _optimize_cutoff():
    """Optimize the cutoff frequency parameter."""
    t0 = time.time()
    
    # Grid search first
    best_cutoff = 0.1
    best_val = 1e18
    
    for cf in np.linspace(0.01, 0.45, 50):
        if time.time() - t0 > DEADLINE * 0.3:
            break
        h = _bandlimited_h(cf)
        if h is not None:
            v = _value(h)
            if v < best_val:
                best_val = v
                best_cutoff = cf
    
    # Refine with minimize_scalar
    if time.time() - t0 < DEADLINE * 0.5:
        try:
            res = minimize_scalar(
                _value_bandlimited,
                bounds=(max(0.005, best_cutoff - 0.1), min(0.48, best_cutoff + 0.1)),
                method='bounded',
                options={'xatol': 1e-6, 'maxiter': 100}
            )
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_cutoff = res.x
        except Exception:
            pass
    
    h_best = _bandlimited_h(best_cutoff)
    if h_best is None:
        h_best = np.ones(N)
        best_val = _value(h_best)
    
    return h_best, best_val, best_cutoff


def _smooth_objective(h, beta=50.0):
    """Smooth surrogate using log-sum-exp for max in autocorrelation."""
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return 1e18
    conv = np.convolve(h, h)
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -700, 700)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_max = m + log_sum / beta
    val = 2.0 * N * soft_max / (s * s)
    return abs(val)


def _smooth_grad(h, beta=50.0):
    """Gradient of the smooth objective."""
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return np.zeros(N), 1e18
    
    conv = np.convolve(h, h)
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -700, 700)
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    soft_max = m + np.log(np.sum(weights)) / beta
    
    d_soft_max = 2.0 * np.correlate(weights, h, mode='valid')
    
    sign = 1.0
    if (2.0 * N * soft_max / (s * s)) < 0:
        sign = -1.0
        
    grad = sign * 2.0 * N * (d_soft_max / (s * s) - 2.0 * soft_max / (s ** 3))
    
    obj = abs(2.0 * N * soft_max / (s * s))
    return grad, obj


def _local_refine(h0, max_time=30.0):
    """Local refinement using smooth objective."""
    t0 = time.time()
    best_h = h0.copy()
    best_val = _value(h0)
    
    for beta in [30.0, 60.0, 100.0, 150.0]:
        if time.time() - t0 > max_time * 0.85:
            break
        try:
            def obj_fn(h):
                return _smooth_objective(h, beta=beta)
            
            def grad_fn(h):
                g, _ = _smooth_grad(h, beta=beta)
                return g
            
            res = minimize(
                obj_fn,
                best_h,
                jac=grad_fn,
                method='L-BFGS-B',
                options={'maxiter': 400, 'maxfun': 800, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    return best_h, best_val


def _multi_start(h_best, n_starts=6, max_time=25.0):
    """Multi-start local refinement with diverse perturbations."""
    t0 = time.time()
    best_h = h_best.copy()
    best_val = _value(h_best)
    rng = np.random.default_rng(2024)
    
    for trial in range(n_starts):
        if time.time() - t0 > max_time * 0.9:
            break
        
        # Different perturbation strategies
        if trial % 3 == 0:
            # Gaussian perturbation
            scale = 0.05 + 0.02 * trial
            perturbation = rng.normal(0, scale, N)
            perturbation -= np.mean(perturbation)
            h_init = best_h + perturbation
        elif trial % 3 == 1:
            # Block perturbation
            nblocks = 8 + trial * 4
            block_size = N // nblocks
            perturbation = np.zeros(N)
            for b in range(nblocks):
                val = rng.normal(0, 0.1)
                perturbation[b*block_size:(b+1)*block_size] = val
            perturbation -= np.mean(perturbation)
            h_init = best_h + perturbation
        else:
            # Frequency-domain perturbation
            h_freq = np.fft.fft(best_h)
            # Perturb in frequency domain
            pert_freq = rng.normal(0, 0.05, N) + 1j * rng.normal(0, 0.05, N)
            # Preserve DC component
            pert_freq[0] = 0
            h_init = np.real(np.fft.ifft(h_freq + pert_freq))
        
        s = np.sum(h_init)
        if abs(s) < 1e-12:
            continue
        h_init = h_init / s
        
        try:
            def obj_fn(h):
                return _smooth_objective(h, beta=50.0)
            
            def grad_fn(h):
                g, _ = _smooth_grad(h, beta=50.0)
                return g
            
            res = minimize(
                obj_fn,
                h_init,
                jac=grad_fn,
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 400, 'ftol': 1e-14, 'gtol': 1e-11}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    return best_h, best_val


def construct():
    t0 = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - t0)
    
    # Phase 1: Band-limited signal optimization (structural innovation)
    h_band, val_band, _ = _optimize_cutoff()
    
    best_h = h_band.copy()
    best_val = val_band
    
    # Phase 2: Local refinement from band-limited starting point
    if time_left() > 35:
        best_h, best_val = _local_refine(best_h, max_time=min(30.0, time_left() - 15.0))
    
    # Phase 3: Multi-start refinement
    if time_left() > 25:
        best_h, best_val = _multi_start(best_h, n_starts=6, max_time=min(25.0, time_left() - 12.0))
    
    # Phase 4: Final high-precision polish
    if time_left() > 15:
        try:
            def obj_fn(h):
                return _smooth_objective(h, beta=200.0)
            
            def grad_fn(h):
                g, _ = _smooth_grad(h, beta=200.0)
                return g
            
            res = minimize(
                obj_fn,
                best_h,
                jac=grad_fn,
                method='L-BFGS-B',
                options={'maxiter': 600, 'maxfun': 1200, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    # Safety: ensure valid output
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        best_h = np.ones(N)
    
    return best_h


if __name__ == '__main__':
    h = construct()
    print(f"Value: {_value(h)}")
    print(f"Sum: {np.sum(h)}")
    print(f"Min: {np.min(h)}, Max: {np.max(h)}")
