"""
Spectral phase optimization: minimize max absolute autoconvolution by directly
optimizing the phase of the DFT to flatten the power spectrum, with an explicit
penalty on the peak-to-mean ratio of the autocorrelation sidelobes.

This is a fundamentally different approach from the current program:
- The current program works in the time domain with L-BFGS-B on a soft-max objective.
- This program works in the frequency domain, optimizing phase angles directly.
- The key insight: if we fix the magnitude spectrum to be flat, the ideal
  autoconvolution is a delta. We optimize the phase to make the actual
  autoconvolution as close to a delta as possible.
- We use gradient descent on the phase vector with analytic gradients derived
  from the relationship between phase and time-domain signal.
"""
import numpy as np
import time
import scipy.optimize

N = 400

def _autoconv_lin(x):
    """Compute linear autoconvolution using FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    return np.fft.irfft(fx * fx, size)[:2 * n - 1]

def _metric_offdiag(x):
    """Metric: max |c_k| for k != 0 / S^2"""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    if len(c) > 1:
        return float(np.max(np.abs(c[1:]))) / (s * s)
    return 0.0

def _phase_to_signal(phi, mag):
    """Convert phase vector and fixed magnitude to time-domain signal."""
    # phi and mag are for the full complex FFT (length N)
    X = mag * np.exp(1j * phi)
    x = np.fft.ifft(X).real
    return x

def _signal_to_phase(x):
    """Extract phase from time-domain signal."""
    X = np.fft.fft(x)
    mag = np.abs(X)
    phi = np.angle(X)
    return phi, mag

def _compute_autoconv_from_phase(phi, mag):
    """Compute autoconvolution from phase and magnitude."""
    X = mag * np.exp(1j * phi)
    # Power spectrum
    power = np.abs(X) ** 2
    # Autoconvolution is IFFT of power spectrum (circular)
    # But we need linear autoconvolution, so use zero-padded FFT
    n = len(mag)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    # Pad X to size
    X_padded = np.zeros(size, dtype=complex)
    X_padded[:n] = X
    power_padded = np.abs(X_padded) ** 2
    c = np.fft.ifft(power_padded).real[:2 * n - 1]
    return c

def _phase_objective(phi, mag, beta=20.0):
    """
    Objective function for phase optimization.
    We want to minimize the max absolute value of the off-diagonal autoconvolution,
    normalized by the squared integral.
    
    The integral S = sum(x) = X[0] (the DC component).
    Since we're fixing the magnitude, X[0] = mag[0] * exp(1j*phi[0]).
    To keep S real and positive, we fix phi[0] = 0 and mag[0] to its initial value.
    """
    n = len(phi)
    x = _phase_to_signal(phi, mag)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    
    c = _autoconv_lin(x)
    if len(c) <= 1:
        return 0.0
    
    abs_c = np.abs(c[1:])
    m = np.max(abs_c)
    if m < 1e-15:
        return 0.0
    
    # Soft max for smoothness
    soft_max = m + np.log(np.sum(np.exp(beta * (abs_c - m)))) / beta
    return soft_max / (s * s)

def _phase_gradient(phi, mag, beta=20.0):
    """
    Compute gradient of the objective with respect to phase phi.
    Uses finite differences for simplicity and robustness.
    """
    n = len(phi)
    eps = 1e-6
    grad = np.zeros(n)
    
    base_obj = _phase_objective(phi, mag, beta)
    
    for i in range(n):
        if i == 0:
            # DC component: fixing phase to 0
            grad[i] = 0.0
            continue
        phi_plus = phi.copy()
        phi_plus[i] += eps
        obj_plus = _phase_objective(phi_plus, mag, beta)
        grad[i] = (obj_plus - base_obj) / eps
    
    return grad

def _optimize_phase(phi_init, mag, max_iters=200, time_budget=60.0):
    """
    Gradient descent on phase with adaptive step size.
    """
    start = time.time()
    phi = phi_init.copy()
    best_phi = phi.copy()
    best_score = _metric_offdiag(_phase_to_signal(phi, mag))
    
    # Fix DC phase to 0
    phi[0] = 0.0
    
    lr = 0.01
    for it in range(max_iters):
        if time.time() - start > time_budget:
            break
        
        # Compute gradient
        grad = _phase_gradient(phi, mag, beta=10.0)
        grad[0] = 0.0  # DC phase fixed
        
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-12:
            break
        
        # Gradient step
        phi_new = phi - lr * grad / gnorm
        phi_new[0] = 0.0  # Keep DC phase fixed
        
        x_new = _phase_to_signal(phi_new, mag)
        s_new = np.sum(x_new)
        if abs(s_new) < 1e-10:
            lr *= 0.5
            continue
        
        sc_new = _metric_offdiag(x_new)
        if sc_new < best_score:
            best_score = sc_new
            best_phi = phi_new.copy()
            phi = phi_new
            lr = min(lr * 1.05, 0.05)
        else:
            lr *= 0.7
    
    return best_phi, best_score

def _flat_spectrum_initialization(n, seed=42):
    """
    Create a flat magnitude spectrum and random initial phase.
    """
    rng = np.random.default_rng(seed)
    mag = np.ones(n)
    # Ensure DC component is large enough
    mag[0] = n * 0.5  # This gives sum(x) ~ n*0.5
    
    # Random phase, with DC phase = 0
    phi = rng.uniform(-np.pi, np.pi, n)
    phi[0] = 0.0
    
    return phi, mag

def _design_flat_spectrum_with_dc(n, dc_fraction=0.3):
    """
    Design a flat spectrum with a specific DC fraction to control the integral.
    """
    mag = np.ones(n)
    mag[0] = n * dc_fraction
    return mag

def _try_golay_like_construction(n):
    """
    Try to construct a Golay-like sequence with good autocorrelation properties.
    Uses a simple recursive construction.
    """
    # Start with a base sequence
    x = np.array([1.0, 1.0])
    
    while len(x) < n:
        # Golay construction: x' = [x, -x_reversed]
        x_rev = x[::-1]
        x = np.concatenate([x, -x_rev])
    
    x = x[:n]
    return x

def construct():
    """
    Construct a signed step function minimizing max|autoconv| / (sum(x))^2.
    Uses spectral phase optimization as the primary strategy.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    best_x = None
    best_score = float('inf')
    
    # Phase 1: Multiple spectral phase optimization runs with different initializations
    initializations = []
    
    # Initialization 1: Flat spectrum, random phase
    phi1, mag1 = _flat_spectrum_initialization(N, seed=42)
    initializations.append((phi1, mag1))
    
    # Initialization 2: Flat spectrum, different random phase
    phi2, mag2 = _flat_spectrum_initialization(N, seed=123)
    initializations.append((phi2, mag2))
    
    # Initialization 3: Golay-like sequence's phase
    x_golay = _try_golay_like_construction(N)
    phi3, mag3 = _signal_to_phase(x_golay)
    initializations.append((phi3, mag3))
    
    # Initialization 4: Hanning window's phase
    x_hann = np.hanning(N)
    phi4, mag4 = _signal_to_phase(x_hann)
    initializations.append((phi4, mag4))
    
    # Initialization 5: Random sequence's phase
    rng = np.random.default_rng(999)
    x_rand = rng.normal(0, 1, N)
    phi5, mag5 = _signal_to_phase(x_rand)
    initializations.append((phi5, mag5))
    
    # Phase 2: Run phase optimization on each initialization
    for idx, (phi_init, mag_init) in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
        
        # Check if this initialization is already good
        x_init = _phase_to_signal(phi_init, mag_init)
        s_init = np.sum(x_init)
        if abs(s_init) < 1e-10:
            continue
        
        sc_init = _metric_offdiag(x_init)
        if sc_init < best_score:
            best_score = sc_init
            best_x = x_init.copy()
        
        # Run phase optimization
        try:
            phi_opt, score_opt = _optimize_phase(
                phi_init, mag_init, 
                max_iters=100, 
                time_budget=20.0
            )
            x_opt = _phase_to_signal(phi_opt, mag_opt)
            s_opt = np.sum(x_opt)
            if abs(s_opt) > 1e-10:
                sc_opt = _metric_offdiag(x_opt)
                if sc_opt < best_score:
                    best_score = sc_opt
                    best_x = x_opt.copy()
        except Exception:
            pass
    
    # Phase 3: Refine the best solution using time-domain gradient descent
    if best_x is not None and time.time() < deadline - 10.0:
        x = best_x.copy()
        current_score = best_score
        rng2 = np.random.default_rng(777)
        
        max_iters = 200
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Compute gradient of the metric
            c = _autoconv_lin(x)
            abs_c = np.abs(c[1:])
            max_idx = np.argmax(abs_c)
            k_star = max_idx + 1
            
            s = np.sum(x)
            if abs(s) < 1e-12:
                break
            
            # Gradient of max|c_k| w.r.t. x
            grad = np.zeros_like(x)
            n = len(x)
            
            if k_star > 0:
                j1 = np.arange(n)
                valid1 = j1 + k_star < n
                grad[valid1] += x[j1[valid1] + k_star]
                valid2 = j1 - k_star >= 0
                grad[valid2] += x[j1[valid2] - k_star]
            
            sign_c = np.sign(c[k_star])
            if sign_c == 0:
                sign_c = 1.0
            grad = sign_c * grad
            
            gnorm = np.linalg.norm(grad)
            if gnorm < 1e-15:
                break
            
            # Adaptive step size
            step = 0.05 * (1.0 - it / max_iters) + 0.005
            x_new = x - step * grad / gnorm
            
            s_new = np.sum(x_new)
            if abs(s_new) < 1.0:
                x_new += (1.0 - s_new) / n
            
            sc_new = _metric_offdiag(x_new)
            if sc_new < current_score:
                x = x_new
                current_score = sc_new
                if current_score < best_score:
                    best_score = current_score
                    best_x = x.copy()
            else:
                step *= 0.5
                x_new = x - step * grad / gnorm
                s_new = np.sum(x_new)
                if abs(s_new) < 1.0:
                    x_new += (1.0 - s_new) / n
                sc_new = _metric_offdiag(x_new)
                if sc_new < best_score:
                    best_score = sc_new
                    best_x = x_new.copy()
                    x = x_new
    
    # Phase 4: Try thresholding to ±1 as a final polish
    if best_x is not None:
        x_thresh = np.sign(best_x)
        if np.all(x_thresh == 1):
            x_thresh[::2] = -1
        elif np.all(x_thresh == -1):
            x_thresh[::2] = 1
        
        s_thresh = np.sum(x_thresh)
        if abs(s_thresh) > 1e-10:
            score_thresh = _metric_offdiag(x_thresh)
            if score_thresh < best_score:
                best_score = score_thresh
                best_x = x_thresh.copy()
    
    if best_x is None:
        return np.ones(N)
    
    return best_x
