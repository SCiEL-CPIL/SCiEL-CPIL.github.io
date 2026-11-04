"""
Combined approach: Spectral parameterization (flat spectrum design) + smooth optimization polish.

Key insight: A sequence with a flat power spectrum has an autoconvolution that is
approximately a delta function (peak at lag 0, near-zero elsewhere). By parameterizing
the sequence in the frequency domain as x = IFFT(|X|^2 * exp(i*phase)), we reduce the
effective dimensionality and make it easier to find designs where the off-diagonal
autoconvolution terms are small relative to the squared integral.

Strategy:
1. Generate candidates with flat or near-flat magnitude spectra and optimized phases.
2. Use smooth objective (soft-max) + L-BFGS-B to refine.
3. Final gradient descent polish.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _autoconv(x):
    """Linear autoconvolution via FFT."""
    n = len(x)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    X = np.fft.rfft(x, fsize)
    c = np.fft.irfft(X * X, fsize)[:size]
    return c

def _metric_offdiag(x):
    """Compute max |c_k| for k != 0 divided by (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv(x)
    if len(c) > 1:
        max_val = np.max(np.abs(c[1:]))
    else:
        max_val = 0.0
    return max_val / (s * s)

def _objective_smooth(x, temp=1.0):
    """Smooth approximation of the off-diagonal metric for optimization."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    c = _autoconv(x)
    if len(c) > 1:
        abs_c = np.abs(c[1:])
    else:
        return 0.0
    max_c = np.max(abs_c)
    if max_c < 1e-15:
        return 0.0
    soft_max = max_c + temp * np.log(np.sum(np.exp((abs_c - max_c) / temp)))
    return soft_max / (s * s)

def _flat_spectrum_candidate(mag_profile, phases, rng):
    """
    Build a real sequence from a specified magnitude spectrum and phases.
    mag_profile: array of length N//2+1 (rfft size)
    phases: array of same length
    """
    n_half = N // 2 + 1
    # Build complex spectrum
    X = np.zeros(n_half, dtype=complex)
    for k in range(n_half):
        X[k] = mag_profile[k] * np.exp(1j * phases[k])
    # For real signal, need conjugate symmetry in full spectrum
    # rfft gives first half; irfft reconstructs assuming symmetry
    x = np.fft.irfft(X, n=N)
    return x

def _optimize_phases_flat(mag_profile, max_iter=300, deadline=None):
    """
    Optimize phases to minimize off-diagonal autoconvolution, given fixed magnitudes.
    Returns optimized phase array and the resulting sequence.
    """
    n_half = N // 2 + 1
    
    # Initial random phases
    rng = np.random.default_rng(123)
    phases = rng.uniform(-np.pi, np.pi, n_half)
    
    def phase_obj(ph):
        x = _flat_spectrum_candidate(mag_profile, ph, None)
        return _metric_offdiag(x)
    
    # Use L-BFGS-B on the phase vector
    best_phases = phases.copy()
    best_val = phase_obj(phases)
    
    try:
        res = minimize(
            phase_obj,
            phases,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if res.fun < best_val:
            best_val = res.fun
            best_phases = res.x
    except Exception:
        pass
    
    x = _flat_spectrum_candidate(mag_profile, best_phases, None)
    return x, best_val

def _gradient_descent_polish(x0, max_iter=500, initial_lr=0.01, deadline=None):
    """Projected gradient descent with annealed temperature for final polish."""
    x = x0.copy().astype(float)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return x
    
    # Normalize sum to 1
    x = x / s
    
    best_x = x.copy()
    best_val = _metric_offdiag(best_x)
    
    lr = initial_lr
    temp = 0.5
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        c = _autoconv(x)
        if len(c) <= 1:
            break
        abs_c = np.abs(c[1:])
        
        max_c = np.max(abs_c)
        if max_c < 1e-15:
            break
        
        exp_vals = np.exp((abs_c - max_c) / temp)
        sum_exp = np.sum(exp_vals)
        soft_max = max_c + temp * np.log(sum_exp)
        
        grad_abs_c = exp_vals / sum_exp
        c_off = c[1:]
        sign_c = np.sign(c_off)
        grad_c = grad_abs_c * sign_c
        
        w = np.zeros(2 * N - 1)
        w[1:] = grad_c
        
        fsize = 1
        while fsize < 2 * N:
            fsize *= 2
        X = np.fft.rfft(x, fsize)
        W = np.fft.rfft(w, fsize)
        grad_x = np.fft.irfft(W * np.conj(X), fsize)[:N]
        
        s = np.sum(x)
        metric = soft_max / (s * s)
        
        grad_metric = grad_x / (s * s) - 2.0 * soft_max * np.ones(N) / (s ** 3)
        
        x_new = x - lr * grad_metric
        
        s_new = np.sum(x_new)
        if abs(s_new) < 1e-12:
            x_new = x + 1e-6 * np.random.randn(N)
            s_new = np.sum(x_new)
        x_new = x_new / s_new
        
        val = _metric_offdiag(x_new)
        if val < best_val:
            best_val = val
            best_x = x_new.copy()
        
        x = x_new
        lr *= 0.995
        temp *= 0.999
        
    return best_x

def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    
    best_x = np.ones(N)
    best_val = _metric_offdiag(best_x)
    
    rng = np.random.default_rng(42)
    n_half = N // 2 + 1
    
    candidates = []
    
    # 1. All ones (baseline)
    candidates.append(np.ones(N))
    
    # 2. Flat magnitude spectrum with random phases (multiple tries)
    for trial in range(8):
        if time.time() > deadline - 60:
            break
        
        # Uniform magnitude
        mag = np.ones(n_half)
        x, val = _optimize_phases_flat(mag, max_iter=200, deadline=deadline - 60)
        if val < best_val:
            best_val = val
            best_x = x.copy()
        candidates.append(x)
    
    # 3. Tapered magnitude spectrum (less sharp transitions)
    for taper in [0.5, 0.8, 1.5]:
        if time.time() > deadline - 40:
            break
        
        # Smooth taper
        k = np.arange(n_half)
        mag = 1.0 / (1.0 + (k / n_half) ** taper)
        x, val = _optimize_phases_flat(mag, max_iter=200, deadline=deadline - 40)
        if val < best_val:
            best_val = val
            best_x = x.copy()
        candidates.append(x)
    
    # 4. Random binary sequences with non-zero mean
    for _ in range(10):
        if time.time() > deadline - 30:
            break
        r = rng.random(N)
        x = np.where(r > 0.35, 1.0, -1.0)
        if abs(np.sum(x)) > 5:
            candidates.append(x)
    
    # 5. Block patterns
    for block_size in [10, 20, 50]:
        if time.time() > deadline - 20:
            break
        pattern = np.zeros(N)
        for i in range(N):
            if (i // block_size) % 2 == 0:
                pattern[i] = 1.0
            else:
                pattern[i] = -1.0
        pattern[0] += 0.5
        candidates.append(pattern)
    
    # Optimize top candidates with L-BFGS-B on smooth objective
    # Sort by current metric to prioritize best
    scored = [(c, _metric_offdiag(c)) for c in candidates]
    scored.sort(key=lambda t: t[1])
    
    for x0, cur_val in scored[:10]:
        if time.time() > deadline - 25:
            break
        
        x0 = x0.copy()
        s0 = np.sum(x0)
        if abs(s0) < 1e-6:
            continue
        x0 = x0 / s0
        
        try:
            # First pass with higher temperature
            res1 = minimize(
                lambda x: _objective_smooth(x, temp=1.0),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            x_mid = res1.x
            
            # Second pass with lower temperature
            res2 = minimize(
                lambda x: _objective_smooth(x, temp=0.1),
                x_mid,
                method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            x_opt = res2.x
            
            val = _metric_offdiag(x_opt)
            if val < best_val:
                best_val = val
                best_x = x_opt.copy()
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 5:
        try:
            polished = _gradient_descent_polish(
                best_x,
                max_iter=800,
                initial_lr=0.005,
                deadline=deadline - 5
            )
            val = _metric_offdiag(polished)
            if val < best_val:
                best_val = val
                best_x = polished
        except Exception:
            pass
    
    return best_x
