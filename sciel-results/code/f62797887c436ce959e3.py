"""
New strategy: Direct optimization of the autoconvolution peak via iterative
phase-only updates in the frequency domain, combined with a randomized
spectral-shaping approach. The key structural difference is that we
optimize the phases directly to flatten the autoconvolution, and we use
a different parameterization: we work with the full complex FFT and update
phases via a gradient-like rule derived from the autoconvolution structure.

Additionally, we use a multi-start approach with different initial spectral
shapes (not just flat magnitude) to escape local minima.
"""

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2


def _autoconv_max_abs(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))


def _integral(x):
    return float(np.sum(x))


def _metric(x):
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)


def _autoconv_values(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return conv[:2 * N - 1]


def _soft_max_abs(conv, beta=500.0):
    abs_conv = np.abs(conv)
    max_val = np.max(abs_conv)
    shifted = abs_conv - max_val
    return max_val + np.log(np.sum(np.exp(beta * shifted))) / beta


def _smooth_metric(x, beta=300.0):
    conv = _autoconv_values(x)
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _soft_max_abs(conv, beta) / (s * s)


def _phase_gradient_update(x, beta=200.0, lr=0.1):
    """
    Update the phases of the FFT to reduce the peak of the autoconvolution.
    This is a structural step: we compute the gradient of the soft-max of
    the autoconvolution with respect to the phases, and update accordingly.
    """
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = _autoconv_values(x)
    s = _integral(x)
    if abs(s) < 1e-12:
        return x
    
    abs_conv = np.abs(conv)
    max_val = np.max(abs_conv)
    shifted = abs_conv - max_val
    weights = np.exp(beta * shifted)
    weights /= np.sum(weights)
    
    # Compute gradient with respect to phases
    n_freq = len(fx)
    phase_grad = np.zeros(n_freq)
    
    for k in range(2 * N - 1):
        if weights[k] < 1e-10:
            continue
        sign_k = np.sign(conv[k]) if conv[k] != 0 else 1.0
        for m in range(1, n_freq):
            # Contribution to phase gradient from frequency m
            # The autoconvolution at lag k involves sum of x[j]*x[j+k]
            # In frequency domain, this relates to |X(m)|^2 * exp(i*phi)
            # Simplified: accumulate contribution
            phase_grad[m] += 2.0 * sign_k * weights[k] * np.imag(fx[m])
    
    # Normalize and apply
    phase_grad_norm = np.linalg.norm(phase_grad)
    if phase_grad_norm > 1e-15:
        phase_grad /= phase_grad_norm
    
    # Update phases
    for m in range(1, n_freq):
        if abs(fx[m]) > 1e-10:
            new_phase = np.angle(fx[m]) - lr * phase_grad[m]
            fx[m] = abs(fx[m]) * np.exp(1j * new_phase)
    
    x_new = np.fft.irfft(fx, _FFT_SIZE)[:N]
    return x_new


def _spectral_shaping(x_init, shape_type, rng):
    """
    Apply a spectral shaping to the initial signal. Different shapes
    provide diverse starting points.
    """
    fx = np.fft.rfft(x_init, _FFT_SIZE)
    n_freq = len(fx)
    k = np.arange(n_freq)
    
    if shape_type == 0:
        # Flat magnitude
        mags = np.ones(n_freq)
    elif shape_type == 1:
        # Linearly decreasing magnitude
        mags = 1.0 - 0.5 * k / (n_freq - 1)
    elif shape_type == 2:
        # Quadratic decreasing
        mags = 1.0 - 0.3 * (k / (n_freq - 1)) ** 2
    elif shape_type == 3:
        # Random magnitude with smoothing
        mags = np.exp(-0.5 * (k / (n_freq - 1)) ** 2)
    elif shape_type == 4:
        # Cosine taper
        mags = 0.5 * (1 + np.cos(np.pi * k / (n_freq - 1)))
    else:
        # Random smooth magnitude
        n_points = 20
        base = np.random.default_rng(rng.integers(0, 10000)).random(n_points)
        mags = np.interp(k, np.linspace(0, n_freq - 1, n_points), base)
        mags = np.maximum(mags, 0.1)
    
    # Preserve DC
    target_sum = _integral(x_init)
    mags[0] = abs(target_sum)
    
    # Random phases
    phases = rng.random(n_freq) * 2 * np.pi
    phases[0] = 0.0
    
    fx_new = mags * np.exp(1j * phases)
    x_new = np.fft.irfft(fx_new, _FFT_SIZE)[:N]
    
    # Normalize to have same integral
    s_new = _integral(x_new)
    if abs(s_new) > 1e-12:
        x_new = x_new * (target_sum / s_new)
    
    return x_new


def _iterative_phase_optimize(x_init, deadline, rng, max_iters=5000, beta=200.0, lr=0.1):
    """
    Iteratively update phases to minimize the soft-max of autoconvolution.
    """
    best_x = x_init.copy()
    best_val = _metric(best_x)
    current_x = x_init.copy()
    
    patience = 0
    best_smooth = _smooth_metric(current_x, beta)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Phase update
        x_new = _phase_gradient_update(current_x, beta, lr)
        
        # Also try a small perturbation
        x_pert = x_new + rng.normal(0, 0.01, size=N)
        
        v_new = _metric(x_new)
        v_pert = _metric(x_pert)
        
        if v_new < best_val:
            best_val = v_new
            best_x = x_new.copy()
            current_x = x_new
            patience = 0
        elif v_pert < best_val:
            best_val = v_pert
            best_x = x_pert.copy()
            current_x = x_pert
            patience = 0
        else:
            patience += 1
            if patience > 50:
                lr *= 0.9
                patience = 0
                if lr < 0.001:
                    break
    
    return best_x, best_val


def _local_search_fast(x, deadline, rng, max_rounds=50000):
    """Fast local search with in-place modifications."""
    best_x = x.copy()
    best_val = _metric(best_x)
    no_improve = 0
    step_scale = 0.15
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        if no_improve > 100:
            step_scale *= 0.9
            no_improve = 0
            if step_scale < 0.001:
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
    """High-precision final polish with coordinate descent."""
    best_x = x.copy()
    best_val = _metric(best_x)
    
    for sweep in range(20):
        if time.time() > deadline:
            break
        
        improved = False
        for idx in range(N):
            if time.time() > deadline:
                break
            
            lo, hi = -0.03, 0.03
            best_delta = 0.0
            best_v = best_val
            
            for _ in range(8):
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
    
    rng = np.random.default_rng(42)
    
    # Multi-start with different spectral shapes
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    n_starts = 5
    time_per_start = (deadline - time.time()) / (n_starts + 2)
    
    for start_idx in range(n_starts):
        if time.time() > deadline - 30:
            break
        
        # Generate diverse starting points
        x_init = _spectral_shaping(np.ones(N), start_idx, rng)
        
        # Phase optimization
        x_opt, val_opt = _iterative_phase_optimize(
            x_init, 
            min(deadline - 25, time.time() + time_per_start * 0.6),
            rng,
            max_iters=3000,
            beta=150 + start_idx * 20,
            lr=0.08
        )
        
        if val_opt < best_val:
            best_val = val_opt
            best_x = x_opt
    
    # Local search
    if time.time() < deadline - 15:
        best_x, best_val = _local_search_fast(best_x, deadline - 15, rng, max_rounds=40000)
    
    # Final polish
    if time.time() < deadline - 2:
        best_x, best_val = _final_polish(best_x, deadline - 2, rng)
    
    return best_x
# EVOLVE-BLOCK-END
