"""Phase-only optimization with fixed amplitude spectrum for C3' minimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 145

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - 3)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective_only(x):
    """Compute objective value only (fast, no gradient)."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    return m / (s * s)

def _soft_max_and_grad(x, beta):
    """
    Compute soft-max surrogate for max|ac| and its gradient.
    
    The autoconvolution ac[k] = sum_i x[i]*x[i+k] for k >= 0
    and ac[k] = ac[-k] for k < 0 (even symmetry).
    
    For the full linear autoconvolution of length 2N-1:
    ac_full[m] = sum_{i} x[i] * x[i + m - (N-1)] where valid indices
    
    Gradient of ac_full[m] w.r.t. x[j]:
    d(ac_full[m])/dx[j] = x[j + m - (N-1)] + x[j - m + (N-1)]
    (with appropriate boundary handling)
    
    Using FFT: if G is the gradient weights on ac_full, then
    grad_x[j] = sum_m G[m] * (x[j+m-(N-1)] + x[j-m+(N-1)])
    This can be computed as a convolution/correlation.
    """
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    m = np.max(abs_ac)
    exps = np.exp(beta * (abs_ac - m))
    log_sum = np.log(np.sum(exps))
    soft_val = m + log_sum / beta
    
    weights = exps / np.sum(exps)
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    grad_ac = weights * signs  # length 2N-1
    
    # Compute grad_x via FFT correlation
    # grad_x[j] = sum_m grad_ac[m] * d(ac_full[m])/dx[j]
    # d(ac_full[m])/dx[j] = x[j+m-(N-1)] + x[j-m+(N-1)]
    # 
    # Let's define h[m] = grad_ac[m-N+1] for m in range(2N-1), shifted to be centered
    # Then grad_x[j] = sum_k h[k] * (x[j+k] + x[j-k])
    # = (h * x)[j] + (h_rev * x)[j] where h_rev[k] = h[-k]
    
    # Center grad_ac: index m from 0 to 2N-2, center at N-1
    # h[k] = grad_ac[k + N - 1] for k in range(-(N-1), N)
    h = np.zeros(2 * N - 1)
    h[N-1 : 2*N-1] = grad_ac[0 : N]
    h[0 : N-1] = grad_ac[2*N-2 : N-1 : -1]
    
    # Now compute correlation: grad_x[j] = sum_k h[k-(N-1)] * (x[j+k-(N-1)] + x[j-k+(N-1)])
    # This is equivalent to: convolve x with h and h_reversed
    
    L_corr = 1
    while L_corr < 4 * N:
        L_corr *= 2
    
    X_f = np.fft.rfft(x, n=L_corr)
    H_f = np.fft.rfft(h, n=L_corr)
    
    # Forward correlation: sum_k h[k] * x[j-k] -> this is (h * x)[j] in correlation sense
    # Using FFT: corr1 = ifft(conj(H_f) * X_f)
    corr1 = np.fft.irfft(np.conj(H_f) * X_f, n=L_corr)[:N]
    
    # Reverse: h_rev[k] = h[-k], so corr2[j] = sum_k h[-k] * x[j-k] = sum_k h[k] * x[j+k]
    # h_rev in FFT: conj(H_f) gives ifft -> h_rev in time
    # Actually, let's just flip h and correlate
    h_rev = h[::-1]
    H_rev_f = np.fft.rfft(h_rev, n=L_corr)
    corr2 = np.fft.irfft(np.conj(H_rev_f) * X_f, n=L_corr)[:N]
    
    grad_x = corr1 + corr2
    return soft_val, grad_x

def _objective_and_grad(x, beta):
    """Compute objective and gradient."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    soft_val, grad_soft = _soft_max_and_grad(x, beta)
    val = soft_val / (s * s)
    grad = grad_soft / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s)
    return val, grad

def _spectral_init(rng, mode=0):
    """
    Generate initial x via spectral phase retrieval.
    mode 0: flat-top magnitude spectrum
    mode 1: raised-cosine magnitude spectrum
    mode 2: random phase with fixed magnitude
    mode 3: Gaussian-tapered magnitude
    mode 4: low-pass with sharp cutoff
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    freqs = np.fft.rfftfreq(L_fft)
    n_freq = L_fft // 2 + 1
    
    if mode == 0:
        # Flat-top window in frequency domain
        mag = np.ones(n_freq)
        taper = np.minimum(1.0, np.abs(freqs) * L_fft / 5)
        mag *= taper
    elif mode == 1:
        # Raised cosine
        mag = 0.5 + 0.5 * np.cos(2 * np.pi * freqs * L_fft * 0.5)
    elif mode == 2:
        # Random magnitude with some structure
        mag = 1.0 + 0.3 * rng.normal(0, 1, size=n_freq)
        mag = np.maximum(0.1, mag)
    elif mode == 3:
        # Gaussian taper
        mag = np.exp(-20 * (freqs * L_fft / 2) ** 2)
    elif mode == 4:
        # Low-pass with sharp cutoff at 0.4
        cutoff = 0.4
        mag = np.where(np.abs(freqs) < cutoff, 1.0, 0.1)
        # Smooth transition
        transition = 0.05
        mag = np.where(np.abs(freqs) < cutoff - transition, 1.0,
                       np.where(np.abs(freqs) < cutoff + transition,
                                0.5 * (1 + np.cos(np.pi * (np.abs(freqs) - cutoff) / transition)),
                                0.1))
    else:
        mag = np.ones(n_freq)
    
    # Random phase
    phase = rng.uniform(0, 2 * np.pi, size=n_freq)
    phase[0] = 0  # DC component real
    
    # Build spectrum
    spectrum = mag * np.exp(1j * phase)
    spectrum[0] = mag[0]  # Keep DC real and positive
    
    # Inverse FFT to get time domain
    x_full = np.fft.irfft(spectrum, n=L_fft)
    x = x_full[:N]
    
    # Normalize so sum is positive
    s = np.sum(x)
    if s < 0:
        x = -x
    
    return x

def _amplitude_optimize(x0, rng, iters=30, beta=50.0):
    """
    Optimize the amplitude spectrum while keeping phase fixed.
    This is a different parameterization that can escape local minima.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    n_freq = L_fft // 2 + 1
    
    # Get current spectrum
    X = np.fft.rfft(x0, n=L_fft)
    phase = np.angle(X)
    phase[0] = 0
    amp = np.abs(X).copy()
    
    best_amp = amp.copy()
    best_val = _objective_only(x0)
    
    for it in range(iters):
        if _check_deadline():
            break
        
        # Perturb amplitudes
        amp_trial = amp * (1 + 0.05 * rng.normal(0, 1, size=n_freq))
        amp_trial = np.maximum(0.01, amp_trial)
        
        # Reconstruct
        spectrum = amp_trial * np.exp(1j * phase)
        spectrum[0] = amp_trial[0]
        x_trial = np.fft.irfft(spectrum, n=L_fft)[:N]
        
        s = np.sum(x_trial)
        if s < 0:
            x_trial = -x_trial
        
        val = _objective_only(x_trial)
        if val < best_val:
            best_val = val
            amp = amp_trial.copy()
            best_amp = amp.copy()
    
    # Reconstruct best
    spectrum = best_amp * np.exp(1j * phase)
    spectrum[0] = best_amp[0]
    x_best = np.fft.irfft(spectrum, n=L_fft)[:N]
    s = np.sum(x_best)
    if s < 0:
        x_best = -x_best
    
    return x_best, best_val

def construct():
    best_x = np.ones(N)
    best_val = _objective_only(best_x)
    
    if _check_deadline():
        return best_x
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial points
    t = np.linspace(0, 1, N)
    initial_points = [
        np.ones(N),
        1.0 + 0.3 * np.sin(2 * np.pi * t),
        1.0 + 0.5 * np.cos(2 * np.pi * t),
        1.0 + 0.3 * np.sin(4 * np.pi * t),
        1.0 + 0.5 * np.cos(4 * np.pi * t),
        np.exp(1.5 * t),
        np.exp(-1.5 * t),
        np.exp(0.5 * t),
        np.exp(-0.5 * t),
        1.0 + 0.5 * (t - 0.5),
        1.0 + 0.8 * (t - 0.5)**2,
        1.0 + 0.5 * np.sin(6 * np.pi * t) * np.cos(2 * np.pi * t),
        1.0 + 0.3 * (t - 0.5)**3,
        np.maximum(0.5, 1.0 - 2.0 * np.abs(t - 0.5)),
        np.maximum(0.5, 1.0 - 3.0 * np.abs(t - 0.5)),
        1.0 + 0.4 * np.sin(8 * np.pi * t),
        1.0 + 0.4 * np.cos(10 * np.pi * t),
        np.maximum(0.3, 1.0 + 0.6 * np.sin(2 * np.pi * 3 * t)),
        # Add more structured initializations
        1.0 + 0.4 * np.sin(2 * np.pi * 5 * t) * np.cos(2 * np.pi * t),
        1.0 + 0.3 * np.cos(2 * np.pi * 7 * t),
        np.maximum(0.2, 1.0 - 4.0 * np.abs(t - 0.5)),
        1.0 + 0.6 * np.sin(2 * np.pi * 2 * t) * np.sin(2 * np.pi * 3 * t),
    ]
    
    # Add spectral initializations (more modes)
    for mode in range(5):
        for _ in range(4):
            initial_points.append(_spectral_init(rng, mode))
    
    # Add random initializations
    for _ in range(10):
        scale = rng.uniform(0.5, 1.5)
        initial_points.append(scale * rng.normal(1.0, 0.2, size=N))
    
    # L-BFGS optimization from each initial point with adaptive beta
    # Use lower beta for exploration, higher for exploitation
    betas_explore = [20.0, 40.0]
    betas_exploit = [80.0, 150.0]
    
    # Phase 1: Exploration with lower beta
    for x0 in initial_points:
        if _check_deadline():
            break
        
        x_start = x0 + rng.normal(0, 0.02, size=N)
        
        for beta in betas_explore:
            if _check_deadline():
                break
            
            try:
                def obj_and_grad_func(x, _beta=beta):
                    return _objective_and_grad(x, _beta)
                
                result = minimize(obj_and_grad_func, x_start, method='L-BFGS-B',
                                  jac=True, options={'maxiter': 150, 'gtol': 1e-8, 'ftol': 1e-10})
                
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x.copy()
                    
            except Exception:
                pass
    
    if _check_deadline():
        return best_x
    
    # Phase 2: Exploitation with higher beta from best solution
    for beta in betas_exploit:
        if _check_deadline():
            break
        try:
            def obj_and_grad_func(x, _beta=beta):
                return _objective_and_grad(x, _beta)
            
            result = minimize(obj_and_grad_func, best_x, method='L-BFGS-B',
                              jac=True, options={'maxiter': 200, 'gtol': 1e-10, 'ftol': 1e-12})
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
                
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Amplitude optimization (different parameterization)
    for _ in range(6):
        if _check_deadline():
            break
        x_amp, val_amp = _amplitude_optimize(best_x, rng, iters=50, beta=50.0)
        if val_amp < best_val:
            best_val = val_amp
            best_x = x_amp.copy()
    
    if _check_deadline():
        return best_x
    
    # Final polish with very high beta
    for beta in [300.0, 600.0]:
        if _check_deadline():
            break
        try:
            def obj_and_grad_func(x, _beta=beta):
                return _objective_and_grad(x, _beta)
            
            result = minimize(obj_and_grad_func, best_x, method='L-BFGS-B',
                              jac=True, options={'maxiter': 300, 'gtol': 1e-12, 'ftol': 1e-14})
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
                
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Coordinate descent refinement with adaptive step size
    step_size = 0.005
    no_improve = 0
    max_no_improve = 5000
    patience = 0
    
    while no_improve < max_no_improve and not _check_deadline():
        n_perturb = rng.integers(2, 30)
        indices = rng.choice(N, size=n_perturb, replace=False)
        perturbation = np.zeros(N)
        perturbation[indices] = rng.normal(0, step_size, size=n_perturb)
        x_trial = best_x + perturbation
        val = _objective_only(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            step_size = max(0.0005, step_size * 0.95)
            no_improve = 0
            patience = 0
        else:
            no_improve += 1
            patience += 1
            if patience > 500:
                step_size = min(0.02, step_size * 1.02)
                patience = 0
            else:
                step_size = min(0.02, step_size * 1.01)
    
    return best_x
# EVOLVE-BLOCK-END
