"""Model-and-optimize: spectral phase retrieval + L-BFGS + coordinate descent + spectral amplitude polish."""
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
    
    ac[k] = sum_{i,j: i+j=k} x[i]*x[j] for k=0..2N-2
    d(soft_max)/dx[n] = sum_k grad_ac[k] * d(ac[k])/dx[n]
    d(ac[k])/dx[n] = sum_{i: i+k-i in [0,N-1]} x[k-i] + sum_{j: n+j in [0,N-1]} x[j]
                   = (x * x)[k] reversed + (x * x)[k] ... 
    More precisely: ac[k] = sum_{i=max(0,k-N+1)}^{min(k,N-1)} x[i]*x[k-i]
    d(ac[k])/dx[n] = x[k-n] (if k-n in [0,N-1]) + x[n] (if k-n in [0,N-1])
                   = 2*x[k-n] if 0<=k-n<N, else x[k-n] if n==0 or n==k... 
    Actually: d(ac[k])/dx[n] = x[k-n] + x[n-k]... no.
    
    ac[k] = sum_i x[i]*x[k-i]
    d/dx[n] ac[k] = x[k-n] (from first factor) + x[n] (from second factor, when k-i=n => i=k-n)
    Wait: d/dx[n] sum_i x[i]*x[k-i] = x[k-n] (when i=n, derivative of x[i] is delta_{i,n}) 
           + sum_i x[i]*d/dx[n] x[k-i] = x[k-n] + x[k-n] (when k-i=n, i=k-n)
    = 2*x[k-n] if 0 <= k-n < N, else 0 (with boundary care)
    
    So grad_x[n] = sum_k grad_ac[k] * 2 * x[k-n] (when 0<=k-n<N)
    = 2 * (grad_ac * x)[n] where * is cross-correlation
    
    This is a single FFT cross-correlation!
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
    
    # grad_x[n] = sum_k grad_ac[k] * 2 * x[k-n] for valid k-n
    # This is 2 * cross_correlation(grad_ac, x) evaluated at appropriate indices
    # cross_correlation(a, b)[n] = sum_k a[k] * b[k-n]
    # We need: sum_k grad_ac[k] * x[k-n] for k in [0, 2N-2], k-n in [0, N-1]
    # = sum_k grad_ac[k] * x[k-n] * indicator(0<=k-n<N)
    # This is the cross-correlation of grad_ac (padded) with x at lag n
    
    L_corr = 1
    while L_corr < 4 * N:
        L_corr *= 2
    
    # Pad grad_ac to length L_corr, zero-pad x
    g_padded = np.zeros(L_corr)
    g_padded[:2*N-1] = grad_ac
    
    X_f = np.fft.rfft(x, n=L_corr)
    G_f = np.fft.rfft(g_padded, n=L_corr)
    
    # Cross-correlation: sum_k g_padded[k] * x[k-n]
    # = irfft(G_f * conj(X_f)) gives sum_k g_padded[k] * x[(k-n) mod L]
    # We need the "true" cross-correlation without wraparound
    # Since g_padded has support [0, 2N-2] and x has support [0, N-1],
    # and L_corr >= 4N > 2N-1 + N-1, there's no wraparound issue.
    
    corr = np.fft.irfft(G_f * np.conj(X_f), n=L_corr)
    # corr[n] = sum_k g_padded[k] * x[k-n] (with x zero outside [0,N-1])
    # We want grad_x[n] = 2 * corr[n] for n = 0..N-1
    
    grad_x = 2.0 * corr[:N]
    
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
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    if mode == 0:
        freqs = np.fft.rfftfreq(L_fft)
        mag = np.ones(L_fft // 2 + 1)
        taper = np.minimum(1.0, np.abs(freqs) * L_fft / 5)
        mag *= taper
    elif mode == 1:
        freqs = np.fft.rfftfreq(L_fft)
        mag = 0.5 + 0.5 * np.cos(2 * np.pi * freqs * L_fft * 0.5)
    elif mode == 2:
        mag = 1.0 + 0.3 * rng.normal(0, 1, size=L_fft // 2 + 1)
        mag = np.maximum(0.1, mag)
    elif mode == 3:
        freqs = np.fft.rfftfreq(L_fft)
        mag = np.exp(-20 * (freqs * L_fft / 2) ** 2)
    elif mode == 4:
        freqs = np.fft.rfftfreq(L_fft)
        cutoff = 0.4
        transition = 0.05
        mag = np.where(np.abs(freqs) < cutoff - transition, 1.0,
                       np.where(np.abs(freqs) < cutoff + transition,
                                0.5 * (1 + np.cos(np.pi * (np.abs(freqs) - cutoff) / transition)),
                                0.1))
    else:
        mag = np.ones(L_fft // 2 + 1)
    
    phase = rng.uniform(0, 2 * np.pi, size=L_fft // 2 + 1)
    phase[0] = 0
    
    spectrum = mag * np.exp(1j * phase)
    spectrum[0] = mag[0]
    
    x_full = np.fft.irfft(spectrum, n=L_fft)
    x = x_full[:N]
    
    s = np.sum(x)
    if s < 0:
        x = -x
    
    return x

def _spectral_phase_optimize(x0, rng, max_iters=200, lr=0.01):
    """
    Optimize the phase of the spectrum while keeping magnitude fixed.
    This enforces the step-function constraint in the time domain by
    projecting back to a valid time-domain signal after each update.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    n_freq = L_fft // 2 + 1
    
    # Get current spectrum
    X = np.fft.rfft(x0, n=L_fft)
    mag = np.abs(X).copy()
    phase = np.angle(X).copy()
    phase[0] = 0
    
    best_x = x0.copy()
    best_val = _objective_only(x0)
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        # Compute gradient in time domain
        val, grad = _objective_and_grad(best_x, 50.0)
        
        # Transform gradient to frequency domain
        G = np.fft.rfft(grad, n=L_fft)
        
        # Update phase: phase_new = phase - lr * imag(X * conj(G)) / |X|
        # This is the gradient of the objective w.r.t. phase
        X_curr = mag * np.exp(1j * phase)
        phase_grad = np.imag(X_curr * np.conj(G)) / (mag + 1e-10)
        phase_grad[0] = 0  # Keep DC real
        
        # Update phase
        phase -= lr * phase_grad
        
        # Reconstruct time domain
        spectrum = mag * np.exp(1j * phase)
        spectrum[0] = mag[0]
        x_trial = np.fft.irfft(spectrum, n=L_fft)[:N]
        
        s = np.sum(x_trial)
        if s < 0:
            x_trial = -x_trial
            phase = -phase
            phase[0] = 0
        
        val_trial = _objective_only(x_trial)
        
        if val_trial < best_val:
            best_val = val_trial
            best_x = x_trial.copy()
            
            # Adaptive learning rate
            if it % 20 == 0 and it > 0:
                lr = max(0.001, lr * 0.95)
        else:
            # Revert and reduce learning rate
            phase += lr * phase_grad
            lr = max(0.001, lr * 0.8)
    
    return best_x, best_val

def _amplitude_optimize(x0, rng, iters=40, beta=50.0):
    """
    Optimize the amplitude spectrum while keeping phase fixed.
    This is a different parameterization that can escape local minima.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    n_freq = L_fft // 2 + 1
    
    X = np.fft.rfft(x0, n=L_fft)
    phase = np.angle(X)
    phase[0] = 0
    amp = np.abs(X).copy()
    
    best_amp = amp.copy()
    best_val = _objective_only(x0)
    
    for it in range(iters):
        if _check_deadline():
            break
        
        amp_trial = amp * (1 + 0.05 * rng.normal(0, 1, size=n_freq))
        amp_trial = np.maximum(0.01, amp_trial)
        
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
    
    spectrum = best_amp * np.exp(1j * phase)
    spectrum[0] = best_amp[0]
    x_best = np.fft.irfft(spectrum, n=L_fft)[:N]
    s = np.sum(x_best)
    if s < 0:
        x_best = -x_best
    
    return x_best, best_val

def _peak_targeted_polish(x0, rng, max_iters=800, step_size=0.01):
    """
    Identify dominant autoconvolution lags and apply targeted perturbations
    to reduce the peak absolute value. Vectorized contribution computation.
    """
    best_x = x0.copy()
    best_val = _objective_only(x0)
    ac = _autoconv_fft(best_x)
    abs_ac = np.abs(ac)
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        # Recompute autoconvolution periodically to track peak shifts
        if it % 20 == 0:
            ac = _autoconv_fft(best_x)
            abs_ac = np.abs(ac)
        
        # Find the top k lags contributing to the peak
        top_k = 8
        sorted_idx = np.argsort(-abs_ac)[:top_k]
        
        # For each dominant lag, try to reduce it
        improved = False
        for k in sorted_idx:
            if _check_deadline():
                break
            
            # Vectorized computation of d(ac[k])/dx
            # d(ac[k])/dx[n] = 2*x[k-n] for 0<=k-n<N, with boundary handling
            # For n in [0, N-1], k-n is valid when 0 <= k-n < N
            # i.e., k-N+1 <= n <= k
            n_lo = max(0, k - N + 1)
            n_hi = min(k, N - 1)
            
            if n_hi >= n_lo:
                # contrib[n] = 2*x[k-n] for n in [n_lo, n_hi]
                # k-n ranges from k-n_lo to k-n_hi
                idx_in_x = k - np.arange(n_lo, n_hi + 1)
                contrib_segment = 2.0 * best_x[idx_in_x]
                
                # Build full contribution vector
                contrib = np.zeros(N)
                contrib[n_lo:n_hi+1] = contrib_segment
                
                norm_contrib = np.linalg.norm(contrib)
                if norm_contrib > 1e-10:
                    direction = -contrib / norm_contrib
                    
                    # Try a few step sizes
                    for scale in [step_size, step_size * 0.5, step_size * 1.5, step_size * 2.0]:
                        x_trial = best_x + scale * direction
                        val = _objective_only(x_trial)
                        if val < best_val:
                            best_val = val
                            best_x = x_trial.copy()
                            improved = True
                            break
        
        if improved:
            step_size = max(0.001, step_size * 0.93)
        else:
            step_size = min(0.08, step_size * 1.03)
    
    return best_x, best_val

def construct():
    best_x = np.ones(N)
    best_val = _objective_only(best_x)
    
    if _check_deadline():
        return best_x
    
    rng = np.random.default_rng(42)
    
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
    ]
    
    for mode in range(5):
        for _ in range(3):
            initial_points.append(_spectral_init(rng, mode))
    
    for _ in range(8):
        scale = rng.uniform(0.5, 1.5)
        initial_points.append(scale * rng.normal(1.0, 0.2, size=N))
    
    betas = [40.0, 100.0]
    
    for x0 in initial_points:
        if _check_deadline():
            break
        
        x_start = x0 + rng.normal(0, 0.01, size=N)
        
        for beta in betas:
            if _check_deadline():
                break
            
            try:
                def obj_and_grad_func(x, _beta=beta):
                    return _objective_and_grad(x, _beta)
                
                result = minimize(obj_and_grad_func, x_start, method='L-BFGS-B',
                                  jac=True, options={'maxiter': 300, 'gtol': 1e-9, 'ftol': 1e-12, 'maxls': 50})
                
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x.copy()
                    
            except Exception:
                pass
    
    if _check_deadline():
        return best_x
    
    # Spectral phase optimization on the best solution
    x_phase, val_phase = _spectral_phase_optimize(best_x, rng, max_iters=200, lr=0.015)
    if val_phase < best_val:
        best_val = val_phase
        best_x = x_phase.copy()
    
    if _check_deadline():
        return best_x
    
    # Amplitude optimization
    for _ in range(3):
        if _check_deadline():
            break
        x_amp, val_amp = _amplitude_optimize(best_x, rng, iters=30)
        if val_amp < best_val:
            best_val = val_amp
            best_x = x_amp.copy()
    
    if _check_deadline():
        return best_x
    
    # Peak-targeted polish
    x_peak, val_peak = _peak_targeted_polish(best_x, rng, max_iters=500, step_size=0.01)
    if val_peak < best_val:
        best_val = val_peak
        best_x = x_peak.copy()
    
    if _check_deadline():
        return best_x
    
    # Polish with higher beta L-BFGS
    for beta in [200.0, 500.0, 1000.0, 2000.0, 4000.0]:
        if _check_deadline():
            break
        try:
            def obj_and_grad_func(x, _beta=beta):
                return _objective_and_grad(x, _beta)
            
            result = minimize(obj_and_grad_func, best_x, method='L-BFGS-B',
                              jac=True, options={'maxiter': 300, 'gtol': 1e-11, 'ftol': 1e-14, 'maxls': 40})
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
                
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Coordinate descent refinement with structured perturbations
    step_size = 0.02
    no_improve = 0
    max_no_improve = 12000
    best_ever = best_val
    t_arr = np.arange(N)
    
    while no_improve < max_no_improve and not _check_deadline():
        # Mix of random perturbations and structured ones
        r = rng.random()
        if r < 0.2:
            # Single coordinate
            idx = rng.integers(0, N)
            delta = step_size * rng.choice([-1, 1])
            x_trial = best_x.copy()
            x_trial[idx] += delta
        elif r < 0.45:
            # Small random block
            n_perturb = rng.integers(2, 10)
            indices = rng.choice(N, size=n_perturb, replace=False)
            perturbation = np.zeros(N)
            perturbation[indices] = rng.normal(0, step_size, size=n_perturb)
            x_trial = best_x + perturbation
        elif r < 0.7:
            # Low-frequency structured perturbation
            freq = rng.integers(1, 8)
            phase = rng.uniform(0, 2 * np.pi)
            perturbation = step_size * np.sin(2 * np.pi * freq * t_arr / N + phase)
            x_trial = best_x + perturbation
        elif r < 0.85:
            # Multi-frequency structured perturbation
            freq1 = rng.integers(1, 5)
            freq2 = rng.integers(5, 15)
            phase1 = rng.uniform(0, 2 * np.pi)
            phase2 = rng.uniform(0, 2 * np.pi)
            perturbation = step_size * (0.7 * np.sin(2 * np.pi * freq1 * t_arr / N + phase1) +
                                        0.3 * np.sin(2 * np.pi * freq2 * t_arr / N + phase2))
            x_trial = best_x + perturbation
        else:
            # Random sparse large perturbation
            n_perturb = rng.integers(5, 25)
            indices = rng.choice(N, size=n_perturb, replace=False)
            perturbation = np.zeros(N)
            perturbation[indices] = rng.normal(0, step_size * 2.5, size=n_perturb)
            x_trial = best_x + perturbation
        
        val = _objective_only(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            step_size = max(0.0005, step_size * 0.96)
            no_improve = 0
        else:
            no_improve += 1
            step_size = min(0.1, step_size * 1.02)
        
        # Periodic restart from slightly perturbed best
        if no_improve > 3000 and rng.random() < 0.02:
            x_restart = best_x + rng.normal(0, 0.08, size=N)
            val_restart = _objective_only(x_restart)
            if val_restart < best_val:
                best_val = val_restart
                best_x = x_restart.copy()
                no_improve = 0
                step_size = 0.02
    
    return best_x
# EVOLVE-BLOCK-END
