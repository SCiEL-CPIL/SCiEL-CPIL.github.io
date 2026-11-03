"""Spectral amplitude optimization + multi-start L-BFGS + adaptive polish with prolate basis refinement."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import eval_legendre

START_TIME = time.time()
TIME_LIMIT = 148

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
    Uses faster 3N FFT size for correlation.
    """
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    m = np.max(abs_ac)
    # Numerical stability: subtract max before exp
    exps = np.exp(beta * (abs_ac - m))
    sum_exps = np.sum(exps)
    if sum_exps < 1e-300:
        soft_val = m
        weights = np.zeros_like(abs_ac)
        weights[np.argmax(abs_ac)] = 1.0
    else:
        soft_val = m + np.log(sum_exps) / beta
        weights = exps / sum_exps
    
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    grad_ac = weights * signs  # length 2N-1
    
    # Cross-correlation: corr[n] = sum_k grad_ac[k] * x[k-n] for 0<=k-n<N
    L_corr = 1
    while L_corr < 3 * N:
        L_corr *= 2
    
    g_padded = np.zeros(L_corr)
    g_padded[:2*N-1] = grad_ac
    
    X_f = np.fft.rfft(x, n=L_corr)
    G_f = np.fft.rfft(g_padded, n=L_corr)
    
    corr = np.fft.irfft(G_f * np.conj(X_f), n=L_corr)
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
    
    freqs = np.fft.rfftfreq(L_fft)
    n_freq = L_fft // 2 + 1
    
    if mode == 0:
        mag = np.ones(n_freq)
        taper = np.minimum(1.0, np.abs(freqs) * L_fft / 5)
        mag *= taper
    elif mode == 1:
        mag = 0.5 + 0.5 * np.cos(2 * np.pi * freqs * L_fft * 0.5)
    elif mode == 2:
        mag = 1.0 + 0.3 * rng.normal(0, 1, size=n_freq)
        mag = np.maximum(0.1, mag)
    elif mode == 3:
        mag = np.exp(-20 * (freqs * L_fft / 2) ** 2)
    elif mode == 4:
        cutoff = 0.4
        transition = 0.05
        mag = np.where(np.abs(freqs) < cutoff - transition, 1.0,
                       np.where(np.abs(freqs) < cutoff + transition,
                                0.5 * (1 + np.cos(np.pi * (np.abs(freqs) - cutoff) / transition)),
                                0.1))
    else:
        mag = np.ones(n_freq)
    
    phase = rng.uniform(0, 2 * np.pi, size=n_freq)
    phase[0] = 0
    
    spectrum = mag * np.exp(1j * phase)
    spectrum[0] = mag[0]
    
    x_full = np.fft.irfft(spectrum, n=L_fft)
    x = x_full[:N]
    
    s = np.sum(x)
    if s < 0:
        x = -x
    
    return x

def _spectral_phase_optimize(x0, rng, max_iters=150, lr=0.01):
    """
    Optimize the phase of the spectrum while keeping magnitude fixed.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    n_freq = L_fft // 2 + 1
    
    X = np.fft.rfft(x0, n=L_fft)
    mag = np.abs(X).copy()
    phase = np.angle(X).copy()
    phase[0] = 0
    
    best_x = x0.copy()
    best_val = _objective_only(x0)
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        val, grad = _objective_and_grad(best_x, 50.0)
        G = np.fft.rfft(grad, n=L_fft)
        
        X_curr = mag * np.exp(1j * phase)
        phase_grad = np.imag(X_curr * np.conj(G)) / (mag + 1e-10)
        phase_grad[0] = 0
        
        phase -= lr * phase_grad
        
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
            if it % 20 == 0 and it > 0:
                lr = max(0.001, lr * 0.95)
        else:
            phase += lr * phase_grad
            lr = max(0.001, lr * 0.8)
    
    return best_x, best_val

def _amplitude_optimize(x0, rng, iters=40, beta=50.0):
    """
    Optimize the amplitude spectrum while keeping phase fixed.
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
    Identify dominant autoconvolution lags and apply targeted perturbations.
    """
    best_x = x0.copy()
    best_val = _objective_only(x0)
    ac = _autoconv_fft(best_x)
    abs_ac = np.abs(ac)
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        if it % 15 == 0:
            ac = _autoconv_fft(best_x)
            abs_ac = np.abs(ac)
        
        top_k = 10
        sorted_idx = np.argsort(-abs_ac)
        top_lags = sorted_idx[:top_k]
        
        # Targeted perturbation: reduce the peaks by adjusting x at positions
        # that contribute most to those lags
        for lag in top_lags:
            if _check_deadline():
                break
            
            # For lag k, ac[k] = sum_i x[i]*x[i+k]
            # To reduce |ac[k]|, perturb x in opposite direction to x[i+k]
            k = int(lag)
            if k == 0:
                # Self-correlation: reduce by making x more uniform
                perturb = -step_size * (best_x - np.mean(best_x))
            else:
                # Shifted correlation
                perturb = np.zeros(N)
                # x[i] contributes with x[i+k] for i < N-k
                # and x[i-k] for i >= k
                for i in range(N - k):
                    perturb[i] -= step_size * best_x[i + k] / (N - k)
                for i in range(k, N):
                    perturb[i] -= step_size * best_x[i - k] / k
            
            x_trial = best_x + perturb
            val = _objective_only(x_trial)
            if val < best_val:
                best_val = val
                best_x = x_trial.copy()
                step_size = max(0.001, step_size * 0.95)
            else:
                step_size = min(0.05, step_size * 1.05)
    
    return best_x, best_val

def _prolate_basis_optimize(x0, rng, num_coeffs=30, max_iters=200):
    """
    Refine x by optimizing coefficients in a prolate spheroidal wave basis.
    Prolate functions are maximally bandlimited and have smooth structure.
    We express x = x0 + sum_j c_j * p_j(t) where p_j are prolate-like basis
    functions, and optimize c_j to reduce the peak autoconvolution.
    """
    if _check_deadline():
        return x0, _objective_only(x0)
    
    t = np.linspace(0, 1, N)
    
    # Build a basis of smooth, bandlimited-like functions
    # Use a mix of low-frequency sinusoids and prolate-inspired shapes
    basis = []
    
    # Low-frequency sinusoids
    for k in range(1, min(num_coeffs // 2 + 1, 15)):
        basis.append(np.sin(2 * np.pi * k * t))
        basis.append(np.cos(2 * np.pi * k * t))
    
    # Prolate-inspired: use Legendre polynomials as smooth basis
    for k in range(min(num_coeffs // 3, 10)):
        # Map t to [-1, 1] for Legendre
        t_leg = 2 * t - 1
        p_k = np.array([eval_legendre(k, xi) for xi in t_leg])
        basis.append(p_k)
    
    # Ensure we have enough basis functions
    while len(basis) < num_coeffs:
        k = len(basis) + 1
        basis.append(np.sin(2 * np.pi * k * t))
    
    basis = basis[:num_coeffs]
    B = np.column_stack(basis)  # N x num_coeffs
    
    best_c = np.zeros(num_coeffs)
    best_x = x0.copy()
    best_val = _objective_only(x0)
    
    # Use L-BFGS to optimize coefficients
    def obj_c(c, _beta=100.0):
        x_trial = x0 + B @ c
        s = np.sum(x_trial)
        if abs(s) < 1e-10:
            return 1e10, np.zeros(num_coeffs)
        val, grad_x = _objective_and_grad(x_trial, _beta)
        grad_c = B.T @ grad_x
        return val, grad_c
    
    try:
        result = minimize(obj_c, best_c, method='L-BFGS-B',
                          jac=True, options={'maxiter': max_iters, 'gtol': 1e-10, 'ftol': 1e-12})
        if result.fun < best_val:
            best_c = result.x
            best_x = x0 + B @ best_c
            best_val = result.fun
    except Exception:
        pass
    
    if _check_deadline():
        return best_x, best_val
    
    # Try with different beta values
    for beta in [50.0, 200.0]:
        if _check_deadline():
            break
        try:
            def obj_c2(c, _beta=beta):
                x_trial = x0 + B @ c
                s = np.sum(x_trial)
                if abs(s) < 1e-10:
                    return 1e10, np.zeros(num_coeffs)
                val, grad_x = _objective_and_grad(x_trial, _beta)
                grad_c = B.T @ grad_x
                return val, grad_c
            
            result = minimize(obj_c2, best_c, method='L-BFGS-B',
                              jac=True, options={'maxiter': 100, 'gtol': 1e-10, 'ftol': 1e-12})
            if result.fun < best_val:
                best_c = result.x
                best_x = x0 + B @ best_c
                best_val = result.fun
        except Exception:
            pass
    
    return best_x, best_val

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
        1.0 + 0.4 * np.sin(2 * np.pi * 5 * t) * np.cos(2 * np.pi * t),
        1.0 + 0.3 * np.cos(2 * np.pi * 7 * t),
        np.maximum(0.2, 1.0 - 4.0 * np.abs(t - 0.5)),
        1.0 + 0.6 * np.sin(2 * np.pi * 2 * t) * np.sin(2 * np.pi * 3 * t),
        # Sparse and high-frequency patterns
        np.where(rng.random(N) < 0.3, 1.0, 0.2),
        1.0 + 0.7 * np.sin(12 * np.pi * t),
        1.0 + 0.5 * np.cos(14 * np.pi * t),
        np.where(t < 0.3, 1.5, np.where(t > 0.7, 1.5, 0.5)),
        1.0 + 0.8 * np.sin(2 * np.pi * 6 * t) * np.cos(2 * np.pi * 3 * t),
    ]
    
    # Add spectral initializations
    for mode in range(5):
        for _ in range(3):
            initial_points.append(_spectral_init(rng, mode))
    
    # Add random initializations
    for _ in range(8):
        scale = rng.uniform(0.5, 1.5)
        initial_points.append(scale * rng.normal(1.0, 0.2, size=N))
    
    # L-BFGS optimization from each initial point with adaptive beta
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
                                  jac=True, options={'maxiter': 120, 'gtol': 1e-8, 'ftol': 1e-10})
                
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
                              jac=True, options={'maxiter': 180, 'gtol': 1e-10, 'ftol': 1e-12})
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
                
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Prolate basis refinement
    x_prolate, val_prolate = _prolate_basis_optimize(best_x, rng, num_coeffs=30, max_iters=200)
    if val_prolate < best_val:
        best_val = val_prolate
        best_x = x_prolate.copy()
    
    if _check_deadline():
        return best_x
    
    # Spectral phase optimization
    for _ in range(3):
        if _check_deadline():
            break
        x_phase, val_phase = _spectral_phase_optimize(best_x, rng, max_iters=100)
        if val_phase < best_val:
            best_val = val_phase
            best_x = x_phase.copy()
    
    if _check_deadline():
        return best_x
    
    # Amplitude optimization
    for _ in range(4):
        if _check_deadline():
            break
        x_amp, val_amp = _amplitude_optimize(best_x, rng, iters=40, beta=50.0)
        if val_amp < best_val:
            best_val = val_amp
            best_x = x_amp.copy()
    
    if _check_deadline():
        return best_x
    
    # Peak-targeted polish
    x_peak, val_peak = _peak_targeted_polish(best_x, rng, max_iters=600)
    if val_peak < best_val:
        best_val = val_peak
        best_x = x_peak.copy()
    
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
                              jac=True, options={'maxiter': 200, 'gtol': 1e-12, 'ftol': 1e-14})
            
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
    max_no_improve = 3000
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
