"""Model-and-optimize: spectral amplitude shaping + multi-start L-BFGS + fast random polish."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 140  # safety margin before 150s kill

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - 5)

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
    """Compute the raw objective: max|ac| / (sum(x))^2."""
    ac = _autoconv_fft(x)
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    return np.max(np.abs(ac)) / (s * s)

def _soft_max_and_grad(x, beta):
    """
    Compute soft-max surrogate for max|ac| and its gradient.
    Returns (soft_max_value, gradient_array)
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
    grad_ac = weights * signs
    
    g = grad_ac  # length 2N-1
    cumsum_g = np.concatenate([[0], np.cumsum(g)])
    idx_end = np.minimum(np.arange(N) + N - 1, 2*N-2) + 1
    idx_start = np.arange(N)
    window_sums = cumsum_g[idx_end] - cumsum_g[idx_start]
    term2 = x * window_sums
    
    L_fft = 1
    while L_fft < 4*N:
        L_fft *= 2
    X_f = np.fft.rfft(x, n=L_fft)
    G_f = np.fft.rfft(g, n=L_fft)
    corr = np.fft.irfft(X_f * np.conj(G_f), n=L_fft)
    term1 = corr[:N]
    
    grad_x = term1 + term2
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

def _spectral_init(rng):
    """
    Generate initial points via spectral amplitude shaping.
    Fix the amplitude spectrum to a window shape, randomize phase,
    then take the real part and clip to positive values.
    """
    L_fft = 1
    while L_fft < 2*N:
        L_fft *= 2
    
    # Various window shapes for the amplitude spectrum
    k = np.arange(L_fft // 2 + 1)
    t = k / (L_fft // 2)
    
    windows = [
        np.ones(L_fft // 2 + 1),  # flat
        np.exp(-0.5 * (t - 0.5)**2 / (0.15**2)),  # Gaussian
        np.exp(-0.5 * (t - 0.5)**2 / (0.25**2)),  # wider Gaussian
        1.0 / (1.0 + (t / 0.3)**2),  # Lorentzian
        np.sinc(0.5 * t) ** 2,  # sinc-squared
    ]
    
    inits = []
    for w in windows:
        # Ensure w is real and non-negative
        w = np.maximum(w, 0)
        # Random phase
        phase = rng.uniform(0, 2*np.pi, size=L_fft // 2 + 1)
        # Build complex spectrum
        spec = w * np.exp(1j * phase)
        # Make it Hermitian symmetric
        spec_full = np.zeros(L_fft, dtype=complex)
        spec_full[:L_fft//2+1] = spec
        spec_full[L_fft//2+1:] = np.conj(spec[1:L_fft//2][::-1])
        # Inverse FFT
        x = np.fft.ifft(spec_full).real[:N]
        # Shift to be positive
        x = x - np.min(x) + 0.1
        inits.append(x)
    
    # Also add some structured inits
    t_arr = np.linspace(0, 1, N)
    inits.append(np.ones(N))
    inits.append(1.0 + 0.5 * np.sin(2 * np.pi * t_arr))
    inits.append(1.0 + 0.5 * np.cos(2 * np.pi * t_arr))
    inits.append(np.exp(t_arr))
    inits.append(np.exp(-t_arr))
    inits.append(1.0 + 0.8 * (t_arr - 0.5)**2)
    inits.append(1.0 + 0.5 * (t_arr - 0.5))
    
    # Random inits
    for _ in range(10):
        inits.append(rng.normal(1.0, 0.3, size=N))
    
    # Sparse-like inits
    for _ in range(5):
        x = np.ones(N) * 0.1
        n_spikes = rng.integers(3, 10)
        idx = rng.choice(N, size=n_spikes, replace=False)
        x[idx] = 1.0
        inits.append(x)
    
    # Multi-frequency inits
    for _ in range(5):
        x = np.ones(N)
        for freq in rng.integers(1, 5, size=3):
            x += 0.3 * np.sin(2 * np.pi * freq * t_arr + rng.uniform(0, 2*np.pi))
        inits.append(x)
    
    return inits

def construct():
    best_x = np.ones(N)
    best_val = _objective_only(best_x)
    
    if _check_deadline():
        return best_x
    
    rng = np.random.default_rng(12345)
    
    # Phase 1: Multi-start L-BFGS with spectral inits
    betas = [30.0, 50.0, 100.0]
    initial_points = _spectral_init(rng)
    
    for beta in betas:
        if _check_deadline():
            break
        for x0 in initial_points:
            if _check_deadline():
                break
            
            x_start = x0 + rng.normal(0, 0.01, size=N)
            
            try:
                def obj_and_grad_func(x):
                    return _objective_and_grad(x, beta)
                
                result = minimize(obj_and_grad_func, x_start, method='L-BFGS-B',
                                  jac=True, options={'maxiter': 150, 'gtol': 1e-8, 'ftol': 1e-10})
                
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x.copy()
                    
            except Exception:
                pass
    
    if _check_deadline():
        return best_x
    
    # Phase 2: Polish with higher beta
    for beta in [200.0, 500.0]:
        if _check_deadline():
            break
        try:
            def obj_and_grad_func(x):
                return _objective_and_grad(x, beta)
            
            result = minimize(obj_and_grad_func, best_x, method='L-BFGS-B',
                              jac=True, options={'maxiter': 300, 'gtol': 1e-10, 'ftol': 1e-12})
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
                
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Fast random perturbation polish using objective-only (no gradient)
    step_size = 0.01
    n_steps = 0
    max_steps = 500000
    
    while n_steps < max_steps:
        if _check_deadline():
            break
        perturbation = rng.normal(0, step_size, size=N)
        x_trial = best_x + perturbation
        val = _objective_only(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            step_size *= 0.999
        else:
            step_size = min(0.1, step_size * 1.01)
        n_steps += 1
    
    # Phase 4: Final coordinate descent polish
    if not _check_deadline():
        coord_steps = 0
        max_coord_steps = 200000
        while coord_steps < max_coord_steps:
            if _check_deadline():
                break
            improved = False
            for i in range(0, N, max(1, N // 200)):
                if _check_deadline():
                    break
                for delta in [0.01, -0.01, 0.005, -0.005, 0.02, -0.02]:
                    x_trial = best_x.copy()
                    x_trial[i] += delta
                    val = _objective_only(x_trial)
                    if val < best_val:
                        best_val = val
                        best_x = x_trial
                        improved = True
                        break
            coord_steps += 1
            if not improved:
                break
    
    return best_x
# EVOLVE-BLOCK-END
