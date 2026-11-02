"""Model-and-optimize: spectral phase retrieval + L-BFGS with epigraph penalty + coordinate descent."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

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

def _epigraph_penalty_and_grad(x, t, alpha):
    """
    Compute the epigraph penalty objective and its gradient.
    L(x, t) = t + alpha * sum_k max(0, |C_k| - t)^2
    """
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    # Hinge loss
    hinge = np.maximum(0, abs_ac - t)
    penalty = np.sum(hinge ** 2)
    val = t + alpha * penalty
    
    # Gradient w.r.t t
    grad_t = 1 - 2 * alpha * np.sum(hinge)
    
    # Gradient w.r.t x
    # dL/dC_k = 2 * alpha * hinge_k * sign(C_k)
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    g_k = 2 * alpha * hinge * signs
    
    # grad_x[n] = sum_k g_k * 2 * x[k-n]
    # This is 2 * cross_correlation(g_k, x)[n]
    # g_k has length 2N-1. x has length N.
    # We need R[n] = sum_{k} g_k x_{k-n} where 0 <= k-n < N.
    
    # Use FFT for cross-correlation
    L_corr = 1
    while L_corr < 3 * N: # Safe padding
        L_corr *= 2
    
    g_padded = np.zeros(L_corr)
    g_padded[:2*N-1] = g_k
    
    X_f = np.fft.rfft(x, n=L_corr)
    G_f = np.fft.rfft(g_padded, n=L_corr)
    
    # Cross-correlation: IFFT(G * conj(X))
    corr = np.fft.irfft(G_f * np.conj(X_f), n=L_corr)
    grad_x = 2.0 * corr[:N]
    
    return val, (grad_x, grad_t)

def _epigraph_optimize(x0, rng, alphas=[10, 100, 1000, 10000]):
    """
    Optimize x and t using L-BFGS-B on the epigraph penalty.
    """
    best_x = x0.copy()
    best_val = _objective_only(x0)
    
    # Initial t
    ac = _autoconv_fft(best_x)
    t0 = np.max(np.abs(ac))
    
    z0 = np.concatenate([best_x, [t0]])
    
    for alpha in alphas:
        if _check_deadline():
            break
            
        def obj_and_grad(z, _alpha=alpha):
            x = z[:N]
            t = z[N]
            val, (gx, gt) = _epigraph_penalty_and_grad(x, t, _alpha)
            g = np.concatenate([gx, [gt]])
            return val, g
        
        try:
            result = minimize(obj_and_grad, z0, method='L-BFGS-B', jac=True,
                              options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-12, 'maxls': 30})
            
            if result.fun < 1e9: # Sanity check
                x_opt = result.x[:N]
                t_opt = result.x[N]
                
                # Evaluate true objective
                true_val = _objective_only(x_opt)
                
                if true_val < best_val:
                    best_val = true_val
                    best_x = x_opt.copy()
                    z0 = np.concatenate([best_x, [t_opt]])
                    
        except Exception:
            pass
            
    return best_x, best_val

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

def _polish(x0, rng, max_iters=2000):
    """
    Coordinate descent polish.
    """
    best_x = x0.copy()
    best_val = _objective_only(x0)
    step_size = 0.05
    no_improve = 0
    
    t_arr = np.arange(N)
    
    for it in range(max_iters):
        if _check_deadline():
            break
            
        r = rng.random()
        if r < 0.2:
            # Single coordinate
            idx = rng.integers(0, N)
            delta = step_size * rng.choice([-1, 1])
            x_trial = best_x.copy()
            x_trial[idx] += delta
        elif r < 0.5:
            # Small random block
            n_perturb = rng.integers(2, 10)
            indices = rng.choice(N, size=n_perturb, replace=False)
            perturbation = np.zeros(N)
            perturbation[indices] = rng.normal(0, step_size, size=n_perturb)
            x_trial = best_x + perturbation
        else:
            # Structured low freq
            freq = rng.integers(1, 8)
            phase = rng.uniform(0, 2 * np.pi)
            perturbation = step_size * np.sin(2 * np.pi * freq * t_arr / N + phase)
            x_trial = best_x + perturbation
            
        val = _objective_only(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            step_size = max(0.001, step_size * 0.9)
            no_improve = 0
        else:
            no_improve += 1
            step_size = min(0.2, step_size * 1.05)
            
            if no_improve > 500:
                # Restart
                x_restart = best_x + rng.normal(0, 0.1, size=N)
                val_restart = _objective_only(x_restart)
                if val_restart < best_val:
                    best_val = val_restart
                    best_x = x_restart.copy()
                    no_improve = 0
                    step_size = 0.1
                else:
                    no_improve = 0 # Reset counter to try again with different noise
                    step_size = 0.05

    return best_x, best_val

def construct():
    best_x = np.ones(N)
    best_val = _objective_only(best_x)
    
    if _check_deadline():
        return best_x
        
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initial_points = []
    for mode in range(4):
        for _ in range(5):
            initial_points.append(_spectral_init(rng, mode))
            
    # Add some random Gaussian inits
    for _ in range(10):
        initial_points.append(rng.normal(1.0, 0.5, size=N))
        
    # Add sine waves
    t = np.linspace(0, 1, N)
    for f in [1, 2, 3, 5, 7, 10, 15]:
        initial_points.append(1.0 + 0.5 * np.sin(2 * np.pi * f * t))
        initial_points.append(1.0 + 0.5 * np.cos(2 * np.pi * f * t))

    alphas = [50, 500, 5000, 50000]
    
    for x0 in initial_points:
        if _check_deadline():
            break
            
        x_start = x0 + rng.normal(0, 0.01, size=N)
        
        # Run epigraph optimization
        x_opt, val_opt = _epigraph_optimize(x_start, rng, alphas=alphas)
        
        if val_opt < best_val:
            best_val = val_opt
            best_x = x_opt.copy()
            
    if _check_deadline():
        return best_x
        
    # Polish the best result
    x_polished, val_polished = _polish(best_x, rng, max_iters=3000)
    
    if val_polished < best_val:
        best_val = val_polished
        best_x = x_polished.copy()
        
    return best_x
# EVOLVE-BLOCK-END
