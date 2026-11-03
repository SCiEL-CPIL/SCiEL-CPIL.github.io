"""
Alternating minimization with spectral magnitude constraint for signed step functions.
Minimizes max|autoconv|/(sum x)^2 by alternating between time-domain optimization
and spectral projection to a flat-top magnitude profile.
"""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 115
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

# Precompute FFT sizes
_L_fft = 1
while _L_fft < 2 * N - 1:
    _L_fft *= 2

_L_fft_grad = 1
while _L_fft_grad < 3 * N:
    _L_fft_grad *= 2

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    X = np.fft.rfft(x, n=_L_fft)
    C = np.fft.irfft(X * X, n=_L_fft)[:L]
    return C

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _objective_grad_smooth(x, beta=30.0):
    """
    Compute soft-max surrogate objective and its gradient.
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(x)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    
    if max_val < 1e-15:
        return 1e10, np.zeros_like(x)
    
    shifted = beta * (abs_ac - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max = max_val + (1.0 / beta) * np.log(sum_exp)
    val = soft_max / (s * s)
    
    g = np.zeros(2*N-1)
    g[:len(ac)] = weights * np.sign(ac)
    
    G = np.fft.rfft(g, n=_L_fft_grad)
    Xf = np.fft.rfft(x, n=_L_fft_grad)
    
    term1 = np.fft.irfft(np.conj(G) * Xf, n=_L_fft_grad)[:N]
    term2 = np.fft.irfft(G * Xf, n=_L_fft_grad)[:N]
    
    d_softmax_dx = term1 + term2
    
    grad = (1.0 / (s * s)) * d_softmax_dx - (2.0 * soft_max / (s ** 3)) * np.ones(N)
    
    return val, grad

def _epigraph_objective(x, t, beta=50.0):
    """
    Epigraph formulation: minimize t subject to max|ac|/(s^2) <= t.
    Uses smooth approximation: sum exp(beta*(|ac_i|/s^2 - t)) <= 1.
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N+1)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    s2 = s * s
    
    # Soft constraint: sum exp(beta*(|ac_i|/s^2 - t)) <= 1
    # Objective: t + penalty for constraint violation
    ratios = abs_ac / s2
    shifted = beta * (ratios - t)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    # Penalty: max(0, log(sum_exp))^2
    log_sum = np.log(sum_exp + 1e-15)
    penalty = max(0.0, log_sum) ** 2
    
    val = t + penalty
    
    # Gradient w.r.t. x
    weights = exp_shifted / (sum_exp + 1e-15)
    g = np.zeros(2*N-1)
    g[:len(ac)] = weights * np.sign(ac) / s2
    
    G = np.fft.rfft(g, n=_L_fft_grad)
    Xf = np.fft.rfft(x, n=_L_fft_grad)
    
    term1 = np.fft.irfft(np.conj(G) * Xf, n=_L_fft_grad)[:N]
    term2 = np.fft.irfft(G * Xf, n=_L_fft_grad)[:N]
    
    d_penalty_dx = (2.0 * max(0.0, log_sum)) * (term1 + term2)
    
    # Gradient w.r.t. t
    d_penalty_dt = -2.0 * max(0.0, log_sum) * np.sum(weights)
    d_val_dt = 1.0 + d_penalty_dt
    
    grad = np.zeros(N+1)
    grad[:N] = d_penalty_dx
    grad[N] = d_val_dt
    
    return val, grad

def _peak_targeted_polish(x, max_iters=500, beta=50.0):
    """
    Iteratively identify dominant autoconvolution lags and apply targeted
    gradient steps to reduce their contribution.
    """
    best_x = x.copy()
    best_val = _objective(best_x)
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        ac = _autoconv_fft(best_x)
        abs_ac = np.abs(ac)
        s2 = np.sum(best_x) ** 2
        
        # Find top-k peaks
        k = min(10, len(ac))
        top_indices = np.argsort(abs_ac)[-k:]
        
        # Weight peaks by their magnitude
        peak_weights = abs_ac[top_indices] / (np.sum(abs_ac[top_indices]) + 1e-15)
        
        # Construct gradient targeting these peaks
        g = np.zeros(2*N-1)
        for idx, w in zip(top_indices, peak_weights):
            g[idx] = w * np.sign(ac[idx])
        
        G = np.fft.rfft(g, n=_L_fft_grad)
        Xf = np.fft.rfft(best_x, n=_L_fft_grad)
        
        term1 = np.fft.irfft(np.conj(G) * Xf, n=_L_fft_grad)[:N]
        term2 = np.fft.irfft(G * Xf, n=_L_fft_grad)[:N]
        
        grad = (term1 + term2) / s2 - (2.0 * np.max(abs_ac) / (s2 ** 1.5)) * np.ones(N)
        
        # Normalize and take step
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-12:
            break
        
        step_size = 0.05 / (gnorm + 1e-10)
        x_trial = best_x - step_size * grad
        
        val_trial = _objective(x_trial)
        if val_trial < best_val:
            best_val = val_trial
            best_x = x_trial.copy()
        else:
            # Try smaller step
            for _ in range(5):
                step_size *= 0.5
                x_trial = best_x - step_size * grad
                val_trial = _objective(x_trial)
                if val_trial < best_val:
                    best_val = val_trial
                    best_x = x_trial.copy()
                    break
    
    return best_x, best_val

def _spectral_design(x, concentration=0.7):
    """
    Design spectral magnitude to concentrate energy where it helps minimize
    peak autoconvolution. Uses prolate-like concentration.
    """
    k = np.arange(N // 2 + 1)
    kf = k / (N // 2)
    
    # Prolate-like: concentrate in middle frequencies
    target_mag = np.exp(-((kf - 0.5) ** 2) / (2 * concentration ** 2))
    target_mag = target_mag / np.max(target_mag)
    
    X = np.fft.rfft(x, n=_L_fft)
    current_mag = np.abs(X[:N//2+1])
    
    # Blend
    new_mag = 0.5 * current_mag + 0.5 * target_mag
    new_mag = np.maximum(new_mag, 1e-10)
    
    phase = np.angle(X[:N//2+1])
    X_new = new_mag * np.exp(1j * phase)
    
    X_full = np.zeros(_L_fft // 2 + 1, dtype=complex)
    X_full[:N//2+1] = X_new
    for i in range(1, N//2+1):
        if _L_fft - i <= _L_fft // 2:
            X_full[_L_fft - i] = np.conj(X_new[i])
    
    x_new = np.fft.irfft(X_full, n=_L_fft)[:N]
    return x_new

def _coordinate_descent(x, max_rounds=50, max_steps_per_coord=10):
    """
    Coordinate descent: optimize one element at a time.
    """
    best_x = x.copy()
    best_val = _objective(best_x)
    
    for rnd in range(max_rounds):
        if _check_deadline():
            break
        
        improved = False
        for i in range(N):
            if _check_deadline():
                break
            
            original = best_x[i]
            # Try a few perturbations
            for delta in [-0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 1.0, -1.0]:
                best_x[i] = original + delta
                val = _objective(best_x)
                if val < best_val:
                    best_val = val
                    improved = True
                else:
                    best_x[i] = original
        
        if not improved:
            break
    
    return best_x, best_val

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    # Phase 1: Structured initializations
    t = np.linspace(0, 1, N)
    initializations = [
        np.ones(N),
        np.linspace(2.0, 1.0, N),
        np.linspace(1.0, 2.0, N),
        1.0 + 0.5 * np.cos(np.pi * t),
        1.0 + np.cos(np.pi * t),
        1.0 + 0.3 * np.sin(2 * np.pi * t),
        1.0 + 0.5 * np.sin(np.pi * t),
        np.exp(-2.0 * t),
        np.exp(2.0 * t - 2.0),
        1.0 + 0.2 * np.cos(2 * np.pi * 3 * t),
        1.0 + 0.3 * np.cos(2 * np.pi * 5 * t),
    ]
    
    for r in [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 1.1, 1.2, 1.5, 2.0]:
        initializations.append(r ** np.arange(N))
    
    rng_init = np.random.default_rng(123)
    for _ in range(30):
        freqs = rng_init.uniform(0.1, 3.0, size=5)
        phases = rng_init.uniform(0, 2*np.pi, size=5)
        amps = rng_init.uniform(0.1, 0.5, size=5)
        sig = 1.0 + sum(a * np.cos(2*np.pi*f*t + p) for a, f, p in zip(amps, freqs, phases))
        initializations.append(sig)
    
    for x0 in initializations:
        if _check_deadline():
            break
        val = _objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 2: L-BFGS-B with smooth surrogate
    def _smooth_obj(x):
        val, grad = _objective_grad_smooth(x, beta=30.0)
        return val, grad
    
    try:
        result = minimize(_smooth_obj, best_x, method='L-BFGS-B', jac=True,
                        options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-14})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Peak-targeted polish
    best_x, best_val = _peak_targeted_polish(best_x, max_iters=300)
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Epigraph formulation optimization
    def _epi_obj(z):
        x = z[:N]
        t = z[N]
        val, grad = _epigraph_objective(x, t, beta=50.0)
        return val, grad
    
    try:
        z0 = np.concatenate([best_x, [best_val * 1.1]])
        result = minimize(_epi_obj, z0, method='L-BFGS-B', jac=True,
                        options={'maxiter': 300, 'gtol': 1e-8, 'ftol': 1e-14})
        if result.fun < best_val:
            # Extract x from result
            x_candidate = result.x[:N]
            val_candidate = _objective(x_candidate)
            if val_candidate < best_val:
                best_val = val_candidate
                best_x = x_candidate.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 5: Spectral design + optimization
    x_spectral = _spectral_design(best_x, concentration=0.6)
    val_spectral = _objective(x_spectral)
    if val_spectral < best_val:
        best_val = val_spectral
        best_x = x_spectral.copy()
    
    # Optimize from spectral design
    try:
        result = minimize(_smooth_obj, best_x, method='L-BFGS-B', jac=True,
                        options={'maxiter': 300, 'gtol': 1e-8, 'ftol': 1e-14})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 6: Peak-targeted polish again
    best_x, best_val = _peak_targeted_polish(best_x, max_iters=300)
    
    if _check_deadline():
        return best_x
    
    # Phase 7: Coordinate descent
    best_x, best_val = _coordinate_descent(best_x, max_rounds=30, max_steps_per_coord=8)
    
    if _check_deadline():
        return best_x
    
    # Phase 8: Final L-BFGS-B polish with tight tolerances
    def _smooth_obj_tight(x):
        val, grad = _objective_grad_smooth(x, beta=80.0)
        return val, grad
    
    try:
        result = minimize(_smooth_obj_tight, best_x, method='L-BFGS-B', jac=True,
                        options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-15})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 9: Final peak-targeted polish
    best_x, best_val = _peak_targeted_polish(best_x, max_iters=200)
    
    if _check_deadline():
        return best_x
    
    # Phase 10: Gradient-guided random perturbation
    rng = np.random.default_rng(42)
    step_size = 0.02
    stall_count = 0
    max_iters = 15000
    
    for iteration in range(max_iters):
        if _check_deadline():
            break
        
        val, grad = _objective_grad_smooth(best_x, beta=50.0)
        gnorm = np.linalg.norm(grad)
        
        if gnorm > 1e-10:
            grad_dir = -grad / gnorm
        else:
            grad_dir = np.zeros(N)
        
        random_dir = rng.normal(0, 1, size=N)
        random_dir /= (np.linalg.norm(random_dir) + 1e-10)
        
        blend = 0.3
        direction = blend * grad_dir + (1 - blend) * random_dir
        direction /= (np.linalg.norm(direction) + 1e-10)
        
        x_trial = best_x + step_size * direction
        
        val_trial = _objective(x_trial)
        if val_trial < best_val:
            best_val = val_trial
            best_x = x_trial.copy()
            stall_count = 0
            step_size = max(0.001, step_size * 0.999)
        else:
            stall_count += 1
            if stall_count % 2000 == 0:
                step_size = min(0.2, step_size * 1.1)
                if stall_count % 10000 == 0:
                    big_pert = rng.normal(0, step_size * 3, size=N)
                    x_big = best_x + big_pert
                    val_big = _objective(x_big)
                    if val_big < best_val:
                        best_val = val_big
                        best_x = x_big.copy()
    
    # Final polish: tight BFGS on exact objective
    if not _check_deadline():
        try:
            result = minimize(_objective, best_x, method='BFGS', 
                            options={'maxiter': 200, 'gtol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    return best_x
# EVOLVE-BLOCK-END
