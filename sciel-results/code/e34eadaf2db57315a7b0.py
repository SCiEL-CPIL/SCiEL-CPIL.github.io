"""Combined program: smooth surrogate + gradient descent for signed step functions."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 115  # safety margin before 120s kill
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT for speed."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _smooth_objective_and_grad(x, beta=50.0):
    """Smooth surrogate using log-sum-exp for max(|ac|) / (sum x)^2, with gradient."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    # Log-sum-exp for smooth max
    max_val = np.max(abs_ac)
    if max_val < 1e-15:
        smooth_max = max_val
        weights = np.ones_like(abs_ac) / len(abs_ac)
    else:
        shifted = abs_ac - max_val
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        smooth_max = (max_val + np.log(sum_exp)) / beta
        weights = exp_shifted / sum_exp
    
    val = smooth_max / (s * s)
    
    # Gradient of smooth_max w.r.t. x:
    c = weights * np.sign(ac)
    
    # Term A: (c * x)_j = sum_m c[j+m] * x[m] for m in [0,N-1] where 0<=j+m<2N-1
    term_A = np.zeros(N)
    for j in range(N):
        m_lo = 0
        m_hi = min(N - 1, 2 * N - 2 - j)
        if m_lo <= m_hi:
            ms = np.arange(m_lo, m_hi + 1)
            term_A[j] = np.sum(x[ms] * c[j + ms])
    
    # Term B: x[j] * sum_{k=j}^{j+N-1} c[k]
    c_padded = np.zeros(2 * N - 1)
    c_padded[:len(c)] = c
    c_cumsum = np.cumsum(np.concatenate([[0], c_padded]))
    window_sums = np.zeros(N)
    for j in range(N):
        k_lo = j
        k_hi = min(j + N, 2 * N - 1)
        window_sums[j] = c_cumsum[k_hi] - c_cumsum[k_lo]
    term_B = x * window_sums
    
    grad_smooth_max = term_A + term_B
    
    # Gradient of val = smooth_max / s^2
    grad = grad_smooth_max / (s * s) - 2.0 * smooth_max / (s * s * s)
    
    return val, grad

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    # Phase 1: Try structured initializations
    t = np.linspace(0, 1, N)
    initializations = [
        np.ones(N),
        np.linspace(2.0, 1.0, N),
        np.linspace(1.0, 2.0, N),
        1.0 + 0.5 * np.cos(np.pi * t),
        1.0 + np.cos(np.pi * t),
        1.0 + t**2,
        1.0 + 4.0 * (t - 0.5)**2,
        np.exp(2.0 * t),
        np.exp(-2.0 * t),
        1.0 + 0.3 * np.sin(2 * np.pi * t),
        1.0 + 0.5 * np.sin(np.pi * t),
        np.ones(N) * np.sign(np.sin(np.pi * t)),
        np.ones(N) * np.sign(np.cos(np.pi * t)),
        1.0 + 0.5 * np.sin(2 * np.pi * t),
        1.0 - 0.5 * np.sin(2 * np.pi * t),
    ]
    
    for r in [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5, 2.0]:
        initializations.append(r ** np.arange(N))
    
    for x0 in initializations:
        if _check_deadline():
            break
        val = _objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 2: Gradient-based optimization with multiple beta values
    def _smooth_obj_and_grad(x, beta=50.0):
        val, grad = _smooth_objective_and_grad(x, beta)
        return val, grad
    
    for beta in [50.0, 30.0, 20.0]:
        if _check_deadline():
            break
        try:
            result = minimize(lambda x: _smooth_objective_and_grad(x, beta)[0], 
                            best_x, method='L-BFGS-B', 
                            jac=lambda x: _smooth_objective_and_grad(x, beta)[1],
                            options={'maxiter': 150, 'gtol': 1e-5, 'ftol': 1e-8})
            if result.fun < best_val:
                exact_val = _objective(result.x)
                if exact_val < best_val:
                    best_val = exact_val
                    best_x = result.x.copy()
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 3: BFGS with exact objective for polish
    try:
        result = minimize(_objective, best_x, method='BFGS', options={'maxiter': 100, 'gtol': 1e-6})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Random perturbation refinement with adaptive step
    rng = np.random.default_rng(42)
    step_size = 0.05
    stall_count = 0
    
    for iteration in range(20000):
        if _check_deadline():
            break
        
        perturbation = rng.normal(0, step_size, size=N)
        x_trial = best_x + perturbation
        
        val = _objective(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            stall_count = 0
            step_size = max(0.001, step_size * 0.999)
        else:
            stall_count += 1
            if stall_count % 2000 == 0:
                step_size = min(0.5, step_size * 1.03)
    
    return best_x
# EVOLVE-BLOCK-END
