"""Model-and-optimize approach for signed step functions minimizing max|autoconv|/(sum x)^2."""
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

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective_normalized(x, beta=50.0):
    """
    Objective: max(|autoconv(x/sum(x))|) using soft-max surrogate.
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(x)
    
    x_norm = x / s
    ac = _autoconv_fft(x_norm)
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    
    if max_val < 1e-15:
        return 1e10, np.zeros_like(x)
    
    shifted = beta * (abs_ac - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max = max_val + (1.0 / beta) * np.log(sum_exp)
    
    # Gradient computation
    g = np.zeros(2*N-1)
    g[:len(ac)] = weights * np.sign(ac)
    
    L_fft = 1
    while L_fft < 3*N:
        L_fft *= 2
    
    G = np.fft.rfft(g, n=L_fft)
    Xf = np.fft.rfft(x_norm, n=L_fft)
    
    term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft)[:N]
    term2 = np.fft.irfft(G * Xf, n=L_fft)[:N]
    
    d_softmax_dx_norm = term1 + term2
    
    sum_grad_norm = np.sum(d_softmax_dx_norm)
    grad_x = (1.0 / s) * (d_softmax_dx_norm - x_norm * sum_grad_norm)
    
    return soft_max, grad_x

def _spectral_init(rng):
    """
    Generate a signal x of length N such that its linear autoconvolution
    has a low peak. We do this by designing the spectrum to be as flat as possible.
    """
    L = 2 * N - 1
    # We want |fft(x, L)|^2 to be flat.
    # Let Y be a length-L vector with |Y[k]| = 1.
    # y = ifft(Y). x = y[:N].
    
    # Random phases
    Y = np.ones(L, dtype=complex)
    Y[1:] = np.exp(1j * rng.uniform(0, 2*np.pi, size=L-1))
    # Enforce symmetry for real y
    for k in range(1, L//2):
        Y[L-k] = np.conj(Y[k])
    if L % 2 == 0:
        Y[L//2] = 1.0
        
    y = np.fft.ifft(Y).real
    x = y[:N]
    
    # Normalize x to have sum 1 to avoid large values
    s = np.sum(x)
    if abs(s) > 1e-6:
        x = x / s
    else:
        x = np.ones(N) / N
        
    return x

def construct():
    best_x = np.ones(N)
    best_val = _objective_normalized(best_x, beta=50.0)[0]
    
    if _check_deadline():
        return best_x

    rng = np.random.default_rng(42)
    
    # Phase 1: Spectral Initialization
    initializations = []
    for _ in range(20):
        x_init = _spectral_init(rng)
        initializations.append(x_init)
        
    # Add some standard initializations
    t = np.linspace(0, 1, N)
    initializations.extend([
        np.ones(N),
        np.cos(np.pi * t),
        np.sin(np.pi * t),
        1.0 + 0.5 * np.cos(2 * np.pi * t),
    ])
    
    for x0 in initializations:
        if _check_deadline():
            break
        val, _ = _objective_normalized(x0, beta=50.0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 2: Optimization with increasing sharpness
    betas = [50.0, 100.0, 200.0, 500.0]
    
    for beta in betas:
        if _check_deadline():
            break
        
        def _obj_grad(x, b=beta):
            val, grad = _objective_normalized(x, beta=b)
            return val, grad

        try:
            result = minimize(_obj_grad, best_x, method='L-BFGS-B', jac=True,
                            options={'maxiter': 300, 'gtol': 1e-6, 'ftol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
            
    if _check_deadline():
        return best_x
        
    # Phase 3: Final High-Precision Polish
    def _final_obj(x):
        val, grad = _objective_normalized(x, beta=1000.0)
        return val, grad

    try:
        result = minimize(_final_obj, best_x, method='L-BFGS-B', jac=True,
                        options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-14})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    return best_x
# EVOLVE-BLOCK-END
