"""
Spectral-design + L-BFGS-B minimax for signed step functions 
minimizing max|autoconv|/(sum x)^2.
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

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
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

def _soft_max_grad(x, beta=50.0):
    """
    Compute the smooth surrogate objective and its gradient.
    Objective: (1/beta) * log(sum(exp(beta * |ac|))) / (sum x)^2
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(x)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    # Soft max calculation
    # Shift by max for numerical stability
    m = np.max(abs_ac)
    if m < 1e-15:
        # If autoconv is nearly zero, gradient is zero
        return 0.0, np.zeros_like(x)
    
    exp_terms = np.exp(beta * (abs_ac - m))
    sum_exp = np.sum(exp_terms)
    # Soft max value
    soft_max_val = m + (1.0 / beta) * np.log(sum_exp)
    
    # Objective value
    val = soft_max_val / (s * s)
    
    # Gradient calculation
    # d(soft_max)/dx_j = sum_k w_k * sign(ac_k) * d(ac_k)/dx_j
    # where w_k = exp(beta * (abs_ac_k - m)) / sum_exp
    w = exp_terms / sum_exp
    
    # d(ac_k)/dx_j = x_{j+k} + x_{j-k} (with bounds)
    # We need to compute G_j = sum_k w_k * sign(ac_k) * (x_{j+k} + x_{j-k})
    
    # Let v_k = w_k * sign(ac_k)
    v = w * np.sign(ac)
    
    # Compute sum_k v_k * x_{j+k} and sum_k v_k * x_{j-k}
    # This is a cross-correlation problem.
    # Let's use FFT.
    L = 2 * N - 1
    L_fft = 1
    while L_fft < N + L:
        L_fft *= 2
    
    V = np.fft.rfft(v, n=L_fft)
    X = np.fft.rfft(x, n=L_fft)
    
    # Term 1: sum_k v_k * x_{j+k}
    # This is the cross-correlation of v and x, but shifted.
    # corr[j] = sum_k v[k] * x[j+k]
    # In FFT: IFFT(conj(FFT(v)) * FFT(x)) gives sum_k v[k] * x[k+j] (circular)
    # Since we padded, it should be linear for the valid range.
    term1 = np.fft.irfft(np.conj(V) * X, n=L_fft)[:N]
    
    # Term 2: sum_k v_k * x_{j-k}
    # This is the convolution of v and x.
    # conv[j] = sum_k v[k] * x[j-k]
    # In FFT: IFFT(FFT(v) * FFT(x))
    term2 = np.fft.irfft(V * X, n=L_fft)[:N]
    
    d_softmax_dx = term1 + term2
    
    # Full gradient:
    # f(x) = S(x) / s^2
    # df/dx_j = (1/s^2) * dS/dx_j - (2*S/s^3) * 1
    # Note: S(x) here is the soft_max_val
    
    grad = d_softmax_dx / (s * s) - (2.0 * soft_max_val / (s ** 3)) * np.ones(N)
    
    return val, grad

def _spectral_design(seed=0, beta=50.0):
    """
    Design x via PSD optimization: minimize soft-max(|autoconv|) / (sum x)^2
    by optimizing the log-power-spectrum.
    """
    half = N // 2
    rng = np.random.RandomState(seed)
    
    # Start with flat PSD (white noise)
    params = np.zeros(half + 1)
    
    best_params = params.copy()
    best_val = 1e10
    
    # Simple gradient descent on log-PSD
    lr = 0.01
    for it in range(200):
        if _check_deadline():
            break
        
        P_half = np.exp(params)
        P_full = np.zeros(N)
        P_full[:half+1] = P_half
        P_full[half+1:] = P_half[:half][::-1]
        
        # c = IFFT(P_full) gives autoconvolution (circular)
        c = np.fft.irfft(P_full, n=N)
        
        # Use circular autoconv as proxy (close to linear for smooth PSDs)
        abs_c = np.abs(c)
        s = np.sum(c)  # sum of autoconv = (sum x)^2 for circular
        
        if abs(s) < 1e-10:
            continue
        
        # Soft-max surrogate for max|c|
        m = np.max(abs_c)
        exp_c = np.exp((abs_c - m) / beta)
        softmax_val = m + np.log(np.sum(exp_c)) / beta
        
        obj = softmax_val / (s * s)
        
        if obj < best_val:
            best_val = obj
            best_params = params.copy()
        
        # Gradient w.r.t. params: d(obj)/d(params[k]) = d(obj)/d(P_half[k]) * P_half[k]
        # d(softmax_val)/d(P_half[k]) = (1/beta) * exp_c[k] / sum(exp_c) * (1/s^2) * s^2 + ... 
        # This gets complex. Use finite differences or a simpler approach.
        
        # Simpler: use Nelder-Mead or just random perturbations
        # Let's try coordinate ascent on params
        improved = False
        for k in range(half + 1):
            for delta in [0.1, -0.1, 0.05, -0.05]:
                trial = params.copy()
                trial[k] += delta
                P_half_t = np.exp(trial)
                P_full_t = np.zeros(N)
                P_full_t[:half+1] = P_half_t
                P_full_t[half+1:] = P_half_t[:half][::-1]
                c_t = np.fft.irfft(P_full_t, n=N)
                abs_c_t = np.abs(c_t)
                s_t = np.sum(c_t)
                if abs(s_t) < 1e-10:
                    continue
                m_t = np.max(abs_c_t)
                exp_c_t = np.exp((abs_c_t - m_t) / beta)
                sm_t = m_t + np.log(np.sum(exp_c_t)) / beta
                obj_t = sm_t / (s_t * s_t)
                if obj_t < obj:
                    params = trial
                    obj = obj_t
                    improved = True
                    break
        if not improved:
            break
    
    # Recover x from best PSD: x = IFFT(sqrt(P_full))
    P_half_best = np.exp(best_params)
    P_full_best = np.zeros(N)
    P_full_best[:half+1] = P_half_best
    P_full_best[half+1:] = P_half_best[:half][::-1]
    
    # x such that |FFT(x)|^2 = P_full, so x = IFFT(sqrt(P_full) * phase)
    # For simplicity, take x = IFFT(sqrt(P_full)) (real, zero-phase)
    x = np.fft.irfft(np.sqrt(np.maximum(P_full_best, 0)), n=N)
    
    return x

def _coordinate_descent_polish(x, max_iters=100):
    """Fast coordinate descent to polish the solution."""
    best_x = x.copy()
    best_val = _objective(best_x)
    
    deltas = [0.05, -0.05, 0.1, -0.1, 0.02, -0.02]
    
    for it in range(max_iters):
        if _check_deadline():
            break
        improved = False
        for j in range(N):
            for d in deltas:
                trial = best_x.copy()
                trial[j] += d
                val = _objective(trial)
                if val < best_val:
                    best_val = val
                    best_x = trial.copy()
                    improved = True
                    break
        if not improved:
            break
    
    return best_x, best_val

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    # Generate diverse initializations
    inits = []
    
    # Spectral designs with different seeds
    for seed in [0, 1, 2]:
        if _check_deadline():
            break
        try:
            x_spec = _spectral_design(seed=seed, beta=50.0)
            inits.append(x_spec)
        except Exception:
            pass
    
    # Standard initializations
    inits.append(np.ones(N))
    inits.append(np.random.RandomState(42).randn(N))
    inits.append(np.sin(2 * np.pi * np.arange(N) / 50.0))
    inits.append(np.sin(2 * np.pi * np.arange(N) / 20.0))
    inits.append(np.random.RandomState(7).rand(N) * 2 - 1)
    
    # Add some smooth structured inits
    t = np.linspace(0, 1, N)
    inits.append(1.0 + 0.5 * np.cos(np.pi * t))
    inits.append(1.0 + np.cos(np.pi * t))
    
    # Allocate time budget among restarts
    # We have ~110 seconds. Let's try 5-10 restarts.
    time_per_restart = 15
    
    for x0 in inits:
        if _check_deadline():
            break
        
        try:
            # Use L-BFGS-B with the smooth surrogate
            # We need to pass the objective and gradient
            # scipy minimize requires a function that returns (val, grad) if jac=True
            
            def _obj_grad(x, beta=50.0):
                return _soft_max_grad(x, beta)
            
            # Run minimization
            # We use a fixed beta for the surrogate. 
            # A larger beta approximates max better but makes gradient steeper.
            # Let's try beta=50 first.
            result = minimize(_obj_grad, x0, method='L-BFGS-B', jac=True,
                              options={'maxiter': 200, 'gtol': 1e-6, 'ftol': 1e-12})
            
            if result.success or result.fun < best_val:
                val_opt = _objective(result.x) # Evaluate true objective
                if val_opt < best_val:
                    best_val = val_opt
                    best_x = result.x.copy()
                    
        except Exception:
            pass
    
    # Final polish with coordinate descent
    if not _check_deadline():
        best_x, best_val = _coordinate_descent_polish(best_x, max_iters=100)
    
    return best_x
# EVOLVE-BLOCK-END
