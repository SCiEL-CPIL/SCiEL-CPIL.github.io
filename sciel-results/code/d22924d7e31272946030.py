"""Combined program: Spectral-domain parameterization + smooth optimization + gradient polish."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 400

def _autoconv(x):
    """Linear autoconvolution via FFT."""
    n = len(x)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    X = np.fft.rfft(x, fsize)
    c = np.fft.irfft(X * X, fsize)[:size]
    return c

def _metric_offdiag(x):
    """Compute max |c_k| for k != 0 divided by (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv(x)
    if len(c) > 1:
        max_val = np.max(np.abs(c[1:]))
    else:
        max_val = 0.0
    return max_val / (s * s)

def _objective_smooth(x, temp=1.0):
    """Smooth approximation of the off-diagonal metric for optimization."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    c = _autoconv(x)
    if len(c) > 1:
        abs_c = np.abs(c[1:])
    else:
        return 0.0
    max_c = np.max(abs_c)
    soft_max = max_c + temp * np.log(np.sum(np.exp((abs_c - max_c) / temp)))
    return soft_max / (s * s)

def _gradient_descent_polish(x0, max_iter=500, initial_lr=0.01, deadline=None):
    """Projected gradient descent with annealed temperature for final polish."""
    x = x0.copy().astype(float)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return x
    x = x / s
    
    best_x = x.copy()
    best_val = _metric_offdiag(best_x)
    
    lr = initial_lr
    temp = 0.5
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        c = _autoconv(x)
        if len(c) <= 1:
            break
        abs_c = np.abs(c[1:])
        
        max_c = np.max(abs_c)
        if max_c < 1e-15:
            break
        
        exp_vals = np.exp((abs_c - max_c) / temp)
        sum_exp = np.sum(exp_vals)
        soft_max = max_c + temp * np.log(sum_exp)
        
        grad_abs_c = exp_vals / sum_exp
        c_off = c[1:]
        sign_c = np.sign(c_off)
        grad_c = grad_abs_c * sign_c
        
        w = np.zeros(2 * N - 1)
        w[1:] = grad_c
        
        fsize = 1
        while fsize < 2 * N:
            fsize *= 2
        X = np.fft.rfft(x, fsize)
        W = np.fft.rfft(w, fsize)
        grad_x = np.fft.irfft(W * np.conj(X), fsize)[:N]
        
        s = np.sum(x)
        grad_metric = grad_x / (s * s) - 2.0 * soft_max * np.ones(N) / (s ** 3)
        
        x_new = x - lr * grad_metric
        
        s_new = np.sum(x_new)
        if abs(s_new) < 1e-12:
            x_new = x + 1e-6 * np.random.randn(N)
            s_new = np.sum(x_new)
        x_new = x_new / s_new
        
        val = _metric_offdiag(x_new)
        if val < best_val:
            best_val = val
            best_x = x_new.copy()
        
        x = x_new
        lr *= 0.995
        temp *= 0.999
        
    return best_x

def _spectral_to_signal(coeffs, phases):
    """Convert spectral coefficients and phases to a real signal."""
    n = len(coeffs)
    signal = np.zeros(N)
    for k in range(n):
        freq = (k + 1) * N // (2 * n + 1)
        if freq >= N // 2:
            break
        # Add cosine component
        t = np.arange(N)
        signal += coeffs[k] * np.cos(2 * np.pi * freq * t / N + phases[k])
    return signal

def _spectral_objective(params):
    """Objective for spectral parameter optimization."""
    n_coeffs = len(params) // 2
    coeffs = np.abs(params[:n_coeffs]) + 0.01  # Ensure positive
    phases = params[n_coeffs:]
    
    signal = _spectral_to_signal(coeffs, phases)
    s = np.sum(signal)
    if abs(s) < 1e-12:
        return 1e10
    
    c = _autoconv(signal)
    if len(c) > 1:
        abs_c = np.abs(c[1:])
        max_c = np.max(abs_c)
        # Use hard max for spectral optimization (faster, less smooth)
        return max_c / (s * s)
    return 0.0

def construct():
    t0 = time.time()
    deadline = t0 + 110.0
    
    best_x = np.ones(N)
    best_val = _metric_offdiag(best_x)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Spectral-domain global optimization (reduced dimensionality)
    if time.time() < deadline - 40:
        # Try different numbers of spectral components
        for n_comp in [4, 6, 8]:
            if time.time() > deadline - 40:
                break
            
            # Bounds for coefficients and phases
            bounds = [(0, 1.0)] * n_comp + [(0, 2 * np.pi)] * n_comp
            
            # Multiple random starts for differential evolution
            for seed in range(3):
                if time.time() > deadline - 40:
                    break
                
                try:
                    result = differential_evolution(
                        _spectral_objective,
                        bounds,
                        maxiter=50,
                        popsize=15,
                        tol=1e-6,
                        seed=seed,
                        polish=False
                    )
                    
                    if result.success or result.fun < best_val:
                        params = result.x
                        n_coeffs = len(params) // 2
                        coeffs = np.abs(params[:n_coeffs]) + 0.01
                        phases = params[n_coeffs:]
                        
                        signal = _spectral_to_signal(coeffs, phases)
                        val = _metric_offdiag(signal)
                        
                        if val < best_val:
                            best_val = val
                            best_x = signal.copy()
                except Exception:
                    pass
    
    # Phase 2: Local refinement with L-BFGS-B on best spectral solution
    if time.time() < deadline - 30:
        x0 = best_x.copy()
        s0 = np.sum(x0)
        if abs(s0) > 1e-6:
            x0 = x0 / s0
            
        try:
            # Annealing temperature for smooth objective
            for temp in [1.0, 0.5, 0.1, 0.05]:
                if time.time() > deadline - 30:
                    break
                
                res = minimize(
                    lambda x: _objective_smooth(x, temp=temp),
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                x0 = res.x
                val = _metric_offdiag(x0)
                if val < best_val:
                    best_val = val
                    best_x = x0.copy()
        except Exception:
            pass
    
    # Phase 3: Additional diverse candidates and local optimization
    if time.time() < deadline - 20:
        candidates = []
        
        # Random sequences with non-zero mean
        for _ in range(10):
            x = rng.choice([-1, 1], N)
            if abs(np.sum(x)) > 10:
                candidates.append(x)
        
        # Structured patterns
        for block_size in [10, 20, 50]:
            pattern = np.zeros(N)
            for i in range(N):
                if (i // block_size) % 2 == 0:
                    pattern[i] = 1.0
                else:
                    pattern[i] = -1.0
            pattern[0] += 0.1
            candidates.append(pattern)
        
        for x0 in candidates:
            if time.time() > deadline - 20:
                break
            
            x0 = x0.copy()
            s0 = np.sum(x0)
            if abs(s0) < 1e-6:
                continue
            x0 = x0 / s0
            
            try:
                res = minimize(
                    lambda x: _objective_smooth(x, temp=0.5),
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                x_opt = res.x
                val = _metric_offdiag(x_opt)
                if val < best_val:
                    best_val = val
                    best_x = x_opt.copy()
            except Exception:
                pass
    
    # Phase 4: Final gradient descent polish
    if time.time() < deadline - 5:
        try:
            polished = _gradient_descent_polish(
                best_x,
                max_iter=1000,
                initial_lr=0.005,
                deadline=deadline - 5
            )
            val = _metric_offdiag(polished)
            if val < best_val:
                best_val = val
                best_x = polished
        except Exception:
            pass
    
    return best_x
