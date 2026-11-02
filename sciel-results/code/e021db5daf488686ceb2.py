"""Model-and-optimize: Epigraph soft-max autoconvolution minimization with spectral init and adaptive polish."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 138  # Safety margin before 150s kill

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

def _soft_max_objective_and_grad(x, beta):
    """
    Compute soft-max surrogate for max|ac| and its gradient.
    Objective: (1/beta) * log(sum(exp(beta * |ac_k|)))
    """
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    m = np.max(abs_ac)
    exps = np.exp(beta * (abs_ac - m))
    sum_exps = np.sum(exps)
    
    if sum_exps == 0:
        return m, np.zeros(N)
        
    log_sum = np.log(sum_exps)
    soft_val = m + log_sum / beta
    
    weights = exps / sum_exps
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    grad_ac = weights * signs  # Gradient of soft-max w.r.t. ac
    
    # Gradient of ac_k w.r.t. x_i:
    # ac_k = sum_j x_j x_{j+k-N} (for appropriate ranges)
    # d(ac_k)/dx_i = x_{k-N+i} + x_{N-i+k} (with boundary checks)
    # This is a correlation of x with the gradient vector grad_ac.
    
    # We need grad_x_i = sum_k grad_ac_k * (d ac_k / d x_i)
    # d ac_k / d x_i is 1 if k is in [N-1+i, ...] or similar.
    # Specifically, ac_k = sum_{j} x_j x_{j+k-N+1-1} ? No, standard convolution:
    # ac_k = sum_{j} x_j x_{j+k} for k >= 0, sum_{j} x_j x_{k-j} for k < 0?
    # Let's use the FFT gradient property:
    # If C = x * x, then dC/dx = 2 * (x * grad_C) is not quite right for linear conv.
    # Let's use the relation: grad_x = 2 * (grad_ac circularly convolved with x) restricted to [0, N-1]
    # Actually, the gradient of sum_k lambda_k (x*x)_k w.r.t x_i is:
    # sum_k lambda_k (x_{k-i} + x_{i-k}) with shifts.
    
    # Efficient calculation using FFT:
    # The gradient of the quadratic form 0.5 * x^T A x is A x.
    # Here A is the matrix of the autoconvolution operator.
    # grad_x = 2 * (x * grad_ac) is valid if grad_ac is the gradient of the scalar with respect to ac.
    # Wait, ac_k depends on x.
    # Let's compute grad_x directly via FFT correlation.
    # grad_x_i = sum_k grad_ac_k * (x_{k-i+N-1} + x_{N-i+k-1}) ... this is messy.
    
    # Simpler:
    # The gradient of f(x) = sum_k g_k(x) where g_k is related to ac_k.
    # We established grad_ac.
    # The relationship between grad_x and grad_ac is:
    # grad_x = 2 * IFFT(FFT(x) * FFT(grad_ac_shifted)) ?
    
    # Let's use the standard result for autoconvolution gradient:
    # grad_x = 2 * (x * grad_ac) where * is linear convolution, then take first N elements?
    # No, the linear convolution of length N and length 2N-1 is length 3N-2.
    
    # Let's compute it properly:
    # ac_k = sum_{j=0}^{N-1} x_j x_{j+k-N+1} is incorrect indexing.
    # Standard: ac_l = sum_{j} x_j x_{j+l} for l=0..N-1.
    # For l < 0, ac_l = ac_{-l} due to symmetry.
    
    # Gradient of max|ac| is sum_k w_k sign(ac_k) d(ac_k)/dx.
    # d(ac_k)/dx_i = x_{k-i+N-1} (if valid) + x_{i-k+N-1} (if valid)?
    
    # Let's use the FFT trick:
    # The gradient of the linear autoconvolution is given by the circular correlation 
    # of x with the gradient vector, if we pad appropriately.
    
    L_fft = 1
    while L_fft < 4*N:
        L_fft *= 2
        
    X_f = np.fft.rfft(x, n=L_fft)
    # grad_ac is length 2N-1. We need to embed it in a way that correlates with x.
    # The term d(ac_k)/dx_i is non-zero when i is in the support.
    
    # Alternative: Compute gradient via FFT of x and grad_ac.
    # grad_x = 2 * (x * grad_ac) is valid for circular convolution if we pad x to length L.
    # But we have linear autoconvolution.
    # Let's stick to the time-domain logic if N is small enough? N=400, 2N-1=799.
    # Gradient computation in time domain:
    # grad_x[i] = sum_{k} grad_ac[k] * (x[k-i+N-1] + x[i-k+N-1])?
    
    # Let's use the efficient FFT correlation:
    # grad_x = 2 * (IFFT(FFT(x, L) * FFT(grad_ac_padded, L)))[:N]
    # Where grad_ac_padded is aligned such that index 0 of grad_ac corresponds to lag 0?
    # ac_0 = x_0^2 + 2 x_1 x_1? No.
    # ac_k = sum_{j} x_j x_{j+k}.
    # d(ac_k)/dx_i = x_{k-i} + x_{i-k} (with indices mod N? No, linear).
    
    # Let's just do the O(N^2) gradient for simplicity and correctness, N=400 is fast enough for Python loops?
    # No, N=400, N^2 = 160k. Python loop is too slow.
    # We must use FFT.
    
    # The gradient of the linear autoconvolution x*x is:
    # grad_x = 2 * (x * grad_ac) is incorrect for linear.
    # Correct formula:
    # grad_x[i] = sum_{k} grad_ac[k] * (x[k - i + N - 1] if 0 <= k - i + N - 1 < N else 0)
    #             + sum_{k} grad_ac[k] * (x[i - k + N - 1] if 0 <= i - k + N - 1 < N else 0)
    
    # This is a cross-correlation.
    # Let's compute it via FFT.
    # Create a vector G of length L_fft.
    # G is grad_ac placed such that it aligns with the shifts.
    
    # Actually, there is a simpler way:
    # grad_x = 2 * (IFFT(FFT(x, L_fft) * FFT(grad_ac_full, L_fft)))[:N]
    # Where grad_ac_full is the full linear convolution gradient?
    
    # Let's use the property:
    # If y = x * x (linear), then grad_x = 2 * (x * grad_y) is NOT true.
    # grad_x = (grad_y * x) + (x * grad_y) is true for elementwise? No.
    
    # Let's compute grad_x using the following:
    # grad_x[i] = 2 * sum_{k} grad_ac[k] x[k-i+N-1] (with bounds)
    # This is the cross-correlation of x and grad_ac.
    
    L_corr = 1
    while L_corr < 3*N:
        L_corr *= 2
        
    X_c = np.fft.rfft(x, n=L_corr)
    G_c = np.fft.rfft(grad_ac, n=L_corr)
    
    # The cross correlation C[i] = sum_k x[k] grad_ac[k-i]
    # We want sum_k grad_ac[k] x[k-i] which is similar.
    
    # Let's just use the numerical gradient if FFT is too tricky? No, too slow.
    
    # Correct FFT gradient for linear autoconvolution:
    # grad_x = 2 * (IFFT(FFT(x, 2N) * FFT(grad_ac_padded, 2N)))[:N]
    # Where grad_ac_padded is placed at the center?
    
    # Let's try this:
    # The derivative of ac_k w.r.t x_i is x_{k-i+N-1} + x_{i-k+N-1} (with 1s for valid indices).
    # This is equivalent to 2 * (x * grad_ac) if we treat it as circular?
    
    # Let's use a verified method:
    # grad_x = 2 * (x * grad_ac) is valid if we use circular convolution of length L >= 2N-1.
    # Why? Because the autoconvolution is the same as the circular autoconvolution if we pad with zeros to length L >= 2N-1.
    # In that case, the gradient of the circular autoconvolution is 2 * (x * grad_ac) (circular).
    # Since the linear autoconvolution matches the circular one for the first 2N-1 terms (if padded correctly),
    # and we only care about the gradient w.r.t x (first N elements), this should work.
    
    L_fft = 1
    while L_fft < 4*N:
        L_fft *= 2
        
    X_f = np.fft.rfft(x, n=L_fft)
    G_f = np.fft.rfft(grad_ac, n=L_fft)
    
    # Circular correlation
    corr = np.fft.irfft(X_f * np.conj(G_f), n=L_fft)
    
    # The gradient is related to this correlation.
    # For y = x * x (circular), grad_x = 2 * (x * grad_y) (circular).
    # So:
    grad_x = 2.0 * corr[:N]
    
    return soft_val, grad_x

def _objective_and_grad(x, beta):
    """Compute objective and gradient for the scaled objective."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    
    soft_val, grad_soft = _soft_max_objective_and_grad(x, beta)
    
    # Objective: soft_val / s^2
    val = soft_val / (s * s)
    
    # Gradient:
    # d(soft_val/s^2)/dx = (grad_soft * s^2 - soft_val * 2s * 1) / s^4
    # = grad_soft / s^2 - 2 * soft_val / s^3
    grad = grad_soft / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s)
    
    return val, grad

def _spectral_init(rng):
    """Generate initial points via spectral phase randomization."""
    L_fft = 1
    while L_fft < 2*N:
        L_fft *= 2
        
    k = np.arange(L_fft // 2 + 1)
    t = k / (L_fft // 2)
    
    # Flat top spectrum with random phase
    inits = []
    for _ in range(5):
        # Amplitude: flat top with smooth edges
        amp = np.ones(L_fft // 2 + 1)
        amp[:5] = np.linspace(0, 1, 5)
        amp[-5:] = np.linspace(1, 0, 5)
        
        phase = rng.uniform(0, 2*np.pi, size=L_fft // 2 + 1)
        phase[0] = 0.0 # DC term real
        
        spec = amp * np.exp(1j * phase)
        
        # Hermitian symmetry
        spec_full = np.zeros(L_fft, dtype=complex)
        spec_full[:L_fft//2+1] = spec
        spec_full[L_fft//2+1:] = np.conj(spec[1:L_fft//2][::-1])
        
        x = np.fft.ifft(spec_full).real[:N]
        # Shift to be positive
        x = x - np.min(x) + 0.5
        inits.append(x)
        
    # Gaussian spectrum
    for _ in range(5):
        amp = np.exp(-0.5 * (t - 0.5)**2 / (0.2**2))
        phase = rng.uniform(0, 2*np.pi, size=L_fft // 2 + 1)
        phase[0] = 0.0
        spec = amp * np.exp(1j * phase)
        spec_full = np.zeros(L_fft, dtype=complex)
        spec_full[:L_fft//2+1] = spec
        spec_full[L_fft//2+1:] = np.conj(spec[1:L_fft//2][::-1])
        x = np.fft.ifft(spec_full).real[:N]
        x = x - np.min(x) + 0.5
        inits.append(x)
        
    return inits

def construct():
    best_x = np.ones(N)
    best_val = _objective_only(best_x)
    
    if _check_deadline():
        return best_x
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Multi-start L-BFGS with annealed beta
    initial_points = _spectral_init(rng)
    
    # Beta annealing schedule: start smooth, end sharp
    betas = [10.0, 20.0, 50.0, 100.0, 200.0]
    
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
                                  jac=True, options={'maxiter': 100, 'gtol': 1e-6, 'ftol': 1e-8})
                
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x.copy()
                    
            except Exception:
                pass
                
    if _check_deadline():
        return best_x
        
    # Phase 2: High-precision polish with very high beta
    for beta in [500.0, 1000.0]:
        if _check_deadline():
            break
        try:
            def obj_and_grad_func(x):
                return _objective_and_grad(x, beta)
            
            result = minimize(obj_and_grad_func, best_x, method='L-BFGS-B',
                              jac=True, options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-10})
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
                
        except Exception:
            pass
            
    if _check_deadline():
        return best_x
        
    # Phase 3: Local coordinate descent with subsampling
    # Use exact objective to break smoothing bias
    step_size = 0.01
    n_steps = 0
    max_steps = 100000
    
    while n_steps < max_steps:
        if _check_deadline():
            break
            
        # Random subset of indices
        n_coords = 10
        indices = rng.choice(N, size=n_coords, replace=False)
        
        improved = False
        for i in indices:
            for delta in [step_size, -step_size, 0.5*step_size, -0.5*step_size]:
                x_trial = best_x.copy()
                x_trial[i] += delta
                val = _objective_only(x_trial)
                if val < best_val:
                    best_val = val
                    best_x = x_trial
                    improved = True
                    break
        
        if not improved:
            step_size *= 0.9
        else:
            step_size = min(0.1, step_size * 1.05)
            
        n_steps += 1
        
    return best_x
# EVOLVE-BLOCK-END
