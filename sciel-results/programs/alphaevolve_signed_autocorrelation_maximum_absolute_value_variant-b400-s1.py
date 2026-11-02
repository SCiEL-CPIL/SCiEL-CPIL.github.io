"""Model-and-optimize: parameterize via FFT-based gradient descent on smooth surrogate."""
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

def _soft_max_and_grad(x, beta):
    """
    Compute soft-max surrogate for max|ac| and its gradient.
    Uses log-sum-exp for stability.
    Returns (soft_max_value, gradient_array)
    """
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    # Soft max: (1/beta) * log(sum(exp(beta * abs_ac)))
    # For stability: m = max(abs_ac), soft = m + (1/beta)*log(sum(exp(beta*(abs_ac - m))))
    m = np.max(abs_ac)
    exps = np.exp(beta * (abs_ac - m))
    log_sum = np.log(np.sum(exps))
    soft_val = m + log_sum / beta
    
    # Gradient of soft-max w.r.t. ac:
    # d/d(ac[k]) soft_max = exps[k] / sum(exps) * sign(ac[k])
    weights = exps / np.sum(exps)
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    grad_ac = weights * signs  # d(soft_max)/d(ac[k])
    
    # Now gradient w.r.t. x:
    # ac[k] = sum_j x[j]*x[k-j]
    # d(soft_max)/dx[j] = sum_k grad_ac[k] * d(ac[k])/dx[j]
    # d(ac[k])/dx[j] = x[k-j] + x[j] (with boundary conditions)
    # More precisely: d(ac[k])/dx[j] = x[k-j] if 0<=k-j<N, plus x[j] if 0<=k-j<N and j != k-j
    # Actually: ac[k] = sum_{j} x[j]*x[k-j] where terms with invalid indices are 0
    # d(ac[k])/dx[j] = x[k-j] * I(0<=k-j<N) + x[j] * I(0<=k-j<N)
    # But when j = k-j (i.e., 2j=k), it's 2*x[j] * I(0<=j<N)
    # When j != k-j, it's x[k-j] + x[j] if both valid
    # 
    # Efficient computation: 
    # grad_x[j] = sum_k grad_ac[k] * (x[k-j] + x[j]) where valid
    # = x[j] * sum_k grad_ac[k] * I(0<=k-j<N) + sum_k grad_ac[k]*x[k-j]*I(0<=k-j<N)
    # Let k' = k-j, then second term = sum_{k'} grad_ac[k'+j] * x[k'] * I(0<=k'<N and 0<=k'+j<N)
    # This is a convolution-like operation.
    
    # Alternative: use the fact that the gradient can be computed as:
    # grad_x = 2 * ifft(fft(grad_ac_full) * fft(x_full)) where we zero-pad appropriately
    # But we need to be careful with the linear convolution structure.
    
    # Let's do it directly but efficiently using FFT:
    # grad_x[j] = sum_{k: 0<=k-j<N} grad_ac[k] * (x[k-j] + x[j])
    # = x[j] * sum_{k: 0<=k-j<N} grad_ac[k] + sum_{k: 0<=k-j<N} grad_ac[k] * x[k-j]
    
    # For the first term: for each j, sum grad_ac[k] over k in [j, j+N-1] intersect [0, 2N-2]
    # This is a sliding window sum - can use cumulative sum
    
    # For the second term: it's like a correlation of grad_ac with x
    # sum_k grad_ac[k] * x[k-j] = (grad_ac * x)[j] in some sense
    
    # Let me just compute it with a clean approach:
    # Pad grad_ac to length 2N-1 (it already is)
    # The gradient w.r.t. x is:
    # grad_x = convolve(grad_ac, x) + convolve(x, grad_ac) but restricted...
    
    # Actually, the cleanest way: 
    # d(soft)/dx = 2 * (grad_ac ⊛ x) where ⊛ is the appropriate convolution
    # Let's use: create full gradient via the relation
    # ac = ifft(fft(x)^2)
    # d(soft)/dx = ifft(2 * fft(x) * fft(grad_ac_padded)) ... no, that's not right for linear conv.
    
    # Let me just do the direct O(N^2) but it's too slow for N=400 in the inner loop.
    # 
    # Better: use the spectral method. 
    # The autoconvolution is ac = ifft(|fft(x)|^2) (linear, zero-padded)
    # The gradient of a function f(ac) w.r.t. x is:
    # d(f)/dx = 2 * ifft(fft(x) * fft(2*grad_ac - grad_ac[N-1] ... )) 
    # This is getting complicated. Let me use a simpler numerical approach for the gradient
    # that's still fast enough.
    
    # Actually, for N=400, a direct computation of the gradient taking O(N^2) per step
    # with N=400 means 160K ops per gradient eval. With maybe 5000 iterations, that's 800M ops.
    # Too slow in Python. Let me use FFT-based.
    
    # The correct FFT-based gradient:
    # If ac[k] = sum_j x[j]*x[k-j] (linear conv, zero-padded to length 2N-1)
    # Then in frequency domain: FFT(ac) = FFT(x) * FFT(x) (with zero-padding)
    # d(f)/dx[j] = sum_k grad_ac[k] * d(ac[k])/dx[j]
    # d(ac[k])/dx[j] = x[k-j] (if 0<=k-j<N) + x[j] (if 0<=k-j<N)
    # 
    # Let g = grad_ac (length 2N-1)
    # grad_x[j] = sum_k g[k]*x[k-j]*I(0<=k-j<N) + sum_k g[k]*x[j]*I(0<=k-j<N)
    # 
    # Second term: x[j] * sum_{k: j<=k<=j+N-1} g[k] = x[j] * (cumsum_g[j+N-1] - cumsum_g[j-1])
    # First term: sum_k g[k]*x[k-j]*I(0<=k-j<N) = sum_{m=0}^{N-1} g[m+j]*x[m] where m=k-j
    # This is a "sliding inner product" - can be computed via correlation.
    
    # Use np.correlate or manual FFT:
    # First term for all j: correlate(g, x) but shifted
    # sum_m g[m+j]*x[m] for m in [0,N-1]
    # This is like cross-correlation of g and x, evaluated at lag j
    
    # Let's compute both terms:
    # Term 2: window sum of g
    g = grad_ac  # length 2N-1
    cumsum_g = np.concatenate([[0], np.cumsum(g)])
    # For j in [0, N-1]: sum_{k=j}^{j+N-1} g[k] = cumsum_g[min(j+N-1, 2N-2)+1] - cumsum_g[j]
    idx_end = np.minimum(np.arange(N) + N - 1, 2*N-2) + 1
    idx_start = np.arange(N)
    window_sums = cumsum_g[idx_end] - cumsum_g[idx_start]
    term2 = x * window_sums
    
    # Term 1: sum_{m=0}^{N-1} g[m+j]*x[m] for each j
    # This is cross-correlation. Let's compute via FFT:
    # corr[j] = sum_m g[m+j]*x[m]
    # = sum_m x[m]*g[m+j]
    # This is the correlation of x and g at lag j
    # Using FFT: ifft(fft(x, 2N) * conj(fft(g, 2N))) gives circular correlation
    # But we need linear. Since g has length 2N-1 and x has length N, 
    # we can zero-pad to length >= 3N-1 for safety, or just use length 4N.
    
    # For correlation sum_m x[m]*g[m+j], we need FFT length >= N + (2N-1) - 1 = 3N-2
    # But we only need first N values. Use 4N for safety (next power of 2).
    L_corr = 1
    while L_corr < 3 * N:
        L_corr *= 2
    X_f = np.fft.rfft(x, n=L_corr)
    G_f = np.fft.rfft(g, n=L_corr)
    # Cross-correlation: sum_m x[m]*g[m+j]
    # Using: ifft(fft(x) * conj(fft(g))) gives (x * g_flipped) which is not quite right
    # Let's use: corr[j] = sum_m x[m]*g[m+j]
    # This equals the cross-correlation. In FFT terms:
    # ifft(fft(x) * conj(fft(g))) gives circular correlation
    # For linear, with sufficient zero-padding, corr[j] for j=0..N-1 is at index j
    corr = np.fft.irfft(X_f * np.conj(G_f), n=L_corr)
    term1 = corr[:N]
    
    grad_x = term1 + term2
    return soft_val, grad_x

def _objective_only(x, beta):
    """Compute objective value only (fast, no gradient)."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if beta < 1e6:
        exps = np.exp(beta * (abs_ac - m))
        log_sum = np.log(np.sum(exps))
        soft_val = m + log_sum / beta
    else:
        soft_val = m
    return soft_val / (s * s)

def _objective_and_grad(x, beta):
    """Compute objective and gradient."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    soft_val, grad_soft = _soft_max_and_grad(x, beta)
    # Objective = soft_val / s^2
    val = soft_val / (s * s)
    # Gradient: d/dx (soft/s^2) = grad_soft/s^2 - 2*soft/s^3 * ones
    grad = grad_soft / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s)
    return val, grad

def construct():
    best_x = np.ones(N)
    best_val = _objective_and_grad(best_x, 30.0)[0]
    
    if _check_deadline():
        return best_x
    
    # Multiple random restarts with L-BFGS-B
    rng = np.random.default_rng(12345)
    
    # Try different betas (smoothness levels)
    betas = [20.0, 30.0, 50.0, 100.0]
    
    # Initial points - more diverse
    t = np.linspace(0, 1, N)
    initial_points = [
        np.ones(N),
        np.ones(N) * 0.5,
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
        np.maximum(0.5, 1.0 - 2.0 * np.abs(t - 0.5)),  # tent
        np.maximum(0.5, 1.0 - 3.0 * np.abs(t - 0.5)),  # sharper tent
    ]
    
    # Add random initializations with varied scales
    for _ in range(20):
        scale = rng.uniform(0.5, 1.5)
        initial_points.append(scale * rng.normal(1.0, 0.2, size=N))
    for _ in range(5):
        # Sparse-like initializations
        x_sparse = rng.choice([0.0, 1.0, 2.0], size=N, p=[0.3, 0.5, 0.2])
        initial_points.append(x_sparse)
    
    for beta in betas:
        if _check_deadline():
            break
        for x0 in initial_points:
            if _check_deadline():
                break
            
            # Add small noise to avoid exact symmetry
            x_start = x0 + rng.normal(0, 0.01, size=N)
            
            try:
                def obj_func(x):
                    return _objective_and_grad(x, beta)[0]
                
                def obj_and_grad_func(x):
                    val, grad = _objective_and_grad(x, beta)
                    return val, grad
                
                result = minimize(obj_and_grad_func, x_start, method='L-BFGS-B',
                                  jac=True, options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-10})
                
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x.copy()
                    
            except Exception:
                pass
    
    if _check_deadline():
        return best_x
    
    # Final polish with higher beta
    for beta in [200.0, 500.0, 1000.0]:
        if _check_deadline():
            break
        try:
            def obj_and_grad_func(x, _beta=beta):
                return _objective_and_grad(x, _beta)
            
            result = minimize(obj_and_grad_func, best_x, method='L-BFGS-B',
                              jac=True, options={'maxiter': 200, 'gtol': 1e-10, 'ftol': 1e-12})
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
                
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Final random perturbation refinement using fast objective-only
    step_size = 0.005
    beta_fast = 100.0
    for _ in range(100000):
        if _check_deadline():
            break
        perturbation = rng.normal(0, step_size, size=N)
        x_trial = best_x + perturbation
        val = _objective_only(x_trial, beta_fast)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            step_size *= 0.998
        else:
            step_size = min(0.05, step_size * 1.005)
    
    return best_x
# EVOLVE-BLOCK-END
