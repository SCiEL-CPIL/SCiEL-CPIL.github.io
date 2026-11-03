import numpy as np
import time
from scipy.optimize import minimize

N = 400
START_TIME = time.time()
TIME_LIMIT = 115
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective_normalized(x, beta=100.0):
    """
    Compute soft-max of |autoconv| for x normalized to sum 1.
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(x)
    
    # Normalize x to sum 1 for stable objective
    # Note: We optimize on the normalized manifold, but L-BFGS-B works in unconstrained space.
    # We will use a penalty or just rely on the fact that the objective is scale invariant 
    # if we define it correctly? No, max|C| is homogeneous of degree 2. Sum is degree 1.
    # Ratio is degree 1? No. C is deg 2, Sum^2 is deg 2. Ratio is scale invariant.
    # So we can fix sum(x)=1.
    
    # To use L-BFGS-B effectively, we should optimize x such that sum(x)=1.
    # We can parameterize x to satisfy this, or use a penalty.
    # Penalty: 1e6 * (sum(x) - 1)^2
    
    ac = _autoconv(x)
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    
    if max_val < 1e-15:
        return 1e10, np.zeros_like(x)
    
    # Soft max
    shifted = beta * (abs_ac - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max = max_val + (1.0 / beta) * np.log(sum_exp)
    
    # The value we want to minimize is soft_max / (sum(x))^2
    # But if we force sum(x)=1 via penalty, we just minimize soft_max + penalty
    
    val = soft_max
    penalty = 1e6 * (np.sum(x) - 1.0)**2
    total_val = val + penalty
    
    # Gradient
    # d(soft_max)/dx[j] = sum_k weights[k] * sign(ac[k]) * d(ac[k])/dx[j]
    g = np.zeros(2*N-1)
    g[:len(ac)] = weights * np.sign(ac)
    
    # d(ac[k])/dx[j] = x[j+k] + x[j-k]
    # This is a correlation.
    L_fft = 1
    while L_fft < 3*N:
        L_fft *= 2
    
    G = np.fft.rfft(g, n=L_fft)
    Xf = np.fft.rfft(x, n=L_fft)
    
    # term1: sum_k g[k] x[j+k] -> correlation
    # term2: sum_k g[k] x[j-k] -> correlation
    # Using FFT:
    # corr1[j] = sum_k g[k] x[j+k]
    # This is equivalent to convolution of g with flipped x?
    # Let's use the property: d(ac)/dx is a linear operator.
    # ac = x * x.
    # grad = 2 * (g * x) (circular correlation?) No, linear.
    
    # Efficient calculation:
    # grad_x = 2 * ifft(fft(g) * fft(x)) ??
    # Let's derive:
    # ac[k] = sum_i x[i] x[i+k]
    # d(ac[k])/dx[j] = x[j+k] + x[j-k]
    # sum_k w[k] (x[j+k] + x[j-k])
    # = sum_k w[k] x[j+k] + sum_k w[k] x[j-k]
    
    # Let w be zero-padded to length L_fft.
    # Let x be zero-padded to length L_fft.
    # sum_k w[k] x[j+k] is cross-correlation.
    # corr1 = ifft(conj(fft(w)) * fft(x))
    # corr2 = ifft(fft(w) * fft(x))
    
    corr1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft)[:N]
    corr2 = np.fft.irfft(G * Xf, n=L_fft)[:N]
    
    d_softmax_dx = corr1 + corr2
    
    # Gradient of penalty
    d_penalty_dx = 2 * 1e6 * (np.sum(x) - 1.0) * np.ones(N)
    
    grad = d_softmax_dx + d_penalty_dx
    
    return total_val, grad

def construct():
    best_x = np.ones(N) / N
    best_val = _objective_normalized(best_x, beta=100.0)[0]
    
    # Initializations
    initializations = [
        np.ones(N) / N,
        np.linspace(0, 2/N, N),
        np.linspace(2/N, 0, N),
        (1.0 + np.cos(np.pi * np.linspace(0, 1, N))) / np.sum(1.0 + np.cos(np.pi * np.linspace(0, 1, N))),
    ]
    
    # Random smooth initializations
    rng = np.random.default_rng(123)
    for _ in range(50):
        freqs = rng.uniform(0.1, 5.0, size=5)
        phases = rng.uniform(0, 2*np.pi, size=5)
        amps = rng.uniform(0.1, 0.5, size=5)
        sig = 1.0 + sum(a * np.cos(2*np.pi*f*np.linspace(0, 1, N) + p) for a, f, p in zip(amps, freqs, phases))
        sig = sig / np.sum(sig)
        initializations.append(sig)
    
    # Evaluate initializations
    for x0 in initializations:
        if _check_deadline():
            break
        val, _ = _objective_normalized(x0, beta=100.0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()

    # Optimization phases with increasing beta
    betas = [10, 20, 50, 100, 200, 500]
    
    current_x = best_x.copy()
    
    for beta in betas:
        if _check_deadline():
            break
            
        def _obj(x, b=beta):
            return _objective_normalized(x, beta=b)
            
        try:
            result = minimize(_obj, current_x, method='L-BFGS-B', jac=True,
                              options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-12})
            if result.fun < best_val:
                # Normalize result
                res_x = result.x / np.sum(result.x)
                # Re-evaluate with actual max to be safe
                ac = _autoconv(res_x)
                true_val = np.max(np.abs(ac)) / (np.sum(res_x)**2)
                # Since we normalized sum to 1 in objective, true_val is approx result.fun
                # But let's use the exact metric for comparison
                if true_val < best_val:
                    best_val = true_val
                    best_x = res_x.copy()
                    current_x = res_x.copy()
        except Exception:
            pass
            
    # Final polish: Coordinate Descent on the peak
    if not _check_deadline():
        # Identify peak lag
        ac = _autoconv(best_x)
        abs_ac = np.abs(ac)
        peak_lag = np.argmax(abs_ac)
        
        # Try to reduce the peak by adjusting x
        # Simple gradient step on the exact max? No, non-diff.
        # Use high beta L-BFGS-B again
        def _final_obj(x):
            return _objective_normalized(x, beta=1000.0)
            
        try:
            result = minimize(_final_obj, best_x, method='L-BFGS-B', jac=True,
                              options={'maxiter': 300, 'gtol': 1e-10, 'ftol': 1e-14})
            if result.fun < best_val:
                res_x = result.x / np.sum(result.x)
                ac = _autoconv(res_x)
                true_val = np.max(np.abs(ac)) / (np.sum(res_x)**2)
                if true_val < best_val:
                    best_val = true_val
                    best_x = res_x.copy()
        except Exception:
            pass
            
    return best_x
