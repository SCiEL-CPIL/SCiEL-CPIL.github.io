"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Direct convex optimization over the 600 step heights using FISTA
(Fast Iterative Shrinkage-Thresholding Algorithm) with a smooth approximation
of the max function. The objective is to minimize the smooth max of the
autoconvolution normalized by the square of the L1 norm. We use a spectral
initialization based on the idea that spreading energy in the frequency domain
can reduce time-domain peaks.
"""

import numpy as np
import time

N = 600
TIME_LIMIT = 140  # Safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def smooth_max(x, beta):
    """Smooth approximation to max using log-sum-exp."""
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x)
    if x_max < 1e-15:
        return 0.0
    x_scaled = x / x_max
    shifted = x_scaled - np.max(x_scaled)
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    return x_max * (1.0 + log_sum)

def objective_and_grad(f, beta):
    """
    Compute the objective value and gradient for the smooth max of autoconvolution
    normalized by (sum f)^2.
    
    Objective: J(f) = M_beta(conv(f,f)) / (sum f)^2
    where M_beta is the smooth max.
    
    Gradient is computed using the chain rule.
    """
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf, np.zeros_like(f)
    
    conv = np.convolve(f, f, mode='full')
    
    # Compute smooth max and its gradient with respect to conv
    x = conv
    x_max = np.max(x)
    if x_max < 1e-15:
        return 0.0, np.zeros_like(f)
    
    x_scaled = x / x_max
    shifted = x_scaled - np.max(x_scaled)
    exp_shifted = np.exp(beta * shifted)
    sum_exp = np.sum(exp_shifted)
    
    # Gradient of smooth_max with respect to x
    # d/dx [ x_max * (1 + log(sum(exp(beta*(x/x_max - max(x/x_max)))))/beta ) ]
    # This is tricky because x_max depends on x.
    # Alternative: Use the standard log-sum-exp gradient directly on x.
    # LSE(x) = log(sum(exp(x)))
    # grad LSE = exp(x) / sum(exp(x))
    # But we want to approximate max(x).
    # Let's use the form: max(x) ~ (1/beta) * log(sum(exp(beta * x)))
    # grad = exp(beta * x) / sum(exp(beta * x))
    
    # To avoid overflow, we shift by max
    shifted_x = beta * (x - x_max)
    exp_shifted_x = np.exp(shifted_x)
    sum_exp_shifted = np.sum(exp_shifted_x)
    
    # Gradient of (1/beta) * log(sum(exp(beta * x))) with respect to x
    grad_m = exp_shifted_x / sum_exp_shifted
    
    # Now J(f) = M / s^2
    # dJ/df = (1/s^2) * dM/df - (2 * M / s^3) * ds/df
    # dM/df = grad_m @ (d conv / d f)
    # d conv / d f is a linear operator: (d conv / d f)[k] = sum_j grad_m[j] * f[j-k] (with appropriate indexing)
    # Actually, conv[k] = sum_j f[j] f[k-j]
    # d conv[k] / d f[i] = f[k-i] + f[i-k] (if indices valid)
    # So dM/df[i] = sum_k grad_m[k] * (f[k-i] + f[i-k])
    # This is a convolution of grad_m with f, plus a reversal.
    # Since conv is symmetric, dM/df = 2 * convolve(grad_m, f, mode='same') ?
    # Let's check:
    # conv[k] = sum_j f[j] f[k-j]
    # d conv[k] / d f[i] = f[k-i] (from first term) + f[i-k] (from second term, where j=k-i -> f[j] becomes f[i] and f[k-j] becomes f[k-i])
    # Wait, the second term is f[k-j] where j is the index of f. If we differentiate w.r.t f[i], we get f[k-j] when j=i, so f[k-i].
    # And the first term f[j] when j=i is f[i], times f[k-i].
    # So d conv[k] / d f[i] = f[k-i] + f[i-k].
    # Since f is real, f[k-i] and f[i-k] are just values.
    # The vector dM/df is sum_k grad_m[k] * (f[k-i] + f[i-k])
    # This is equivalent to 2 * convolve(grad_m, f, mode='same') if we handle boundaries carefully.
    # Actually, np.convolve(a, b, 'same') gives the central part of the full convolution.
    # Let's use a direct computation or FFT for speed.
    
    # Using FFT for efficiency
    # grad_m is length 2N-1. f is length N.
    # We need the correlation/convolution that gives a vector of length N.
    # dM/df[i] = sum_k grad_m[k] * (f[k-i] + f[i-k])
    # This is the same as the cross-correlation of grad_m and f, evaluated at lags.
    # Let's use np.correlate or FFT.
    
    # Extend f to length 2N-1 with zeros
    f_ext = np.zeros(2*N - 1)
    f_ext[:N] = f
    
    # grad_m is already length 2N-1
    # We want sum_k grad_m[k] * f_ext[k-i] for i in 0..N-1
    # This is the correlation of grad_m and f_ext.
    # np.correlate(grad_m, f_ext, mode='full')
    # The result length is 2N-1 + N - 1 = 3N-2.
    # We need the central N points corresponding to i=0..N-1.
    
    # Alternative: Use the property that dM/df is the gradient of the quadratic form f^T A f where A is defined by grad_m.
    # Let's just compute it via FFT for safety and speed.
    
    # dM/df = 2 * IFFT(FFT(grad_m) * FFT(f_ext)) ? No, that's convolution.
    # We need the specific sum.
    
    # Let's use the explicit formula with FFT.
    # dM/df[i] = sum_k grad_m[k] * (f[k-i] + f[i-k])
    # Let g[k] = grad_m[k].
    # Term 1: sum_k g[k] f[k-i]. This is (g * f)[i] in convolution terms?
    # (g * f)[n] = sum_k g[k] f[n-k].
    # So sum_k g[k] f[k-i] = sum_k g[k] f[-(i-k)] = (g * f_rev)[i]?
    
    # Let's stick to a simpler, potentially slower but correct method for N=600.
    # dM/df[i] = sum_{k} grad_m[k] * (f[k-i] + f[i-k])
    # We can compute this using np.convolve.
    # Let's define h[k] = grad_m[k].
    # c[i] = sum_k h[k] f[k-i]
    # This is the cross-correlation of h and f.
    # np.correlate(h, f, mode='full')
    
    corr = np.correlate(grad_m, f, mode='full')
    # The result has length 2N-1 + N - 1 = 3N-2.
    # The center is at index N-1.
    # We need the values for i=0 to N-1.
    # The index in corr corresponding to lag i is (N-1) + i.
    # Let's verify.
    # np.correlate(a, v, mode='full') computes sum_k a[k] * v[k-i] ?
    # Documentation: c_k = sum_n a_{n+k} * conj(v_n)
    # So c_k = sum_n a[n+k] v[n]
    # We want sum_k grad_m[k] f[k-i].
    # Let n = k-i => k = n+i.
    # sum_n grad_m[n+i] f[n]
    # This matches c_i = sum_n a[n+i] v[n] with a=grad_m, v=f.
    # So corr[i] gives the value for lag i.
    # The 'full' output has length len(a) + len(v) - 1.
    # The lags range from -(len(v)-1) to len(a)-1.
    # So the index for lag i is i + (len(v)-1).
    # len(v) = N. So index is i + N - 1.
    
    term1 = corr[N-1 : 2*N-1]
    
    # The second term is sum_k grad_m[k] f[i-k]
    # Let n = i-k => k = i-n.
    # sum_n grad_m[i-n] f[n]
    # This is the correlation of f and grad_m?
    # c_k = sum_n v[n+k] a[n] ? No.
    # Let's use the symmetry.
    # sum_k grad_m[k] f[i-k]
    # Let j = i-k => k = i-j.
    # sum_j grad_m[i-j] f[j]
    # This is the convolution of grad_m and f at index i.
    # conv[i] = sum_j grad_m[j] f[i-j]
    # So this term is convolve(grad_m, f, mode='same')[i] ?
    # np.convolve(a, v, 'same') returns the central part of the full convolution.
    # The full convolution has length 2N-1 + N - 1 = 3N-2.
    # The central N points are from index N-1 to 2N-2.
    
    conv_gf = np.convolve(grad_m, f, mode='full')
    term2 = conv_gf[N-1 : 2*N-1]
    
    dM_df = term1 + term2
    
    M = smooth_max(conv, beta)
    
    # Gradient of J = M / s^2
    # dJ/df = (1/s^2) dM_df - (2 M / s^3) * 1 (since ds/df = 1 for all i)
    grad_J = dM_df / (s * s) - (2 * M / (s * s * s)) * np.ones(N)
    
    return M / (s * s), grad_J

def fista(f0, beta, max_iter, time_limit):
    """
    FISTA optimization for minimizing J(f) subject to f >= 0.
    """
    f = f0.copy()
    y = f0.copy()
    t = 1.0
    start_time = time.time()
    
    # Estimate Lipschitz constant L for the gradient
    # This is tricky. We can use a backtracking line search or a fixed step.
    # For now, let's use a fixed step size that is small enough.
    # The gradient involves convolutions, so the Lipschitz constant is related to the spectral norm of the convolution operator.
    # A safe step size is 1 / (2 * max_grad_norm) or similar.
    # Let's start with a small step and adapt if necessary.
    
    step = 0.01
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Gradient step
        obj, grad = objective_and_grad(y, beta)
        if np.isinf(obj):
            break
            
        y_new = y - step * grad
        
        # Projection onto f >= 0
        y_new = np.maximum(y_new, 0.0)
        
        # FISTA update
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        f_new = y_new + ((t - 1) / t_new) * (y_new - f)
        f_new = np.maximum(f_new, 0.0)
        
        f = f_new
        y = f_new
        t = t_new
        
    return f

def construct():
    start_time = time.time()
    
    # Initial guess: constant function
    f0 = np.ones(N)
    
    # Try different betas
    best_f = f0
    best_score = compute_score(f0)
    
    betas = [5.0, 10.0, 20.0, 50.0]
    
    for beta in betas:
        if time.time() - start_time > TIME_LIMIT - 20:
            break
            
        # Run FISTA
        f_res = fista(f0, beta, max_iter=1000, time_limit=TIME_LIMIT - 10)
        
        # Normalize to sum = N for consistency
        s = np.sum(f_res)
        if s > 1e-15:
            f_res = f_res * (N / s)
            
        score = compute_score(f_res)
        if score < best_score:
            best_score = score
            best_f = f_res.copy()
    
    # Final local refinement: simple gradient descent with projection
    if time.time() - start_time < TIME_LIMIT - 5:
        f_curr = best_f.copy()
        step = 0.01
        for i in range(500):
            if time.time() - start_time > TIME_LIMIT - 2:
                break
            obj, grad = objective_and_grad(f_curr, 50.0)
            if np.isinf(obj):
                break
            f_new = f_curr - step * grad
            f_new = np.maximum(f_new, 0.0)
            s_new = np.sum(f_new)
            if s_new < 1e-15:
                break
            f_new = f_new * (N / s_new)
            score_new = compute_score(f_new)
            if score_new < best_score:
                best_score = score_new
                best_f = f_new.copy()
                f_curr = f_new
            else:
                step *= 0.9
    
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
