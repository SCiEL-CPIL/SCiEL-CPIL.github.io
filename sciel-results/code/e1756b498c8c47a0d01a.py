import numpy as np
import time
from scipy.optimize import minimize

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

def _soft_max_abs_offdiag(x, temp=1.0):
    """Soft-max approximation of max |c_k| for k != 0."""
    c = _autoconv(x)
    if len(c) <= 1:
        return 0.0
    abs_c = np.abs(c[1:])
    if len(abs_c) == 0:
        return 0.0
    max_c = np.max(abs_c)
    if max_c < 1e-15:
        return 0.0
    exp_vals = np.exp((abs_c - max_c) / temp)
    soft_max = max_c + temp * np.log(np.sum(exp_vals))
    return soft_max

def _gradient_step(x, temp=1.0):
    """
    Compute gradient of soft_max_abs_offdiag(x) w.r.t x.
    x is assumed to have sum 1.
    """
    c = _autoconv(x)
    if len(c) <= 1:
        return np.zeros(N)
    
    abs_c = np.abs(c[1:])
    max_c = np.max(abs_c)
    if max_c < 1e-15:
        return np.zeros(N)
        
    exp_vals = np.exp((abs_c - max_c) / temp)
    sum_exp = np.sum(exp_vals)
    grad_abs_c = exp_vals / sum_exp
    
    # Gradient of |c_k| w.r.t c_k is sign(c_k)
    # Gradient of c_k w.r.t x
    # c_k = sum_i x_i x_{i+k}
    # dc_k/dx_j = x_{j+k} + x_{j-k} (with boundary conditions)
    # This is equivalent to the convolution of x with a kernel that picks up the relevant terms.
    # Specifically, grad_x = convolve(x, grad_c_kernel)
    
    # Let's compute the gradient via FFT for efficiency
    # c = ifft(fft(x)^2)
    # d c / d x = ifft(2 * fft(x) * grad_c) ... wait, chain rule.
    # c_k = x^T A_k x
    # grad c_k = 2 A_k x (since A_k is symmetric for autoconv? No, A_k is shift matrix)
    # Actually, c_k = sum x_i x_{i+k}.
    # dc_k/dx_j = x_{j+k} + x_{j-k} (if in bounds).
    # This is the same as (x shifted by k) + (x shifted by -k).
    
    # We have weights w_k = grad_abs_c[k-1] * sign(c[k]) for k=1..N-1.
    # Note c[-k] = c[k] for real x? No, linear autoconv is symmetric: c[k] = c[-k].
    # So we sum contributions from all k.
    
    grad_x = np.zeros(N)
    for k in range(1, N):
        w = grad_abs_c[k-1] * np.sign(c[k])
        # Contribution to grad_x from c_k
        # d c_k / d x = 2 * (x shifted by k) + (x shifted by -k) ?
        # Let's check: c_k = sum_i x_i x_{i+k}.
        # d/dx_j = x_{j+k} + x_{j-k} (terms where j=i or j=i+k).
        # So grad_x[j] += w * (x[j+k] + x[j-k])
        
        # Vectorized:
        # shift_plus = np.roll(x, -k) # x[i+k]
        # shift_minus = np.roll(x, k) # x[i-k]
        # But linear autoconv doesn't wrap.
        
    # Efficient FFT approach:
    # c = ifft( |X|^2 )
    # Let G = fft( grad_c ) where grad_c is the vector of weights for c_k.
    # We want grad_x such that grad_x = 2 * ifft( X * conj(G) ) ?
    # Let's derive.
    # c_k = sum_i x_i x_{i+k}
    # X = fft(x)
    # C = |X|^2
    # d C / d X = 2 X
    # d c / d x = ifft( 2 X * dC/dc )
    # dC/dc is the gradient of the objective w.r.t c.
    # Objective J = soft_max(|c_off|).
    # dJ/dc_k = w_k for k != 0.
    # So dC/dc has w_k at index k and w_{-k} at index -k (mod N).
    # Since c is symmetric, w_k = w_{-k}.
    
    # Construct the gradient vector in frequency domain
    # c has length 2N-1.
    # We need to map c indices to FFT bins.
    # This is tricky with linear autoconv.
    
    # Alternative: Use the explicit sum for gradient, but it's O(N^2).
    # N=400, N^2=160,000. This is fast enough for Python if vectorized?
    # No, Python loop over N is slow.
    
    # Let's use the property:
    # grad_x = 2 * (x * w) where * is cross-correlation?
    # c_k = <x, shift_k(x)>
    # grad_x = sum_k w_k * 2 * shift_k(x)
    
    # shift_k(x) is x shifted by k.
    # We can compute this using FFT if we pad.
    
    # Let's just use the FFT relation carefully.
    # For circular autoconv, grad_x = 2 * ifft( X * fft(w) ).
    # For linear, we can embed in circular of size 2N.
    
    size = 2 * N
    fsize = 1
    while fsize < size:
        fsize *= 2
        
    x_padded = np.zeros(fsize)
    x_padded[:N] = x
    
    X_padded = np.fft.rfft(x_padded)
    
    # Construct w in time domain (padded)
    # w_k corresponds to c_k.
    # c_k is at index k in the linear conv result.
    # In the padded FFT, the autoconv is circular.
    # The linear autoconv c_k is the same as circular autoconv for k < N.
    
    w_padded = np.zeros(fsize)
    # c[1:] corresponds to k=1..N-1
    # Also c[-k] = c[k].
    # In circular conv of size fsize, the index for -k is fsize-k.
    # But our c is linear.
    # The gradient of linear autoconv w.r.t x is:
    # grad_x[j] = sum_k w_k (x[j+k] + x[j-k])
    
    # This is equivalent to:
    # grad_x = 2 * ifft( X_padded * fft(w_padded) ) ... ?
    # Let's test with small N.
    
    # Simpler: Just compute the gradient using the explicit formula but vectorized with FFT.
    # grad_x = 2 * ifft( X * conj( W ) ) where W is the FFT of the weight sequence.
    # The weight sequence for the derivative of c_k is effectively a kernel.
    
    # Let's use the time-domain formula with numpy broadcasting for N=400.
    # It might be slow in pure Python loop, but vectorized numpy is fast.
    # grad_x = np.zeros(N)
    # for k in range(1, N):
    #    w = grad_abs_c[k-1] * np.sign(c[k])
    #    # x shifted by k (linear)
    #    # x[j+k] for j=0..N-k-1
    #    # x[j-k] for j=k..N-1
    
    # This is O(N^2) operations, 160k. Numpy can do this in <1ms.
    # Let's do it.
    
    grad_x = np.zeros(N)
    for k in range(1, N):
        w = grad_abs_c[k-1] * np.sign(c[k])
        if w == 0: continue
        # x[j+k]
        grad_x[:-k] += w * x[k:]
        # x[j-k]
        grad_x[k:] += w * x[:-k]
        
    grad_x *= 2.0
    return grad_x

def _optimize_continuous(x0, max_iter=500, deadline=None):
    """Gradient descent on x with sum(x)=1 constraint."""
    x = x0.copy().astype(float)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return x
    x = x / s
    
    best_x = x.copy()
    best_val = _metric_offdiag(best_x)
    
    lr = 0.01
    temp = 1.0
    fsize = 1
    while fsize < 2 * N:
        fsize *= 2
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        grad = _gradient_step(x, temp)
        
        # Project gradient to sum=0 subspace
        grad -= np.mean(grad)
        
        x_new = x - lr * grad
        
        # Project to sum=1
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
        lr *= 0.99
        temp *= 0.995
        
    return best_x

def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    
    best_x = np.ones(N)
    best_val = _metric_offdiag(best_x)
    
    rng = np.random.default_rng(42)
    
    # Initialization 1: All ones
    candidates = [np.ones(N)]
    
    # Initialization 2: Flat spectrum (random phase)
    for _ in range(10):
        half_n = N // 2 + 1
        phase = rng.random(half_n) * 2 * np.pi
        mag = np.ones(half_n)
        # Ensure DC is positive real
        phase[0] = 0
        # Ensure Nyquist is real
        if N % 2 == 0:
            phase[-1] = 0
            
        X = mag * np.exp(1j * phase)
        x = np.fft.irfft(X, N)
        s = np.sum(x)
        if abs(s) > 1e-6:
            x = x / s
            candidates.append(x)
            
    # Optimization
    for x0 in candidates:
        if time.time() > deadline - 10:
            break
        
        try:
            x_opt = _optimize_continuous(x0, max_iter=2000, deadline=deadline - 5)
            val = _metric_offdiag(x_opt)
            if val < best_val:
                best_val = val
                best_x = x_opt.copy()
        except Exception:
            pass
            
    # Final polish with smaller temperature
    if time.time() < deadline - 5:
        try:
            # Use the best found so far
            x_opt = _optimize_continuous(best_x, max_iter=1000, deadline=deadline - 2)
            val = _metric_offdiag(x_opt)
            if val < best_val:
                best_val = val
                best_x = x_opt.copy()
        except Exception:
            pass

    return best_x
