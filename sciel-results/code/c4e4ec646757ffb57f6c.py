import numpy as np
import time
import sys

N = 400

def _autoconv_lin(x):
    """Compute linear autoconvolution using FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    return np.fft.irfft(fx * fx, size)[:2 * n - 1]

def _metric_std(x):
    """Standard metric: max |c_k| / S^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    return float(np.max(np.abs(c))) / (s * s)

def _soft_max_abs(c, beta):
    """Compute soft-max of absolute values and the weights."""
    abs_c = np.abs(c)
    m = np.max(abs_c)
    if m < 1e-15:
        return 0.0, np.zeros_like(c)
    # Numerical stability
    exps = np.exp(beta * (abs_c - m))
    sum_exps = np.sum(exps)
    weights = exps / sum_exps
    val = m + np.log(sum_exps) / beta
    return val, weights

def _gradient_descent_minimize(x0, deadline, beta_start=10.0, beta_end=200.0, max_iters=500):
    """
    Minimize max|c_k|/S^2 using gradient descent on the soft-max surrogate.
    x is constrained to have sum 1.
    """
    x = x0.copy().astype(float)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return x
    x = x / s  # Normalize sum to 1
    
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
        
    best_x = x.copy()
    best_val = _metric_std(x)
    
    # Adam optimizer parameters
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    t = 0
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Anneal beta
        progress = it / max_iters
        beta = beta_start + (beta_end - beta_start) * progress
        
        # Compute autoconvolution
        c = _autoconv_lin(x)
        
        # Compute soft-max weights
        val, weights = _soft_max_abs(c, beta)
        
        # Gradient of soft-max w.r.t. x
        # grad_x[j] = sum_k weights[k] * sign(c[k]) * (x[j-k] + x[j+k])
        # Let g[k] = weights[k] * sign(c[k])
        g = weights * np.sign(c)
        
        # The term sum_k g[k] x[j-k] is the linear convolution g * x
        # The term sum_k g[k] x[j+k] is the linear cross-correlation or reversed conv
        # Note: c is linear autoconv of length 2n-1.
        # We need to align g with x for convolution.
        
        # Efficient computation using FFT
        # conv1[j] = sum_k g[k] x[j-k]
        # conv2[j] = sum_k g[k] x[j+k]
        
        # Let's construct the full gradient in frequency domain or use direct conv if n is small enough?
        # n=400, 2n-1=799. FFT is fast.
        
        # We need to be careful with the indices.
        # c[k] = sum_i x[i] x[i-k] for k in 0..2n-2.
        # d(c[k])/dx[j] = x[j-k] (if j>=k) + x[j+k] (if j+k < n and k>0)
        
        # Let's compute grad_x directly using the weights.
        # grad_x[j] = sum_{k=0}^{2n-2} w_k * s_k * (x[j-k] + x[j+k])
        # where w_k is weight, s_k is sign.
        
        # This is equivalent to:
        # Term 1: sum_k g[k] x[j-k] -> Linear convolution of g and x.
        # Term 2: sum_k g[k] x[j+k] -> This is correlation-like.
        # Let h[k] = g[k].
        # Term 2[j] = sum_k h[k] x[j+k].
        # If we define x_rev[i] = x[n-1-i], this gets messy.
        
        # Simpler: Use the fact that the gradient of the quadratic form x^T A x is 2Ax.
        # Here the objective is not a simple quadratic form due to the max/soft-max.
        # But locally, it is linear in c, and c is quadratic in x.
        
        # Let's just compute the gradient via FFT of the "error" signal.
        # The gradient of sum_k w_k |c_k| with respect to x is related to the inverse FFT of w * sign(c) * F(x)?
        # No, c = IFFT(|F(x)|^2).
        # Let Y = F(x). C = IFFT(Y^2).
        # d L / d Y = 2 Y * (FFT(d L / d C) / N_fft) ?
        # Let's stick to time domain convolution for clarity and correctness, n=400 is small.
        
        grad_x = np.zeros(n)
        
        # Pre-compute g
        # g is length 2n-1
        
        # Loop over k is O(n^2) = 160,000 operations. This is very fast in Python/Numpy if vectorized?
        # No, Python loop is slow. We must vectorize.
        
        # Vectorized Convolution:
        # conv1 = np.convolve(g, x, mode='valid')? No, lengths don't match directly for 'full' logic.
        # We want sum_k g[k] x[j-k].
        # This is the cross-correlation or convolution.
        # Let's use FFT for the gradient calculation.
        
        # G = FFT(g, size)
        # X = FFT(x, size)
        # The gradient of the sum of weighted c_k is not simply G*X.
        # However, c = IFFT(X^2).
        # Let dL/dc = g (length 2n-1).
        # dL/dx = 2 * IFFT( FFT(dL/dc) * X ) ? 
        # Check:
        # L = sum_k g_k c_k
        # c_k = sum_i x_i x_{i-k}
        # dL/dx_j = sum_k g_k (x_{j-k} + x_{j+k})
        # This is exactly 2 * (g * x)_j if g is symmetric?
        # g_k = w_k sign(c_k). c_k is symmetric (autoconv). w_k depends on |c_k|, so w_k is symmetric.
        # sign(c_k) is symmetric if c_k is symmetric? Yes, c_k = c_{2n-2-k}? No, linear autoconv is symmetric around n-1?
        # c_k = sum x_i x_{i-k}. c_{2n-2-k} = sum x_i x_{i-(2n-2-k)}.
        # Actually, linear autoconvolution is symmetric: c_k = c_{2n-2-k} is NOT true.
        # c_k = c_{-k} for circular. For linear, c_k is defined for k=0..2n-2.
        # c_k = sum_{i=max(0,k)}^{n-1} x_i x_{i-k}.
        # c_{2n-2-k} = sum_{i=max(0, 2n-2-k)}^{n-1} x_i x_{i-(2n-2-k)}.
        # Let j = i - (2n-2-k). Then i = j + 2n-2-k.
        # Range of i: 2n-2-k to n-1.
        # Range of j: 0 to n-1-(2n-2-k) = k-n+1.
        # This is getting complicated.
        
        # Let's use the FFT property:
        # If c = IFFT(X^2), then dL/dx = 2 * IFFT(X * FFT(g_padded))
        # where g_padded is g placed in the FFT domain correctly.
        # Since c is linear, we zero-padded to size.
        # The gradient in frequency domain is 2 * X * G, where G is the DFT of the gradient of the loss w.r.t c.
        # But G must be the DFT of the sequence g_k extended appropriately?
        # Actually, if L = sum g_k c_k, and c = IFFT(X^2), then
        # dL/dX = 2 X * FFT(g) / size (scaling depends on FFT def).
        # Then dL/dx = IFFT(dL/dX).
        
        # Let's test this hypothesis.
        # g is length 2n-1.
        # Pad g to size.
        # G = np.fft.rfft(g, size)
        # X = np.fft.rfft(x, size)
        # dL_dX = 2 * X * G
        # grad_x = np.fft.irfft(dL_dX, size)[:n]
        
        # This should work.
        
        g_padded = np.zeros(size)
        g_padded[:len(g)] = g
        G = np.fft.rfft(g_padded, size)
        X = np.fft.rfft(x, size)
        dL_dX = 2 * X * G
        grad_x = np.fft.irfft(dL_dX, size)[:n]
        
        # Update with Adam
        t += 1
        m = beta1 * m + (1 - beta1) * grad_x
        v = beta2 * v + (1 - beta2) * (grad_x ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        x_new = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # Normalize sum to 1
        s_new = np.sum(x_new)
        if abs(s_new) > 1e-12:
            x_new = x_new / s_new
            
        val = _metric_std(x_new)
        if val < best_val:
            best_val = val
            best_x = x_new.copy()
            
        x = x_new
        
    return best_x

def construct():
    deadline = time.time() + 135.0
    
    best_x = np.ones(N)
    best_val = _metric_std(best_x)
    
    rng = np.random.default_rng(42)
    
    # Multiple restarts with random initializations
    for trial in range(10):
        if time.time() > deadline - 5:
            break
            
        # Initialize with random signs or Gaussians
        # Random signs often work well for low autocorrelation
        if trial % 2 == 0:
            x_init = rng.choice([-1.0, 1.0], size=N)
        else:
            x_init = rng.normal(0, 1, size=N)
            
        # Ensure sum is not zero
        s = np.sum(x_init)
        if abs(s) < 1e-6:
            x_init += 1.0
            s = np.sum(x_init)
        x_init = x_init / s
        
        # Run gradient descent
        x_opt = _gradient_descent_minimize(x_init, deadline=deadline - 2)
        val = _metric_std(x_opt)
        
        if val < best_val:
            best_val = val
            best_x = x_opt.copy()
            
    return best_x
