<<<<<<< SEARCH
def _smooth_max_grad(h, beta):
    """
    Gradient of the smooth maximum objective (LogSumExp).
    Returns (value, gradient)
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    # Numerical stability: clip beta*(cs-m) to avoid overflow
    shifted = beta * (cs - m)
    shifted = np.clip(shifted, -500, 500)
    exp_terms = np.exp(shifted)
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights for the soft max
    
    val = m + np.log(sum_exp) / beta
    
    grad = np.zeros(n)
    
    # C_k = 1 - (2/n) * r_k
    # r_k = sum_{i=k}^{n-1} h[i]*h[i-k]
    # dC_k/dh_j = -(2/n) * ( h_{j-k} (if j>=k) + h_{j+k} (if j+k<n) )
    
    for k in range(n):
        if w[k] == 0.0:
            continue
        factor = -(2.0 / n) * w[k]
        
        if k > 0:
            grad[k:] += factor * h[:n-k]
        if k < n:
            grad[:n-k] += factor * h[k:]
            
    return val, grad
=======
def _smooth_max_grad(h, beta):
    """
    Gradient of the smooth maximum objective (LogSumExp).
    Returns (value, gradient)
    Vectorized gradient computation.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    shifted = beta * (cs - m)
    shifted = np.clip(shifted, -500, 500)
    exp_terms = np.exp(shifted)
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    val = m + np.log(sum_exp) / beta
    
    # Vectorized gradient: grad_j = -(2/n) * sum_k w_k * (h_{j-k} + h_{j+k})
    # where terms are included only when indices are valid
    grad = np.zeros(n)
    
    # For each k, the contribution to grad is:
    #   grad[k:] += -(2/n)*w[k] * h[:n-k]   (from h_{j-k} when j>=k)
    #   grad[:n-k] += -(2/n)*w[k] * h[k:]   (from h_{j+k} when j+k<n)
    # We can compute this via convolution-like operations.
    
    # Build the full autocorrelation structure more efficiently:
    # grad_j = -(2/n) * sum_k w_k * (h_{j-k} * 1_{j>=k} + h_{j+k} * 1_{j+k<n})
    
    # Use the fact that sum_k w_k * h_{j-k} is a weighted correlation.
    # Let's compute it via FFT for speed.
    # Actually, since n=100, the loop is fine but let's vectorize with numpy.
    
    # For each j, grad_j = -(2/n) * [ sum_{k<=j} w_k * h_{j-k} + sum_{k < n-j} w_k * h_{j+k} ]
    
    # Compute first sum: for each j, sum_{k=0}^{j} w_k * h_{j-k}
    # This is the convolution of w and h, evaluated at positions 0..n-1
    # conv_w_h[j] = sum_{k=0}^{min(j, n-1)} w[k] * h[j-k]
    
    # Compute second sum: for each j, sum_{k=0}^{n-1-j} w_k * h_{j+k}
    # This is a cross-correlation-like term.
    
    # For n=100, let's just do it with numpy broadcasting for clarity and correctness.
    # But that's O(n^2). Instead, use FFT-based convolution.
    
    # First term: conv(w, h) at lags 0..n-1
    n_fft = 2 * n
    w_pad = np.zeros(n_fft)
    w_pad[:n] = w
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    W = np.fft.rfft(w_pad, n=n_fft)
    H = np.fft.rfft(h_pad, n=n_fft)
    conv_wh = np.fft.irfft(W * H, n=n_fft)
    # conv_wh[j] = sum_{k=0}^{n-1} w[k] * h[j-k] (circular, but we want linear)
    # For linear conv, conv_wh[j] for j in 0..n-1 gives sum_{k=max(0,j-n+1)}^{min(j, n-1)} w[k]*h[j-k]
    # Since both have length n, for j in 0..n-1: sum_{k=0}^{min(j, n-1)} w[k]*h[j-k]
    # This is exactly what we want for the first term.
    first_sum = conv_wh[:n]
    
    # Second term: for each j, sum_{k=0}^{n-1-j} w[k] * h[j+k]
    # Let's define g[k] = w[k] for k=0..n-1
    # second_sum[j] = sum_{k=0}^{n-1-j} w[k] * h[j+k]
    # This is like a cross-correlation. Let's compute it differently.
    # second_sum[j] = sum_{k=0}^{n-1} w[k] * h[j+k] * 1_{j+k < n}
    # = sum_{k=0}^{n-1-j} w[k] * h[j+k]
    
    # We can compute this by reversing h and doing convolution.
    h_rev = h[::-1]
    h_rev_pad = np.zeros(n_fft)
    h_rev_pad[:n] = h_rev
    H_rev = np.fft.rfft(h_rev_pad, n=n_fft)
    conv_wh_rev = np.fft.irfft(W * H_rev, n=n_fft)
    # conv_wh_rev[j] = sum_{k} w[k] * h_rev[j-k] = sum_{k} w[k] * h[n-1-(j-k)]
    # = sum_{k} w[k] * h[n-1-j+k]
    # We want sum_{k=0}^{n-1-j} w[k] * h[j+k]
    # Let m = n-1-j, then we want sum_{k=0}^{m} w[k] * h[n-1-m+k] = sum_{k=0}^{m} w[k] * h_rev[m-k]
    # This is conv(w, h_rev)[m] where m = n-1-j
    # So second_sum[j] = conv_wh_rev[n-1-j]
    second_sum = conv_wh_rev[n-1::-1]
    # But we need to ensure we only sum k from 0 to n-1-j. The linear conv handles this.
    
    grad = -(2.0 / n) * (first_sum + second_sum)
    
    return val, grad
>>>>>>> REPLACE

