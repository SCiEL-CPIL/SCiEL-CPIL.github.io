"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autoconv_fft(h):
    """Compute full autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    # Use next power of 2 for FFT efficiency
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    conv = np.fft.irfft(H * H, n=fft_size)[:size]
    return conv

def _soft_max_obj_and_grad(x, beta=100.0):
    """
    Compute soft-max objective and gradient.
    Objective: 2*N*softmax(conv) / (sum(h))^2
    where softmax(conv) = logsumexp(beta*conv)/beta
    """
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(h)
    
    conv = _autoconv_fft(h)
    m = np.max(conv)
    
    # Compute logsumexp with numerical stability
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    log_sum_exp = m + np.log(np.sum(exp_shifted)) / beta
    
    # Value: 2*N*log_sum_exp / s^2
    val = 2.0 * N * log_sum_exp / (s * s)
    
    # Gradient computation
    # d(log_sum_exp)/d(conv_i) = exp_shifted_i / sum(exp_shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    
    # d(conv_i)/d(h_j) for full convolution:
    # conv[k] = sum_j h[j] * h[k-j] for valid k
    # d(conv[k])/d(h[j]) = h[k-j] + h[k-j] if k-j != j, else 2*h[j]
    # More carefully: conv[k] = sum_{j} h[j] * h[k-j] where indices are valid
    # d(conv[k])/d(h[j]) = h[k-j] + (1 if j == k-j else 0) * h[j]
    # Actually: d/dh_j [sum_l h[l] h[k-l]] = h[k-j] + sum_l [delta_{l,j} h[k-l] + h[l] delta_{k-l,j}]
    # = h[k-j] + h[k-j] = 2*h[k-j] if k-j != j, else h[k-j] + h[j] = 2*h[j]
    # Wait, let me be more careful.
    # conv[k] = sum_{l=0}^{N-1} h[l] * h[k-l] where h[i]=0 for i outside [0,N-1]
    # d(conv[k])/d(h[j]) = h[k-j] + h[k-j] = 2*h[k-j] (since h[k-j] appears twice: once as h[l] with l=j, once as h[k-l] with k-l=j i.e. l=k-j)
    # But if k-j == j, then it's h[j] + h[j] = 2*h[j], same formula.
    # So d(conv[k])/d(h[j]) = 2*h[k-j] for all valid k,j
    
    # Gradient of log_sum_exp w.r.t. h:
    # d(log_sum_exp)/d(h[j]) = sum_k weights[k] * d(conv[k])/d(h[j]) = 2 * sum_k weights[k] * h[k-j]
    # This is 2 * (weights * h) convolved appropriately
    
    # Let's compute: grad_h = 2 * sum_k weights[k] * h[k-j] for each j
    # This is 2 * (weights convolved with h) but with reversed indices
    # Actually: sum_k weights[k] * h[k-j] = (weights * flip(h))[j] where flip means reverse
    # Let's compute via FFT or direct convolution
    
    # weights has length 2*N-1, h has length N
    # For j in [0, N-1]: sum_{k=0}^{2N-2} weights[k] * h[k-j] where h[i]=0 if i<0 or i>=N
    # = sum_{k=j}^{j+N-1} weights[k] * h[k-j]
    # = sum_{i=0}^{N-1} weights[j+i] * h[i]
    
    # This is a sliding dot product. Let's compute efficiently.
    # Create an array where we place h and compute weighted sums
    # grad_logsumexp[j] = sum_{i=0}^{N-1} weights[j+i] * h[i]
    
    # Use FFT for efficiency
    # We want to compute for each j: sum_i h[i] * weights[j+i]
    # This is like a cross-correlation
    
    # Let's just do it directly since N=400 is small
    # grad_logsumexp = np.zeros(N)
    # for j in range(N):
    #     grad_logsumexp[j] = np.dot(h, weights[j:j+N])
    
    # Vectorized version:
    # Create a matrix where row j is weights[j:j+N]
    # Then grad_logsumexp = matrix @ h
    # But this is O(N^2) which is fine for N=400
    
    # More efficient: use FFT
    # sum_i h[i] * weights[j+i] = cross-correlation of h and weights at lag j
    # cross_corr[j] = sum_i h[i] * weights[j+i]
    
    # Use FFT-based cross-correlation
    # cross_corr = np.correlate(weights, h, mode='valid')  # This gives sum_i weights[i] * h[i+j]
    # We want sum_i h[i] * weights[j+i] = sum_i h[i] * weights[j+i]
    # Let u = j+i, then i = u-j, sum_u h[u-j] * weights[u] = cross_corr of h and weights
    
    # Actually, let's just compute it directly for clarity and correctness
    # N=400, so N^2=160000 operations, which is fast
    
    # Create the sliding window matrix
    # M[j, i] = weights[j+i] for j in [0,N-1], i in [0,N-1]
    # grad_logsumexp = M @ h
    
    # Efficient computation using FFT:
    # We want c[j] = sum_{i=0}^{N-1} h[i] * w[j+i] for j=0,...,N-1
    # This is the cross-correlation of h and w at lags 0 to N-1
    # np.correlate(w, h, 'valid') gives sum_i w[i] * h[i+j] which is different
    
    # Let's flip h: h_flip[i] = h[N-1-i]
    # Then sum_i h[i] * w[j+i] = sum_i h_flip[N-1-i] * w[j+i]
    # Let k = N-1-i, i = N-1-k
    # = sum_k h_flip[k] * w[j+N-1-k]
    # This is convolution of h_flip and w at position j+N-1
    
    # Simpler: just use direct computation since N is small
    # But let's try FFT approach properly
    
    # Method: Pad h to length 2N, compute FFT, multiply by FFT of weights (padded), inverse FFT
    # Actually, the cleanest is:
    # c[j] = sum_{i=0}^{N-1} h[i] * w[j+i]
    # This is the (j)-th element of the cross-correlation of w and h where we slide h over w
    
    # Let's use scipy.signal or just do it with FFT
    # Pad both to length >= 2N-1
    fft_len = 1
    while fft_len < 2 * N - 1:
        fft_len *= 2
    
    H_fft = np.fft.rfft(h, n=fft_len)
    W_fft = np.fft.rfft(w, n=fft_len)
    
    # Cross-correlation: c[j] = sum_i h[i] * w[j+i]
    # This equals (h * w_rev)[j + N - 1] where * is convolution and w_rev is reversed
    # Or: c = np.correlate(w, h, mode='full')[N-1:]  # Let me verify
    
    # Let's just compute directly for correctness, N=400 is fine
    # grad_logsumexp = np.array([np.dot(h, w[j:j+N]) for j in range(N)])
    
    # Actually, let's use the FFT method properly:
    # We want c[j] = sum_{i=0}^{N-1} h[i] * w[j+i] for j=0,...,N-1
    # Define H[i] = h[i] for i in [0,N-1], 0 otherwise
    # Define W[k] = w[k] for k in [0,2N-2], 0 otherwise
    # c[j] = sum_i H[i] * W[j+i]
    # This is the cross-correlation <H, W_shifted_by_j>
    # = (H * W_flipped)[j + N - 1] where W_flipped[k] = W[2N-2-k]
    
    # Let's compute via FFT:
    # H_fft = rfft(H, fft_len)
    # W_flip_fft = rfft(W_flipped, fft_len)
    # conv = irfft(H_fft * W_flip_fft)
    # c = conv[N-1 : N-1+N]
    
    # But this is getting complex. For N=400, direct computation is fast enough.
    # Let's benchmark: 400 dot products of length 400 = 160,000 multiplies, which is < 1ms
    
    # Direct computation:
    # grad_logsumexp = np.zeros(N)
    # for j in range(N):
    #     grad_logsumexp[j] = np.dot(h, w[j:j+N])
    
    # Vectorized using stride tricks:
    from numpy.lib.stride_tricks import as_strided
    # Create a view of w with shape (N, N) where row j is w[j:j+N]
    # w has shape (2N-1,)
    # Row j: w[j : j+N]
    # strides: (w.itemsize, w.itemsize)
    # shape: (N, N)
    
    w_sliding = as_strided(w, shape=(N, N), strides=(w.strides[0], w.strides[0]))
    grad_logsumexp = w_sliding @ h
    
    # Now, d(val)/d(h[j]) = 2*N/s^2 * d(log_sum_exp)/d(h[j]) + 2*N*log_sum_exp * d(1/s^2)/d(h[j])
    # d(1/s^2)/d(h[j]) = -2/s^3
    # So: d(val)/d(h[j]) = 2*N/s^2 * grad_logsumexp[j] + 2*N*log_sum_exp * (-2/s^3)
    # = 2*N/s^2 * (grad_logsumexp[j] - 2*log_sum_exp/s)
    
    grad_val = (2.0 * N / (s * s)) * (grad_logsumexp - 2.0 * log_sum_exp / s)
    
    return val, grad_val

def _objective_soft_max(x, beta=100.0):
    """Objective function for scipy minimize."""
    val, _ = _soft_max_obj_and_grad(x, beta)
    return val

def _objective_soft_max_grad(x, beta=100.0):
    """Objective and gradient for scipy minimize."""
    return _soft_max_obj_and_grad(x, beta)

def _objective_hard(x):
    """Hard objective for final evaluation."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = _autoconv_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_h = np.ones(N)
    best_val = _objective_hard(best_h)
    
    rng = np.random.default_rng(42)
    
    # List of initializations
    initializations = []
    
    # 1. Random normal with small mean
    for _ in range(15):
        h0 = rng.standard_normal(N)
        if abs(np.sum(h0)) < 1e-3:
            h0[0] += 1.0
        initializations.append(h0)
        
    # 2. Barker-like / Low autocorrelation sequences
    for _ in range(15):
        h0 = rng.choice([-1, 1], size=N).astype(float)
        h0 += 0.1 * rng.standard_normal()
        initializations.append(h0)
        
    # 3. Sine waves with random phase
    for freq in [1, 2, 3, 5, 7, 10, 15, 20]:
        t = np.linspace(0, 2*np.pi*freq, N, endpoint=False)
        h0 = np.sin(t + rng.random()*2*np.pi)
        h0 += 0.5
        initializations.append(h0)

    # 4. Polynomial perturbations of 1s
    for _ in range(10):
        h0 = np.ones(N)
        h0 += 0.1 * rng.standard_normal(N)
        initializations.append(h0)
    
    # 5. Low-frequency dominant signals
    for _ in range(10):
        t = np.linspace(0, 2*np.pi, N, endpoint=False)
        h0 = 1.0 + 0.3*np.sin(t + rng.random()*2*np.pi) + 0.2*np.sin(2*t + rng.random()*2*np.pi)
        initializations.append(h0)

    # Optimization parameters - use multiple betas
    betas = [30.0, 50.0, 100.0, 200.0]
    
    for init_idx, h0 in enumerate(initializations):
        if time.time() > deadline - 15.0:
            break
        
        h_curr = h0.copy()
        
        for beta in betas:
            if time.time() > deadline - 15.0:
                break
            
            try:
                res = minimize(_objective_soft_max, h_curr, 
                              jac=lambda x: _soft_max_obj_and_grad(x, beta)[1],
                              method='L-BFGS-B',
                              args=(beta,),
                              options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50})
                h_new = res.x
            except Exception:
                h_new = h_curr
            
            val_new = _objective_hard(h_new)
            
            if val_new < best_val:
                best_val = val_new
                best_h = h_new.copy()
                h_curr = h_new.copy()
            else:
                val_curr = _objective_hard(h_curr)
                if val_new < val_curr:
                    h_curr = h_new.copy()

    # Final polish with high beta on the best solution
    if time.time() < deadline - 5.0:
        for beta in [200.0, 500.0, 1000.0]:
            if time.time() > deadline - 5.0:
                break
            try:
                res = minimize(_objective_soft_max, best_h,
                              jac=lambda x: _soft_max_obj_and_grad(x, beta)[1],
                              method='L-BFGS-B',
                              args=(beta,),
                              options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100})
                val = _objective_hard(res.x)
                if val < best_val:
                    best_val = val
                    best_h = res.x.copy()
            except Exception:
                pass

    # Ensure sum is not zero
    if abs(np.sum(best_h)) < 1e-10:
        best_h = np.ones(N)
        best_val = _objective_hard(best_h)
        
    return best_h
# EVOLVE-BLOCK-END
