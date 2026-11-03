import numpy as np
import time

N = 600

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin
    
    # Initialize z with a Gaussian-like profile in log-space
    # This helps avoid flat minima
    x = np.arange(N)
    center = N / 2.0
    sigma = N / 4.0
    z = -0.5 * ((x - center) / sigma) ** 2
    
    best_h = None
    best_score = np.inf
    
    # Softmax parameters
    # We will optimize z, and h = softmax(z)
    
    # Gradient Descent Parameters
    lr = 0.01
    p_power = 10  # Approximation of L-infinity norm
    
    max_iters = 5000
    
    for i in range(max_iters):
        if time.time() - start_time > timeout:
            break
            
        # Compute h from z
        # Subtract max for numerical stability
        z_max = np.max(z)
        exp_z = np.exp(z - z_max)
        h = exp_z / np.sum(exp_z)
        
        # Compute Autocorrelation
        # Use FFT for efficiency
        # Linear autocorrelation via FFT
        # Pad to 2*N
        n_fft = 2 * N
        H = np.fft.fft(h, n=n_fft)
        # Power spectrum
        P = H * np.conj(H)
        # Inverse FFT to get autocorrelation
        # The result is circular correlation of the zero-padded signal
        # which corresponds to linear correlation for lags -N+1 to N-1
        c = np.fft.ifft(P).real
        # The linear autocorrelation is in the middle of c
        # Specifically, lags 0 to N-1 are in c[N-1:N-1+N] ?
        # Let's verify index mapping.
        # np.fft.ifft returns values for lags 0, 1, ..., N_fft-1.
        # For linear correlation of length N signals padded to 2N:
        # Lag k is at index k if k >= 0.
        # Lag -k is at index 2N - k.
        # We want the max of all lags.
        # The peak is usually at lag 0 (index 0) if centered?
        # Wait, np.fft.ifft of |FFT|^2 gives the circular autocorrelation.
        # For zero-padded signals, the circular autocorrelation at lag 0 is the sum of squares.
        # The linear autocorrelation at lag 0 is also sum of squares.
        # The linear autocorrelation at lag k (k>0) is at index k.
        # The linear autocorrelation at lag -k is at index 2N - k.
        # So we just need to look at the whole c array?
        # No, the values for lags > N-1 or < -(N-1) are zero.
        # So we can just take the max of the entire c array?
        # Yes, because the "wrap-around" terms are zero due to padding.
        
        c_max = np.max(c)
        
        if c_max < best_score:
            best_score = c_max
            best_h = h.copy()
        
        # Compute Gradient
        # L = sum(c_k^p)
        # dL/dc_k = p * c_k^(p-1)
        # Avoid 0^(p-1) issues if p<1, but p=10 so it's fine.
        # Clip c to small positive to avoid numerical issues with gradient if c is 0
        c_safe = np.maximum(c, 1e-12)
        g_c = p_power * c_safe ** (p_power - 1)
        
        # dh/dz logic later. First compute dL/dh
        # dL/dh_j = sum_k dL/dc_k * dc_k/dh_j
        # dc_k/dh_j = h_{j+k} + h_{j-k}
        # This is 2 * (g_c * h)_j
        # Convolution of g_c and h
        # g_c has length 2N (from ifft)
        # h has length N
        # We need linear convolution.
        # Use FFT again.
        # Pad g_c and h to 3N? Or just use np.convolve?
        # N=600, 3N=1800. np.convolve is O(N*M) = 600*1800 ~ 1e6. Fast enough.
        # But we are in a loop of 5000. 5e9 ops. Might be slow in Python.
        # Use FFT.
        
        # g_c is defined on lags -N+1 to N-1 (effectively).
        # We want conv(g_c, h).
        # Let's construct the full linear convolution.
        # g_c_len = 2N
        # h_len = N
        # conv_len = 3N - 1
        # We need the part corresponding to j in [0, N-1].
        
        # Efficient way:
        # G = fft(g_c, n=3*N)
        # Hh = fft(h, n=3*N)
        # Conv = ifft(G * Hh)
        # dL/dh = 2 * Conv[? : ?]
        
        # Let's trace indices.
        # g_c[k] corresponds to lag k (shifted?).
        # In our c array from ifft(P):
        # c[0] is lag 0.
        # c[k] is lag k for k=1..N-1.
        # c[2N-k] is lag -k.
        # So g_c is aligned with lags.
        
        # We want sum_k g_c[k] * (h[j+k] + h[j-k])
        # = sum_k g_c[k] h[j+k] + sum_k g_c[k] h[j-k]
        # Let m = j+k => k = m-j. Sum_m g_c[m-j] h[m]
        # This is (h * g_c)[j] if g_c is symmetric?
        # g_c is symmetric because c is symmetric (real h).
        # So dL/dh = 2 * (h * g_c)[j]
        
        # Convolution h * g_c
        # h is length N. g_c is length 2N (but effectively non-zero in middle).
        # Let's just use np.convolve(h, g_c, mode='full')
        # This is O(N * 2N) = 2 N^2.
        # 2 * 600^2 = 720,000 operations.
        # 5000 iterations * 720,000 = 3.6e9 operations.
        # In Python/NumPy, this might take ~1-2 seconds per million ops?
        # NumPy vectorized convolve is fast.
        # Let's try to optimize with FFT if needed, but start with convolve for correctness.
        
        conv = np.convolve(h, g_c, mode='full')
        # We need the indices in conv corresponding to j=0..N-1.
        # conv[k] = sum_i h[i] g_c[k-i]
        # We want sum_k g_c[k] h[j+k] + ...
        # Actually, if g_c is symmetric, (h*g_c)[j] is the standard convolution.
        # The peak of the convolution of two length N and 2N signals is at index N-1 + N-1?
        # Let's just extract the middle N values.
        # The linear convolution of h (len N) and g_c (len 2N) has length 3N-1.
        # The "center" of the correlation is at index N-1 + N-1 = 2N-2?
        # Let's check:
        # If h is delta at 0, g_c is delta at 0, conv is delta at 0.
        # If h is delta at N-1, g_c is delta at N-1, conv is delta at 2N-2.
        # We want dL/dh_j for j=0..N-1.
        # The term h[j+k] suggests shifting h.
        
        # Let's use the gradient property directly:
        # grad_h = 2 * np.convolve(g_c, h, mode='full')
        # We need to align this with h's indices.
        # The correlation c_k = sum_i h_i h_{i+k}.
        # d/dh_j sum_k c_k^p = sum_k p c_k^{p-1} (h_{j+k} + h_{j-k})
        # = sum_k g_c_k h_{j+k} + sum_k g_c_k h_{j-k}
        # First term: sum_m g_c_{m-j} h_m. This is (h * g_c^rev)[j] ?
        # Since g_c is symmetric, g_c^rev = g_c.
        # So it is (h * g_c)[j] in the correlation sense?
        # Standard convolution (h*g_c)[j] = sum_i h_i g_c[j-i].
        # Let i = m. sum_m h_m g_c[j-m].
        # Our term is sum_m h_m g_c[m-j].
        # Since g_c is symmetric, g_c[m-j] = g_c[j-m].
        # So it is exactly (h * g_c)[j].
        
        # So grad_h = 2 * (h * g_c)[j]
        # We compute conv_full = np.convolve(h, g_c, mode='full')
        # Which indices of conv_full correspond to j=0..N-1?
        # conv_full[k] = sum_{i=0}^{N-1} h_i g_c[k-i]
        # We want j such that the "center" of g_c aligns with j?
        # g_c is defined on lags -N+1..N-1.
        # In our array g_c (from ifft), index 0 is lag 0.
        # So g_c[k] is 0 for k > N-1 and k < 2N-N+1?
        # Actually, g_c is non-zero for indices 0..N-1 and 2N-N+1..2N-1.
        # This is confusing.
        
        # Let's use a simpler, robust method:
        # Compute gradient numerically? No, too slow.
        
        # Let's use the FFT method for gradient which is cleaner.
        # dL/dh = 2 * ifft( fft(g_c) * fft(h) ) ?
        # No, that's circular convolution.
        
        # Let's stick to `np.convolve` and find the right slice.
        # g_c is length 2N.
        # h is length N.
        # conv = np.convolve(h, g_c, mode='full')
        # Length 3N-1.
        # We want the part where the overlap is "central".
        # For j=0, we sum g_c[k] h[k]. This is correlation at lag 0.
        # In conv, index k corresponds to sum h_i g_c[k-i].
        # If we set g_c to be centered at 0 in a larger array...
        
        # Alternative:
        # grad_h = 2 * np.correlate(h, g_c, mode='full')
        # np.correlate(h, g_c, 'full')[k] = sum_i h[i] g_c[i+k-N+1]?
        # Documentation: correlate(a, v) is sum a[i] v[i+k].
        # So correlate(h, g_c, 'full')[k] = sum_i h[i] g_c[i+k].
        # We want sum_k g_c[k] h[j+k].
        # Let i = j+k => k = i-j.
        # sum_i g_c[i-j] h[i].
        # This is correlate(h, g_c, 'full') at index j?
        # correlate(h, g_c, 'full')[j] = sum_i h[i] g_c[i+j].
        # This matches sum_i h[i] g_c[i+j].
        # We need sum_i h[i] g_c[i-j].
        # Since g_c is symmetric, g_c[i-j] = g_c[j-i] = g_c[-(i-j)]?
        # g_c is symmetric around 0.
        # So sum_i h[i] g_c[i-j] = sum_i h[i] g_c[j-i].
        # This is the cross-correlation of h and g_c at lag j?
        # Yes.
        
        # So grad_h = 2 * np.correlate(h, g_c, mode='full')[N-1 : 3*N-2 - (N-1) : 1]?
        # The length of 'full' correlate of N and 2N is N + 2N - 1 = 3N - 1.
        # The index for lag 0 is N-1?
        # For correlate(a, v) where len(a)=N, len(v)=M.
        # Output length N+M-1.
        # Lag 0 is at index M-1.
        # Here M=2N. So lag 0 is at index 2N-1.
        # We want lags 0 to N-1.
        # So indices 2N-1 to 2N-1 + N-1.
        
        corr = np.correlate(h, g_c, mode='full')
        grad_h = 2 * corr[2*N-1 : 3*N-2]
        
        # Now compute grad_z
        # h = softmax(z)
        # grad_z = h * (grad_h - h * sum(h * grad_h))
        
        mean_grad = np.sum(h * grad_h)
        grad_z = h * (grad_h - mean_grad)
        
        # Update z
        z -= lr * grad_z
        
    if best_h is None:
        return [1.0] * N
    return best_h.tolist()
