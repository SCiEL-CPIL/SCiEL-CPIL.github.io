import numpy as np
import time
from scipy.optimize import minimize

N = 100
TARGET = N / 2.0

def _linear_overlaps(h):
    """
    Compute linear overlaps O_k = sum_{i=0}^{N-k-1} h_i (1 - h_{i+k})
    for k = 0 to N-1.
    """
    O = np.zeros(N)
    for k in range(N):
        # h_i and h_{i+k}
        # sum h_i (1 - h_{i+k}) = sum h_i - sum h_i h_{i+k}
        # sum h_i over the valid range is not constant TARGET.
        # It's sum_{i=0}^{N-k-1} h_i.
        # This makes it harder.
        
        # Let's just compute directly for small N or use FFT with zero padding.
        pass
    
    # Use FFT with zero padding for linear correlation
    # c_k = sum h_i h_{i+k}
    # We need O_k = S_k - c_k, where S_k = sum_{i=0}^{N-k-1} h_i.
    
    # FFT method for linear correlation:
    # Zero pad h to length 2N.
    h_pad = np.zeros(2 * N)
    h_pad[:N] = h
    H = np.fft.fft(h_pad)
    # c_k is the first N elements of the IFFT of H * conj(H)?
    # No, linear autocorrelation r_k = sum h_i h_{i+k}.
    # This is the first N coefficients of the linear convolution of h and reversed h?
    # Or just use np.correlate.
    
    # np.correlate(h, h, mode='full') gives linear autocorrelation.
    # c[0] is sum h_i^2.
    # c[k] is sum h_i h_{i+k} for k>0.
    c_full = np.correlate(h, h, mode='full')
    # c_full is length 2N-1.
    # c_full[N-1 + k] corresponds to lag k?
    # np.correlate(a, v) is sum a[n+k] conj(v[n]).
    # For autocorrelation, a=v=h.
    # c_full[N-1] is lag 0.
    # c_full[N-1+k] is lag k.
    # c_full[N-1-k] is lag -k.
    
    c = c_full[N-1 : 2*N-1] # lags 0 to N-1
    
    # Now compute S_k = sum_{i=0}^{N-k-1} h_i
    # This is the prefix sum of h.
    prefix_sums = np.cumsum(h)
    # S_k = sum_{i=0}^{N-k-1} h_i = prefix_sums[N-k-1]
    # Note: for k=N-1, S = h[0].
    S = np.zeros(N)
    for k in range(N):
        S[k] = prefix_sums[N - k - 1]
        
    O = S - c
    return O

def _max_linear_overlap(h):
    return float(np.max(_linear_overlaps(h)))

def _soft_max_linear(h, t):
    O = _linear_overlaps(h)
    m = np.max(O)
    return m + t * np.log(np.mean(np.exp((O - m) / t)))

def _soft_max_linear_grad(h, t):
    O = _linear_overlaps(h)
    m = np.max(O)
    w = np.exp((O - m) / t)
    w /= w.sum()
    
    # Gradient of O_k w.r.t h_i
    # O_k = S_k - c_k
    # d S_k / d h_i = 1 if i <= N-k-1 else 0
    # d c_k / d h_i = h_{i+k} + h_{i-k} (with bounds)
    
    grad = np.zeros(N)
    for k in range(N):
        if w[k] < 1e-9: continue
        # d O_k / d h_i
        # Term 1: S_k
        # h_i contributes to S_k if i is in [0, N-k-1]
        # So d S_k / d h_i = 1 for i < N-k
        # Term 2: -c_k
        # c_k = sum_{j=0}^{N-k-1} h_j h_{j+k}
        # d c_k / d h_i = h_{i+k} (if i+k < N and i < N-k) + h_{i-k} (if i-k >= 0 and i-k < N-k)
        # Simplified:
        # For each i:
        #   if i < N-k:
        #       d S_k / d h_i = 1
        #       d c_k / d h_i = h_{i+k} + (h_{i-k} if i>=k else 0)
        #   else:
        #       d S_k / d h_i = 0
        #       d c_k / d h_i = 0 (since i is not in the sum range for c_k? 
        #       Wait, c_k sum is j=0 to N-k-1.
        #       If i > N-k-1, h_i is not in c_k.
        #       So d c_k / d h_i = 0.
        
        # This loop is O(N^2). N=100, so 10000 ops. Fast enough.
        for i in range(N):
            if i < N - k:
                dS = 1.0
                dc = 0.0
                if i + k < N:
                    dc += h[i + k]
                if i - k >= 0:
                    dc += h[i - k]
                grad[i] += w[k] * (dS - dc)
            # else grad[i] += 0
            
    return grad

def _enforce(h):
    h = np.clip(h, 0.0, 1.0)
    s = h.sum()
    if s > 0:
        h = h * (TARGET / s)
        h = np.clip(h, 0.0, 1.0)
    diff = TARGET - h.sum()
    if abs(diff) > 1e-14:
        if diff > 0:
            idx = np.where(h < 1.0 - 1e-12)[0]
        else:
            idx = np.where(h > 1e-12)[0]
        if idx.size:
            h[idx[0]] += diff
            h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    start = time.time()
    deadline = start + 130.0

    rng = np.random.default_rng(12345)
    
    best_h = None
    best_val = np.inf

    # Start with 0.5
    h0 = np.full(N, 0.5)
    
    # Try a few different starting points
    starts = [h0]
    # Tapered
    for a in [1.0, 2.0, 5.0]:
        x = np.linspace(-1, 1, N)
        h_tap = np.exp(-a * x**2)
        h_tap = h_tap / h_tap.sum() * TARGET
        starts.append(_enforce(h_tap))
        
    for h0 in starts:
        if time.time() > deadline - 5.0:
            break
            
        h = h0.copy()
        temps = [1.0, 0.5, 0.2, 0.1, 0.05]
        
        for t in temps:
            if time.time() > deadline - 5.0:
                break
                
            def obj(x):
                return _soft_max_linear(x, t)
                
            def grad(x):
                return _soft_max_linear_grad(x, t)
                
            try:
                res = minimize(obj, h, jac=grad, method='L-BFGS-B',
                               bounds=[(0.0, 1.0)] * N,
                               options={'maxiter': 50, 'ftol': 1e-12})
                h = _enforce(res.x)
            except Exception:
                pass
                
        val = _max_linear_overlap(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()

    if best_h is None:
        best_h = np.full(N, 0.5)
        
    return _enforce(best_h)
