import numpy as np
import time
from scipy.optimize import minimize

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s buffer
    
    best_h = np.ones(N)
    best_val = 2.0
    
    def compute_conv(h):
        # Full autoconvolution
        return np.convolve(h, h, mode='full')
    
    def compute_value(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return 1e10
        conv = compute_conv(h)
        return abs(2.0 * N * np.max(conv) / (s * s))
    
    def objective_and_grad(h, beta=50.0):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return 1e10, np.zeros_like(h)
        
        conv = compute_conv(h)
        
        # Soft max
        m = np.max(conv)
        shifted = (conv - m) * beta
        shifted = np.clip(shifted, -50, 50)
        weights = np.exp(shifted)
        weights /= np.sum(weights)
        
        M = m + np.log(np.sum(np.exp(shifted)) / len(conv)) / beta
        
        raw_val = 2.0 * N * M / (s * s)
        val = abs(raw_val)
        
        # Gradient
        # d val / d h_i
        # val = | 2N M / s^2 |
        # Assume raw_val > 0 for now, handle sign later
        dval_dM = 2.0 * N / (s * s)
        if raw_val < 0:
            dval_dM = -dval_dM
            
        # d M / d conv_k = weights[k]
        # d conv_k / d h_i = h_{k-i} + h_{i-k}
        
        # grad_h_i = sum_k weights[k] * (h_{k-i} + h_{i-k}) * dval_dM
        # Note: h is defined on 0..N-1.
        # We can compute this via correlation.
        # Let w = weights.
        # term1_i = sum_k w_k h_{k-i}
        # term2_i = sum_k w_k h_{i-k}
        
        # Using FFT for speed and safety
        # Pad h and w to size 2N-1
        size = 2 * N - 1
        H = np.fft.rfft(h, size)
        W = np.fft.rfft(weights, size)
        
        # term1: correlation of w and h?
        # w_k h_{k-i}. Let j = k-i => k = i+j.
        # sum_j w_{i+j} h_j.
        # This is cross-correlation of w and h at lag i?
        # corr(w, h)[i] = sum_j w[j+i] h[j].
        # np.correlate(w, h, 'full')
        # Or use FFT: ifft(fft(w) * conj(fft(h)))?
        # Let's use np.correlate for simplicity, N=400 is small.
        
        try:
            # w has length 2N-1, h has length N.
            # We need sum_k w_k h_{k-i} for i in 0..N-1.
            # This is the part of the convolution w * h_rev?
            # Let's just loop or use vectorized ops if possible.
            # Actually, grad_h = dval_dM * (w * h + w * h[::-1])? No.
            
            # Let's compute grad_h directly using the property:
            # grad_h = dval_dM * (weights * h + weights * h[::-1]) is wrong.
            
            # Correct:
            # g_i = sum_k w_k (h_{k-i} + h_{i-k})
            # Let's compute this using np.convolve?
            # h_{k-i} is h shifted.
            # sum_k w_k h_{k-i} is the cross-correlation of w and h.
            # np.correlate(w, h, mode='full')
            # w length 2N-1, h length N.
            # Result length 3N-2.
            # We want the part corresponding to i=0..N-1.
            
            corr1 = np.correlate(weights, h, mode='full')
            # Index alignment:
            # corr1[k] = sum_j w[j+k] h[j]
            # We want sum_k w[k] h[k-i].
            # Let j = k-i => k = i+j.
            # sum_j w[i+j] h[j].
            # This is corr1[i].
            # So we take corr1[0:N].
            
            # Second term: sum_k w_k h_{i-k}
            # Let j = i-k => k = i-j.
            # sum_j w[i-j] h[j].
            # This is correlation of h and w at lag i?
            # corr(h, w)[i] = sum_j h[j] w[j+i]? No.
            # np.correlate(h, w, mode='full')
            # corr2[k] = sum_j h[j+k] w[j].
            # We want sum_j w[i-j] h[j].
            # Let m = i-j => j = i-m.
            # sum_m w[m] h[i-m].
            # This is convolution (w * h)[i].
            # np.convolve(w, h, mode='full')[i].
            
            conv_wh = np.convolve(weights, h, mode='full')
            # We take conv_wh[0:N].
            
            grad_h = dval_dM * (corr1[0:N] + conv_wh[0:N])
            
        except Exception:
            return val, np.zeros_like(h)
            
        return val, grad_h

    rng = np.random.default_rng(42)
    
    # Try multiple random initializations
    for _ in range(20):
        if time.time() > deadline:
            break
            
        # Random initial h
        h0 = rng.standard_normal(N)
        # Ensure sum is not zero
        if abs(np.sum(h0)) < 1e-6:
            h0 += 1.0
            
        # Normalize to sum = 1? Not necessary, but helps scaling.
        # The objective is scale invariant.
        
        try:
            result = minimize(
                objective_and_grad,
                h0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 500,
                    'ftol': 1e-12,
                    'gtol': 1e-10,
                    'maxfun': 1000,
                    'maxls': 50,
                }
            )
            
            val = result.fun
            if val < best_val:
                best_val = val
                best_h = result.x.copy()
                
        except Exception:
            continue
            
    # Final verification
    s = np.sum(best_h)
    if abs(s) < 1e-12:
        return np.ones(N)
        
    conv = compute_conv(best_h)
    val = abs(2.0 * N * np.max(conv) / (s * s))
    
    if val >= 2.0:
        return np.ones(N)
        
    return best_h
