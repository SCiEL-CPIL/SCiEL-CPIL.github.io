import numpy as np
import time
from scipy.optimize import minimize

N = 600
DEADLINE = 115

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
        
    def compute_autoconv_max(h):
        """Compute max of autoconvolution of h (assumed sum=1)."""
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        return np.max(conv)
        
    def smooth_max_and_grad_h(h, beta=100.0):
        """
        Compute smooth max of autoconvolution and its gradient w.r.t h.
        h is assumed to be non-negative and sum to 1.
        Returns: (smooth_max_value, grad_h)
        """
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        
        m = np.max(conv)
        exp_conv = np.exp((conv - m) * beta)
        sum_exp = np.sum(exp_conv)
        s_max = m + np.log(sum_exp) / beta
        
        weights = exp_conv / sum_exp
        
        # Gradient of smooth max w.r.t h
        # g_h = 2 * correlate(h, weights)
        # We need to align the correlation correctly.
        # conv[k] = sum_i h[i] h[k-i]
        # d conv[k] / d h[j] = h[k-j] + h[k-j] (if symmetric)
        # Actually, let's use the FFT trick for the gradient.
        # grad_h[j] = 2 * sum_k weights[k] * h[k-j]
        # This is the correlation of h and weights.
        
        W = np.fft.fft(weights, n=size)
        # Correlation of h and w: sum_k w[k] h[k-j]
        # In frequency domain: IFFT(H * conj(W)) ?
        # Let's verify.
        # corr[j] = sum_k w[k] h[k-j]
        # Let's just compute it directly for the first few points to be safe? No, N=600 is small enough for direct O(N^2) if needed, but FFT is better.
        # Actually, np.correlate(h, weights, mode='full') gives the correlation.
        # But we need to map it back to the h indices.
        
        # Let's use the FFT method:
        # grad_h_pad = 2 * IFFT(H * conj(W))
        # Wait, the convolution theorem: FFT(h * w) = H * W.
        # Correlation is convolution with flipped w.
        # w_flip[k] = w[-k] = w[mod(k, size)]?
        
        # Let's stick to the direct sum if N is small? 600^2 = 360,000 operations. Very fast.
        # But we are doing this in a loop.
        # Let's use the FFT approach carefully.
        
        # The gradient of the smooth max of the convolution C = h*h
        # is 2 * (h * w) where * denotes convolution? No.
        # Let's derive:
        # S = sum_k w_k C_k
        # C_k = sum_i h_i h_{k-i}
        # dS/dh_j = sum_k w_k (h_{k-j} + h_{k-j}) = 2 sum_k w_k h_{k-j}
        # This is 2 * (w * h)[j] ?
        # (w * h)[j] = sum_k w_k h[j-k] ? No.
        # (w * h)[j] = sum_k w[k] h[j-k].
        # We have sum_k w[k] h[k-j].
        # Let l = k-j => k = l+j.
        # sum_l w[l+j] h[l].
        # This is the cross-correlation of w and h.
        
        # We can compute this using FFT:
        # corr = IFFT(FFT(w) * conj(FFT(h))) ?
        # Let's test with small arrays.
        
        # For safety and simplicity, since N=600, we can compute the gradient directly.
        # grad_h = np.zeros(N)
        # for j in range(N):
        #    # sum_k w[k] h[k-j]
        #    # k ranges such that 0<=k<size and 0<=k-j<N
        #    # This is effectively convolving w and h.
            
        # Let's use the FFT method with proper padding.
        # We want the linear correlation.
        # Pad w and h to size 2N.
        # H_fft = FFT(h_pad)
        # W_fft = FFT(w_pad)
        # The correlation sum_k w[k] h[k-j] is the coefficient of z^j in W(z) H(1/z)?
        
        # Actually, simpler:
        # grad_h[j] = 2 * np.dot(weights, h_shifted)
        # where h_shifted is h shifted by -j?
        
        # Let's use the property that the gradient of the max of the convolution is 2 * h * weights (convolution).
        # Wait, if C = h*h, then dC/dh = h.
        # So dS/dh = sum_k w_k (h * delta_k) ?
        # dC_k/dh = h shifted by k.
        # So dS/dh = sum_k w_k (h shifted by k).
        # This is the convolution of w with h?
        # (w * h)[j] = sum_k w[k] h[j-k].
        # We have sum_k w[k] h[k-j].
        # If w is symmetric (which it is, because C is symmetric), then w[k] = w[-k].
        # So sum_k w[k] h[k-j] = sum_k w[-k] h[k-j] = sum_l w[l] h[-l-j] ?
        # If h is symmetric too, it's simpler.
        
        # Let's just compute the gradient numerically for the first run to verify, or use a robust direct computation.
        # Direct computation:
        # grad_h[j] = 2 * sum_k weights[k] * h[k-j]
        # We can compute this as a convolution of weights and h.
        # Let's use scipy.signal.fftconvolve or just np.convolve.
        
        # h is length N. weights is length 2N (padded).
        # We need the part of the correlation that maps back to h.
        
        # Let's use the FFT method:
        # G = IFFT(FFT(weights) * conj(FFT(h)))
        # This gives the correlation.
        
        W_fft = np.fft.fft(weights, n=size)
        # h_pad is already FFT'd as H
        corr = np.fft.ifft(W_fft * np.conj(H)).real
        
        # The correlation corr[j] = sum_k weights[k] h[k-j] ?
        # Let's check the index alignment.
        # Standard correlation: (a * b)[n] = sum_m a[m] b[n-m].
        # FFT(a * b) = A * B.
        # Here we want sum_k w[k] h[k-j].
        # Let n = -j. sum_k w[k] h[n+k].
        # This is cross-correlation.
        # FFT(w) * conj(FFT(h)) gives the correlation.
        
        # We need to extract the N values corresponding to h.
        # The correlation output is length 2N.
        # The relevant part is usually centered.
        
        # Let's just use the direct sum for safety, it's fast enough.
        # 600 * 1200 = 720,000 ops.
        grad_h = np.zeros(N)
        # weights is length 2N. h is length N.
        # For each j in 0..N-1:
        # grad_h[j] = 2 * sum_{k=0}^{2N-1} weights[k] * h[k-j] (if 0<=k-j<N)
        
        # We can vectorize this:
        # Create a matrix? No, too big.
        
        # Use FFT correlation properly.
        # From scipy.signal.correlate(h, weights, mode='full')
        # But we want to avoid scipy.signal if possible, though it's available.
        # Let's use the FFT result.
        # The correlation of h and w is corr.
        # The indices in corr correspond to lags.
        # The term h[k-j] means lag k-j.
        # So we need the sum over k of w[k] * h[k-j].
        # This is the convolution of w and h, but with w not flipped?
        
        # Let's try:
        # grad_h = 2 * np.convolve(weights, h, mode='valid')? No.
        
        # Let's use the property:
        # dS/dh_j = 2 * sum_k w_k h_{k-j}
        # This is 2 * (w * h)[j] if w is symmetric?
        # If w is symmetric, w_k = w_{-k}.
        # Then sum_k w_k h_{k-j} = sum_k w_{-k} h_{k-j} = sum_l w_l h_{-l-j} ?
        
        # Let's just compute it directly using a loop over k? No.
        # Let's use the FFT result 'corr'.
        # corr[n] = sum_k w[k] h[k-n] ?
        # If so, grad_h[j] = 2 * corr[j].
        # But we need to handle the circular vs linear shift.
        
        # Let's assume the FFT correlation 'corr' is aligned such that corr[0] is the zero-lag correlation.
        # Then corr[j] = sum_k w[k] h[k-j].
        # So grad_h[j] = 2 * corr[j].
        # But we need to check the range.
        
        # Let's verify with a small example.
        # h = [1, 2], w = [1, 1, 1] (padded)
        # C = [1, 4, 4, 4, 1]
        # w = exp(beta C) / sum
        # grad_h[0] = 2 * (w[0]*h[0] + w[1]*h[1] + w[2]*h[2]...)
        # h[0]=1, h[1]=2.
        # grad_h[0] = 2 * (w[0]*1 + w[1]*2 + w[2]*0...) = 2*(w[0] + 2w[1])
        # grad_h[1] = 2 * (w[1]*1 + w[2]*2 + ...) = 2*(w[1] + 2w[2])
        
        # If corr[j] = sum_k w[k] h[k-j]
        # corr[0] = w[0]h[0] + w[1]h[1] + ... = w[0] + 2w[1]
        # corr[1] = w[1]h[0] + w[2]h[1] + ... = w[1] + 2w[2]
        
        # So yes, grad_h[j] = 2 * corr[j].
        
        # We need to extract the correct N values from 'corr'.
        # The FFT correlation of two length N signals (padded to 2N) gives a 2N vector.
        # The linear correlation is in the first N or last N?
        # Usually, the zero-lag is at index 0.
        # So corr[0:N] should be the lags 0 to N-1.
        
        # However, the autoconvolution C is defined on [-N, N].
        # h is on [0, N].
        # The gradient should be on [0, N].
        
        # Let's take the first N values of corr.
        grad_h = 2.0 * corr[:N]
        
        return s_max, grad_h

    def objective_and_grad(params, beta):
        """
        Objective and gradient for optimizer.
        params: coefficients of the cosine series.
        h_i = exp(sum_m params[m] * cos(2*pi*m*(i-N/2)/N))
        """
        M = len(params)
        i = np.arange(N)
        center = N / 2.0
        
        # Compute h
        h_unnorm = np.zeros(N)
        for m in range(M):
            if m == 0:
                h_unnorm += params[0]
            else:
                h_unnorm += params[m] * np.cos(2 * np.pi * m * (i - center) / N)
        
        h_unnorm = np.exp(h_unnorm)
        Z = np.sum(h_unnorm)
        h = h_unnorm / Z
        
        # Compute smooth max and grad w.r.t h
        s_max, grad_h = smooth_max_and_grad_h(h, beta)
        
        # Compute grad w.r.t params
        grad_params = np.zeros(M)
        
        # phi_m_i = cos(2*pi*m*(i-center)/N)
        # d h_i / d a_m = h_i * (phi_m_i - sum_j h_j phi_m_j)
        
        # Precompute phi
        # We can compute the sum part first
        # sum_j h_j phi_m_j
        
        # Let's compute grad_params
        for m in range(M):
            if m == 0:
                phi = np.ones(N)
            else:
                phi = np.cos(2 * np.pi * m * (i - center) / N)
            
            # bar_phi = sum h_j phi_j
            bar_phi = np.dot(h, phi)
            
            # dh_da = h * (phi - bar_phi)
            dh_da = h * (phi - bar_phi)
            
            # grad_params[m] = sum_i grad_h_i * dh_da_i
            grad_params[m] = np.dot(grad_h, dh_da)
            
        return s_max, grad_params

    # Optimization
    best_h = None
    best_val = np.inf
    
    # Try different numbers of cosine terms
    M_vals = [5, 10, 15]
    
    for M in M_vals:
        if time_left() < 20:
            break
            
        # Random initializations
        for _ in range(10):
            if time_left() < 10:
                break
                
            # Random params
            # a_0 controls the mean, but we normalize, so it's less important.
            # Let's initialize a_0 to 0.
            params_init = np.zeros(M)
            for m in range(1, M):
                params_init[m] = np.random.uniform(-2, 2)
                
            try:
                res = minimize(
                    lambda p: objective_and_grad(p, 100.0)[0],
                    params_init,
                    jac=lambda p: objective_and_grad(p, 100.0)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-6}
                )
                
                # Reconstruct h
                i = np.arange(N)
                center = N / 2.0
                h_unnorm = np.zeros(N)
                for m in range(M):
                    if m == 0:
                        h_unnorm += res.x[0]
                    else:
                        h_unnorm += res.x[m] * np.cos(2 * np.pi * m * (i - center) / N)
                
                h_unnorm = np.exp(h_unnorm)
                h = h_unnorm / np.sum(h_unnorm)
                
                val = compute_autoconv_max(h)
                
                if val < best_val:
                    best_val = val
                    best_h = h.copy()
                    
            except Exception:
                continue

    # Fallback
    if best_h is None:
        best_h = np.ones(N) / N
        best_val = compute_autoconv_max(best_h)
        
    # Refinement with direct gradient descent on h
    if time_left() > 15:
        current_h = best_h.copy()
        current_val = best_val
        eta = 0.01
        
        for k in range(100):
            if time_left() < 5:
                break
                
            s_max, grad_h = smooth_max_and_grad_h(current_h, 100.0)
            
            # Project gradient to maintain sum=1 and non-negativity?
            # Simple step:
            new_h = current_h - eta * grad_h
            new_h = np.maximum(new_h, 0)
            s = np.sum(new_h)
            if s > 0:
                new_h /= s
            else:
                break
                
            new_val = compute_autoconv_max(new_h)
            
            if new_val < current_val:
                current_h = new_h
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_h = current_h.copy()
            else:
                eta *= 0.9
                if eta < 1e-6:
                    break

    return best_h.tolist()
