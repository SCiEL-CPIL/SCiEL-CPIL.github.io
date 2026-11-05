import numpy as np
import time
import sys

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill

    def true_value(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return float('inf')
        conv_h = np.convolve(h, h)
        return abs(2.0 * N * np.max(conv_h) / (s * s))

    def objective(h, beta=30.0):
        # Smooth max of conv(h,h) / sum(h)^2
        s = np.sum(h)
        if abs(s) < 1e-12:
            return 1e10
        conv_h = np.convolve(h, h)
        c_max = np.max(conv_h)
        # Numerical stability
        if c_max > 1e8:
            return c_max / (s * s)
        exp_term = np.exp(beta * (conv_h - c_max))
        sum_exp = np.sum(exp_term)
        log_sum_exp = c_max + np.log(sum_exp)
        soft_max = log_sum_exp / beta
        return abs(2.0 * N * soft_max / (s * s))

    def gradient(h, beta=30.0):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.zeros(N)
        conv_h = np.convolve(h, h)
        c_max = np.max(conv_h)
        exp_term = np.exp(beta * (conv_h - c_max))
        sum_exp = np.sum(exp_term)
        w = exp_term / sum_exp  # weights over conv entries, length 2N-1
        
        # Gradient of soft_max w.r.t. h:
        # d(soft_max)/d(h_i) = (1/s) * sum_k w[k] * (h[k-i] + h[k-i] if k==2i else 0)
        # More precisely: d(conv[h,h])/dh_i = 2*corr(w, h) with proper indexing
        # Using: grad_i = 2 * sum_k w[k] h[k-i] (with bounds), plus diagonal correction
        # Efficient via FFT:
        # The linear operator L: h -> conv(h,h) has adjoint L*(w) = 2 * ifft(fft(w) * conj(fft(h)))
        # But w is defined on the conv domain (length 2N-1), not frequency domain directly.
        
        # Direct but vectorized approach:
        # grad_i = sum_k w[k] * (h[k-i] if 0<=k-i<N else 0)
        # This is the cross-correlation of w and h, reversed.
        # Let's compute it as: for each i, grad[i] = dot(w[i:i+N], h) + diagonal term
        # Actually: d(conv[h,h][k])/dh[i] = h[k-i] if 0<=k-i<N, plus if k-i==i, it's h[i] (appears twice in product)
        # So d(conv[h,h][k])/dh[i] = h[k-i] (counted once from first h, once from second h if indices match)
        # = h[k-i] + h[k-i] * delta_{k,2i} ... no.
        # conv[h,h][k] = sum_j h[j] h[k-j]
        # d/dh[i] = h[k-i] (from j=i) + h[i] (from k-j=i, j=k-i)
        # = h[k-i] + h[k-i] if k-i==i, i.e., k=2i
        # So: d(conv[h,h][k])/dh[i] = h[k-i] + h[k-i] * (1 if k==2i else 0)
        # = h[k-i] * (1 + delta_{k,2i})
        
        # grad[i] = sum_k w[k] * h[k-i] * (1 + delta_{k,2i})
        #          = sum_k w[k] h[k-i] + w[2i] h[i]  (if 2i < 2N-1)
        
        # First term: sum_k w[k] h[k-i] = (w * h)_i in correlation sense
        # Let's compute T = np.convolve(w, h)  -> T[m] = sum_j w[j] h[m-j]
        # We want sum_k w[k] h[k-i]. Let p = k-i, k = p+i: sum_p w[p+i] h[p]
        # This is the cross-correlation. np.correlate(w, h, 'full')[N-1+i] ?
        # Actually, let's just use the FFT approach for the linear part:
        # The mapping h -> conv(h,h) is quadratic. The gradient of sum_k w[k] conv[h,h][k] w.r.t h
        # is 2 * ifft(fft(w_padded) * conj(fft(h_padded))) where w_padded is w zero-padded to 2N.
        # But w is already in the time domain of the convolution.
        
        # Simpler: use the fact that for fixed w, the function f(h) = sum_k w[k] conv[h,h][k]
        # has gradient: grad[i] = 2 * sum_{k} w[k] h[k-i] (with bounds 0<=k-i<N)
        # The factor of 2 comes from both h's in the product.
        # But wait: conv[h,h][k] = sum_j h[j]h[k-j]. d/dh[i] = h[k-i] (j=i) + h[i] (k-j=i => j=k-i)
        # These are the same term h[k-i] counted twice ONLY if k-i==i, i.e., k=2i.
        # Otherwise they are two different terms: h[k-i] from j=i, and h[k-i] from j=k-i.
        # Wait, no: d/dh[i] of h[j]h[k-j]:
        #   if j==i: derivative is h[k-j] = h[k-i]
        #   if k-j==i, i.e., j=k-i: derivative is h[j] = h[k-i]
        # So if i != k-i, we get 2*h[k-i]. If i == k-i (k=2i), we get h[i] + h[i] = 2*h[i].
        # So actually it's ALWAYS 2*h[k-i] when 0<=k-i<N!
        
        # So grad[i] = 2 * sum_{k: 0<=k-i<N} w[k] h[k-i]
        # Let p = k-i: grad[i] = 2 * sum_{p=0}^{N-1} w[p+i] h[p]  (where w[p+i] is 0 if p+i >= 2N-1)
        
        # This is 2 * (w * h) where * is the appropriate correlation.
        # Let's compute it via FFT:
        # Define W = fft(w, n=2*N)  (w is length 2N-1, pad to 2N)
        # Define H = fft(h, n=2*N)
        # Then the correlation sum_p w[p+i] h[p] is ifft(W * conj(H))[i] ? 
        # Let's think: ifft(W * H)[m] = sum_p W[p] H[m-p] in freq domain = time domain correlation.
        # Actually, np.fft.ifft(np.fft.fft(w) * np.fft.fft(h)) gives circular convolution.
        # We need linear correlation. With zero-padding to 2N, circular = linear.
        
        w_padded = np.zeros(2 * N)
        w_padded[:2*N-1] = w
        h_padded = np.zeros(2 * N)
        h_padded[:N] = h
        
        W = np.fft.fft(w_padded)
        H = np.fft.fft(h_padded)
        
        # We want for each i: sum_p w[p+i] h[p]
        # This is the cross-correlation of w and h at lag i.
        # np.correlate(w, h, 'full') gives sum_p w[p] h[p+i] ... let me check.
        # Actually, let's just compute it directly with the FFT trick:
        # cross_corr[i] = sum_p w[p+i] h[p]
        # = sum_p w[p+i] h[p]
        # Let q = p+i, p = q-i: sum_q w[q] h[q-i]
        # This is the convolution of w and h reversed?
        
        # Easier: just compute grad directly with vectorized dot products.
        # N=400, 2N-1=799. For each i, we need dot(w[i:i+N], h) but w has length 2N-1.
        # w[i:i+N] has min(N, 2N-1-i) elements.
        
        grad = np.zeros(N)
        for i in range(N):
            # sum_{p=0}^{N-1} w[p+i] * h[p], where w[p+i] is 0 if p+i >= 2N-1
            end = min(N, 2*N-1-i)
            if end > 0:
                grad[i] = 2.0 * np.dot(w[i:i+end], h[:end])
        
        # Now, the full objective is abs(2*N*soft_max / s^2)
        # d/dh of (2*N*soft_max / s^2) = 2*N * (d(soft_max)/dh * s^2 - soft_max * 2*s*ds/dh) / s^4
        # = 2*N * (grad_sm * s - 2*soft_max*ds/dh) / s^3
        # where grad_sm = d(soft_max)/dh = grad / s (since soft_max = log_sum_exp/beta, and the w already includes the 1/beta... wait)
        
        # Actually, let me recompute. soft_max = log(sum exp(beta*(conv-cmax)))/beta + cmax
        # d(soft_max)/d(conv[k]) = exp(beta*(conv[k]-cmax))/sum_exp = w[k]
        # So d(soft_max)/dh[i] = sum_k w[k] d(conv[k])/dh[i] = grad[i] as computed above (with factor 2 already included)
        
        # Wait, I included the factor of 2 in grad above. Let me recheck.
        # d(soft_max)/dh[i] = sum_k w[k] * d(conv[k])/dh[i] = sum_k w[k] * 2*h[k-i] (with bounds)
        # So grad[i] as I computed (with the 2.0 factor) IS d(soft_max)/dh[i].
        
        # Now objective = abs(2*N*soft_max / s^2). Assume soft_max/s^2 > 0 (which it should be for reasonable h).
        # d(obj)/dh[i] = 2*N * (grad[i]*s^2 - soft_max*2*s*ds/dh[i]) / s^4
        # = 2*N * (grad[i]*s - 2*soft_max*ds/dh[i]) / s^3
        # ds/dh[i] = 1 for all i.
        
        soft_max_val = c_max + np.log(sum_exp) / beta
        s2 = s * s
        s3 = s2 * s
        
        # d(obj)/dh[i] = 2*N*(grad[i]*s - 2*soft_max_val) / s^3
        # Note: grad[i] already has the factor of 2 from the derivative of conv.
        
        obj_grad = 2.0 * N * (grad * s - 2.0 * soft_max_val) / s3
        
        # But we're minimizing abs(...). If the quantity is positive, this is correct.
        # If negative, we'd need to flip sign. For our purposes, assume positive.
        
        return obj_grad

    best_h = np.ones(N)
    best_val = true_value(best_h)  # Should be 2.0

    # Try multiple random flat-spectrum initializations
    np.random.seed(12345)
    n_restarts = 8
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Create flat-spectrum initialization
        # Random DFT: uniform magnitude, random phases
        freq = np.fft.rfftfreq(N)
        # Use full FFT for simplicity
        H_mag = np.ones(N)  # Flat magnitude
        H_phase = np.random.uniform(0, 2*np.pi, N)
        H = H_mag * np.exp(1j * H_phase)
        # Ensure H[0] is real positive to keep sum positive
        H[0] = abs(H[0])
        h = np.real(np.fft.ifft(H))
        
        # Ensure sum is positive
        if np.sum(h) < 0:
            h = -h
        
        s = np.sum(h)
        if abs(s) < 1e-12:
            continue
        h = h / s  # Normalize to sum 1
        
        # Gradient descent with adaptive step
        h_curr = h.copy()
        step_size = 0.1
        for it in range(150):
            if time.time() > deadline:
                break
            
            g = gradient(h_curr, beta=30.0)
            g_norm = np.linalg.norm(g)
            if g_norm < 1e-10:
                break
            
            # Normalize gradient for stable steps
            g_normed = g / g_norm
            h_trial = h_curr - step_size * g_normed
            
            # Ensure sum stays positive
            s_trial = np.sum(h_trial)
            if abs(s_trial) < 1e-12:
                step_size *= 0.5
                continue
            h_trial = h_trial / s_trial
            
            val_trial = true_value(h_trial)
            val_curr = true_value(h_curr)
            
            if val_trial < val_curr:
                h_curr = h_trial
                step_size = min(step_size * 1.1, 1.0)
            else:
                step_size *= 0.7
        
        val_final = true_value(h_curr)
        if val_final < best_val:
            best_val = val_final
            best_h = h_curr.copy()
        
        if time.time() > deadline - 10:
            break

    # Final validation
    s = np.sum(best_h)
    if abs(s) < 1e-12:
        return np.ones(N)
    
    return best_h
