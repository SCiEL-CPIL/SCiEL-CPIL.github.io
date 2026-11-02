<<<<<<< SEARCH
def _gradient_ascent(h, max_iter=200, lr_init=0.05, tol=1e-8):
    """Gradient ascent on log-ratio to optimize amplitudes."""
    h = h.copy()
    h = np.maximum(h, 1e-10)  # keep positive
    
    best_score = _score(h)
    best_h = h.copy()
    
    lr = lr_init
    prev_score = best_score
    
    for it in range(max_iter):
        l1 = np.sum(h)
        l2_sq = np.sum(h * h)
        sup = _autoconv_sup(h)
        
        if l1 < 1e-10 or sup < 1e-10:
            break
        
        score = l2_sq / (l1 * sup)
        if score > best_score:
            best_score = score
            best_h = h.copy()
        
        if abs(score - prev_score) < tol * max(1.0, abs(prev_score)):
            break
        prev_score = score
        
        # Gradient of log(score) = log(l2_sq) - log(l1) - log(sup)
        # d/dh_i log(l2_sq) = 2*h_i / l2_sq
        # d/dh_i log(l1) = 1 / l1
        # d/dh_i log(sup): need gradient of sup of autoconv
        # sup is max_k conv[k], conv[k] = sum_i h[i]*h[(i+k)%N]
        # d conv[k] / d h_i = h[(i+k)%N] + h[(i-k)%N]
        # For the argmax k*, gradient contribution is h[(i+k*)%N] + h[(i-k*)%N]
        
        H = np.fft.fft(h)
        conv = np.real(np.fft.ifft(H * H))
        k_star = np.argmax(conv)
        
        # Gradient of sup w.r.t. h_i
        # d conv[k_star] / d h_i = h[(i+k_star) % N] + h[(i-k_star) % N]
        idx_plus = (np.arange(N) + k_star) % N
        idx_minus = (np.arange(N) - k_star) % N
        grad_sup = h[idx_plus] + h[idx_minus]
        
        grad_log = (2.0 * h / l2_sq) - (1.0 / l1) - grad_sup / sup
        
        # Update
        h = h + lr * grad_log
        h = np.maximum(h, 1e-10)
        
        # Adaptive learning rate
        if it % 20 == 0:
            lr = max(lr * 0.8, 1e-6)
    
    return best_h, best_score
=======
def _gradient_ascent(h, max_iter=200, lr_init=0.05, tol=1e-8):
    """Gradient ascent on log-ratio to optimize amplitudes."""
    h = h.copy()
    h = np.maximum(h, 1e-10)  # keep positive
    
    best_score = _score(h)
    best_h = h.copy()
    
    lr = lr_init
    prev_score = best_score
    
    for it in range(max_iter):
        l1 = np.sum(h)
        l2_sq = np.sum(h * h)
        
        if l1 < 1e-10:
            break
            
        H = np.fft.fft(h)
        conv = np.real(np.fft.ifft(H * H))
        
        # Use a soft-max to smooth the supremum term for better gradient flow
        # beta controls the sharpness; larger beta approaches hard max
        beta = 100.0 
        max_conv = np.max(conv)
        # Numerical stability for log-sum-exp
        conv_shifted = conv - max_conv
        sum_exp = np.sum(np.exp(beta * conv_shifted))
        if sum_exp < 1e-300:
            sum_exp = 1e-300
        soft_sup = max_conv + np.log(sum_exp) / beta
        
        if soft_sup < 1e-10:
            break
        
        score = l2_sq / (l1 * soft_sup)
        if score > best_score:
            best_score = score
            best_h = h.copy()
        
        if abs(score - prev_score) < tol * max(1.0, abs(prev_score)):
            break
        prev_score = score
        
        # Gradient of log(score) = log(l2_sq) - log(l1) - log(soft_sup)
        # d/dh_i log(l2_sq) = 2*h_i / l2_sq
        # d/dh_i log(l1) = 1 / l1
        
        # Gradient of log(soft_sup)
        # soft_sup = max_conv + (1/beta) * log(sum_j exp(beta * conv_j))
        # d/dh_i soft_sup = (1/beta) * (sum_j exp(beta * conv_j) * beta * d/dh_i conv_j) / sum_j exp(beta * conv_j)
        #                  = sum_j (w_j * d/dh_i conv_j) where w_j = exp(beta*conv_j)/sum_exp
        # Note: d/dh_i conv_j = h[(i+j)%N] + h[(i-j)%N]
        
        w = np.exp(beta * conv_shifted) / sum_exp
        # w is a vector of weights for each shift j
        # We need grad_soft_sup[i] = sum_j w_j * (h[(i+j)%N] + h[(i-j)%N])
        
        # This is a convolution-like operation. 
        # grad_soft_sup = conv(w, h) + conv(w, h_flipped)? 
        # Actually, d conv_j / d h_i is linear.
        # Let's compute it directly for N=50, it's cheap enough (50*50=2500 ops)
        # Or use FFT: 
        # conv_j = sum_k h_k h_{k+j}
        # d/dh_i conv_j = h_{i+j} + h_{i-j}
        # grad_soft_sup_i = sum_j w_j (h_{i+j} + h_{i-j})
        
        # Using FFT for speed:
        # sum_j w_j h_{i+j} is a circular cross-correlation or convolution?
        # Let A_i = sum_j w_j h_{i+j}. This is conv(w, h) at index i?
        # (w * h)_i = sum_j w_j h_{i-j}. 
        # If we define w_rev_j = w_{-j}, then sum_j w_j h_{i+j} = sum_k w_{i-k} h_k ... 
        # Easier: 
        # term1_i = sum_j w_j h_{i+j}
        # term2_i = sum_j w_j h_{i-j}
        # term2 is just conv(w, h) if w is symmetric? No.
        # term2_i = sum_j w_j h_{i-j} = (w * h)_i
        # term1_i = sum_j w_j h_{i+j}. Let k = -j. sum_k w_{-k} h_{i-k}. 
        # This is conv(w_rev, h)_i where w_rev_k = w_{-k}.
        
        W = np.fft.fft(w)
        H_fft = np.fft.fft(h)
        # term2 = ifft(W * H_fft)
        # term1: w_rev in freq domain is W_conj? No. 
        # w_rev[k] = w[-k mod N]. FFT(w_rev) = conj(FFT(w)) if w is real? No, FFT(w_rev) = W_conj only if w is real symmetric?
        # Actually, if w is real, FFT(w_rev) = conj(FFT(w)).
        # So term1 = ifft(conj(W) * H_fft)
        
        term2 = np.real(np.fft.ifft(W * H_fft))
        term1 = np.real(np.fft.ifft(np.conj(W) * H_fft))
        
        grad_soft_sup = term1 + term2
        
        grad_log = (2.0 * h / l2_sq) - (1.0 / l1) - grad_soft_sup / soft_sup
        
        # Update
        h = h + lr * grad_log
        h = np.maximum(h, 1e-10)
        
        # Adaptive learning rate
        if it % 20 == 0:
            lr = max(lr * 0.8, 1e-6)
    
    return best_h, best_score
>>>>>>> REPLACE

