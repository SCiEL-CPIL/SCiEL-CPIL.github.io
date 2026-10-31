"""Erdos minimum overlap - smooth optimization approach."""
import numpy as np
import time

N = 100

def _autocorr(h):
    """Compute circular autocorrelation for all shifts."""
    # autocorr[k] = sum_i h[i] * h[i+k mod N]
    # Use FFT for efficiency
    H = np.fft.fft(h)
    acf = np.fft.ifft(H * np.conj(H)).real
    return acf  # acf[k] = sum_i h[i] * h[(i+k) % N]

def _overlap_scores(h):
    """Compute overlap at each shift: N/2 - autocorr(k)."""
    acf = _autocorr(h)
    return N / 2.0 - acf

def _softmin(vals, tau):
    """Soft-min approximation."""
    m = np.min(vals)
    return m - tau * np.log(np.mean(np.exp((m - vals) / tau)))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Initialize with 0.5
    h = np.full(N, 0.5)
    
    # Parameters
    tau = 0.05  # soft-min temperature
    lr = 0.01
    max_iter = 50000
    
    best_h = h.copy()
    best_score = np.max(_overlap_scores(h))
    
    for it in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        
        # Compute overlaps
        overlaps = _overlap_scores(h)
        
        # Soft-min of autocorr = N/2 - softmax of overlaps
        # We want to minimize max overlap, so minimize softmax(overlaps)
        # softmax(overlaps) ≈ max + tau*log(mean(exp((overlaps-max)/tau)))
        m = np.max(overlaps)
        softmax_val = m + tau * np.log(np.mean(np.exp((overlaps - m) / tau)))
        
        # Gradient of softmax w.r.t. overlaps
        probs = np.exp((overlaps - m) / tau)
        probs /= probs.sum()
        
        # Gradient of overlap_k w.r.t. h[i]
        # overlap_k = sum_i h[i] * (1 - h[(i+k)%N])
        # d(overlap_k)/d(h[i]) = (1 - h[(i+k)%N]) - h[i-k] (circular)
        # So d(softmax)/d(h[i]) = sum_k probs[k] * [(1 - h[(i+k)%N]) - h[(i-k)%N]]
        
        # Compute gradient efficiently
        # term1[i] = sum_k probs[k] * (1 - h[(i+k)%N])
        # term2[i] = sum_k probs[k] * h[(i-k)%N]
        # Use FFT: term1[i] = sum_k probs[k] - sum_k probs[k]*h[(i+k)%N]
        # sum_k probs[k]*h[(i+k)%N] is circular cross-correlation
        
        # Actually let's just compute directly for N=100 (it's fast enough)
        grad = np.zeros(N)
        for k in range(N):
            if probs[k] < 1e-10:
                continue
            # d(overlap_k)/d(h[i]) = (1 - h[(i+k)%N]) - h[(i-k)%N]
            idx_plus = (np.arange(N) + k) % N
            idx_minus = (np.arange(N) - k) % N
            grad += probs[k] * ((1.0 - h[idx_plus]) - h[idx_minus])
        
        # Project gradient to satisfy sum constraint: subtract mean
        grad -= np.mean(grad)
        
        # Gradient step
        h_new = h - lr * grad
        
        # Project to [0,1] and fix sum
        h_new = np.clip(h_new, 0.0, 1.0)
        # Adjust to make sum exactly N/2
        diff = N / 2.0 - np.sum(h_new)
        if diff != 0:
            # Distribute correction, respecting bounds
            h_new += diff / N
            h_new = np.clip(h_new, 0.0, 1.0)
            # Second correction pass
            diff2 = N / 2.0 - np.sum(h_new)
            if abs(diff2) > 1e-12:
                h_new += diff2 / N
                h_new = np.clip(h_new, 0.0, 1.0)
        
        # Check improvement
        overlaps_new = _overlap_scores(h_new)
        score_new = np.max(overlaps_new)
        
        if score_new < best_score:
            best_score = score_new
            best_h = h_new.copy()
        
        h = h_new
        
        # Anneal temperature
        if it % 1000 == 0 and it > 0:
            tau = max(0.005, tau * 0.8)
        # Decay learning rate
        if it % 5000 == 0 and it > 0:
            lr *= 0.8
    
    # Final polish: local refinement with smaller steps
    if time.time() < deadline - 3.0:
        h = best_h.copy()
        lr2 = 0.001
        tau2 = 0.01
        for it2 in range(20000):
            if time.time() > deadline - 2.0:
                break
            overlaps = _overlap_scores(h)
            m = np.max(overlaps)
            probs = np.exp((overlaps - m) / tau2)
            probs /= probs.sum()
            
            grad = np.zeros(N)
            for k in range(N):
                if probs[k] < 1e-10:
                    continue
                idx_plus = (np.arange(N) + k) % N
                idx_minus = (np.arange(N) - k) % N
                grad += probs[k] * ((1.0 - h[idx_plus]) - h[idx_minus])
            
            grad -= np.mean(grad)
            h_new = np.clip(h - lr2 * grad, 0.0, 1.0)
            diff = N / 2.0 - np.sum(h_new)
            h_new += diff / N
            h_new = np.clip(h_new, 0.0, 1.0)
            diff2 = N / 2.0 - np.sum(h_new)
            if abs(diff2) > 1e-12:
                h_new += diff2 / N
                h_new = np.clip(h_new, 0.0, 1.0)
            
            score_new = np.max(_overlap_scores(h_new))
            if score_new < best_score:
                best_score = score_new
                best_h = h_new.copy()
            h = h_new
            if it2 % 2000 == 0 and it2 > 0:
                lr2 *= 0.8
    
    # Ensure exact sum
    result = best_h.copy()
    diff = N / 2.0 - np.sum(result)
    result += diff / N
    result = np.clip(result, 0.0, 1.0)
    # Final fix
    diff = N / 2.0 - np.sum(result)
    if abs(diff) > 1e-15:
        result += diff / N
    
    return result
