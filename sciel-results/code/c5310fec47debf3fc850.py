import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial guess: uniform
    h = np.full(N, 0.5)
    
    # Precompute indices for circular shifts
    # We want to compute overlaps O_k = sum_i h_i * (1 - h_{(i+k)%N})
    # This is equivalent to sum_i h_i - sum_i h_i * h_{(i+k)%N}
    # sum_i h_i is constant 50.
    # So minimizing max O_k is equivalent to maximizing min sum_i h_i * h_{(i+k)%N}
    # Or minimizing max (50 - R_k) where R_k is autocorrelation.
    # This is equivalent to maximizing min R_k.
    
    # However, the prompt says "minimize maximum overlap of h with shifted 1-h".
    # Overlap_k = sum h_i (1-h_{i+k}).
    
    # Use Projected Gradient Descent on the log-sum-exp of Overlaps.
    # L(x) = log(sum(exp(beta * O_k)))
    # Gradient w.r.t h_i:
    # d L / d h_i = beta * sum_k (O_k / sum(exp(beta*O_k))) * d O_k / d h_i
    # d O_k / d h_i = (1 - h_{i+k}) - h_{i-k}  (from term i in O_k and term i-k in O_{k-k})
    # Wait, O_k = sum_j h_j (1 - h_{j+k}).
    # d O_k / d h_i = (1 - h_{i+k})  (from j=i)
    # Plus from O_{k'} where j+k' = i => k' = k-j.
    # Term h_{i-k'} (1-h_i). Deriv is -h_{i-k'}.
    # So d O_k / d h_i = (1 - h_{i+k}) - h_{i-k}.
    
    beta = 10.0
    lr = 0.01
    best_h = h.copy()
    best_score = np.inf
    
    for iter in range(10000):
        if time.time() > deadline:
            break
            
        # Compute overlaps
        # O_k = sum_i h_i - sum_i h_i h_{i+k}
        # R_k = sum_i h_i h_{i+k}
        # Use FFT for speed? N=100 is small, direct is fine.
        R = np.zeros(N)
        for k in range(N):
            R[k] = np.sum(h * np.roll(h, k))
        O = 50.0 - R
        
        # Compute log-sum-exp
        max_O = np.max(O)
        log_sum_exp = max_O + np.log(np.sum(np.exp(O - max_O)))
        
        if log_sum_exp < best_score:
            best_score = log_sum_exp
            best_h = h.copy()
            
        # Compute gradient
        # weights w_k = exp(O_k - max_O) / sum(exp(O_k - max_O))
        exp_O = np.exp(O - max_O)
        w = exp_O / np.sum(exp_O)
        
        # grad[i] = beta * sum_k w_k * ( (1 - h_{i+k}) - h_{i-k} )
        grad = np.zeros(N)
        for k in range(N):
            if w[k] > 1e-9:
                # term from O_k: w_k * (1 - h_{i+k})
                # term from O_{k'} where i-k' = i => k'=0? No.
                # d O_k / d h_i = (1 - h_{i+k}) - h_{i-k}
                # Note: index i-k is circular.
                grad += w[k] * ((1 - np.roll(h, -k)) - np.roll(h, k))
        
        grad *= beta
        
        # Gradient step
        h -= lr * grad
        
        # Project to [0, 1] and sum=50
        h = np.clip(h, 0, 1)
        # Adjust sum
        diff = 50.0 - np.sum(h)
        if np.abs(diff) > 1e-9:
            # Simple projection: add diff/N, then clip, iterate?
            # Or just distribute diff to elements not at bounds.
            h += diff / N
            h = np.clip(h, 0, 1)
            # Re-adjust if needed (simple loop)
            for _ in range(10):
                diff = 50.0 - np.sum(h)
                if np.abs(diff) < 1e-9:
                    break
                # Add to elements with room
                mask = (h < 1) if diff > 0 else (h > 0)
                if np.sum(mask) == 0:
                    break
                h[mask] += diff / np.sum(mask)
                h = np.clip(h, 0, 1)
                
    return best_h
