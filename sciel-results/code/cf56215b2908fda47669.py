import numpy as np
import time

N = 100

def _max_overlap(h):
    max_c = 0.0
    for k in range(N):
        c_k = np.dot(h[:N-k], 1.0 - h[k:])
        if c_k > max_c:
            max_c = c_k
    return max_c

def _calc_gradients(h, k):
    # Gradient of C_k w.r.t h_i
    # C_k = sum_{i=0}^{N-k-1} h_i (1 - h_{i+k})
    # d C_k / d h_i = (1 - h_{i+k}) - h_{i-k} (if i-k >= 0)
    # Note: if i+k >= N, term is 0.
    # If i-k < 0, term is 0.
    g = np.zeros(N)
    # Term 1: h_i * (1 - h_{i+k})
    # For i in 0..N-k-1, d/dh_i = 1 - h_{i+k}
    g[:N-k] += 1.0 - h[k:]
    # Term 2: - h_{i-k} * h_i
    # For i in k..N-1, d/dh_i = - h_{i-k}
    g[k:] -= h[:N-k]
    return g

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initialize with 0.5
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    # Try sinusoidal patterns to get a better start
    for freq in range(1, 10):
        for amp in [0.5, 0.4, 0.3]:
            for phase in [0, np.pi/2]:
                if time.time() > deadline:
                    return best_h
                i = np.arange(N)
                h = 0.5 + amp * np.cos(2 * np.pi * freq * i / N + phase)
                h = np.clip(h, 0.0, 1.0)
                # Correct sum
                diff = N/2.0 - np.sum(h)
                h += diff / N
                h = np.clip(h, 0.0, 1.0)
                # Final correction
                diff = N/2.0 - np.sum(h)
                if diff > 0:
                    h[0] = min(1.0, h[0] + diff)
                else:
                    h[0] = max(0.0, h[0] + diff)
                
                score = _max_overlap(h)
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
                    
    h = best_h.copy()
    current_score = best_score
    
    # Local Search
    # Greedy: find k*, compute gradient, move mass from high grad to low grad
    while time.time() < deadline:
        # Find k*
        c_vals = np.array([np.dot(h[:N-k], 1.0 - h[k:]) for k in range(N)])
        k_star = np.argmax(c_vals)
        
        if c_vals[k_star] <= best_score:
             # No improvement possible? Or we are at a local min for this k
             # Try random perturbation if stuck
             pass
             
        g = _calc_gradients(h, k_star)
        
        # Find i with max g, j with min g
        i = np.argmax(g)
        j = np.argmin(g)
        
        if i == j:
            continue
            
        # Determine max delta
        # h[i] + delta <= 1, h[j] - delta >= 0
        max_delta = min(1.0 - h[i], h[j])
        
        if max_delta <= 1e-6:
            continue
            
        # Try a fraction of max delta
        delta = 0.1 * max_delta
        
        h_new = h.copy()
        h_new[i] += delta
        h_new[j] -= delta
        
        score_new = _max_overlap(h_new)
        
        if score_new < current_score:
            h = h_new
            current_score = score_new
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
        else:
            # If no improvement, try smaller delta or random move
            # To avoid infinite loop, break if stuck for a while
            # Simple: just try a random move
            i = np.random.randint(N)
            j = np.random.randint(N)
            if i == j: continue
            delta = 0.01 * np.random.choice([-1, 1])
            h_new = h.copy()
            h_new[i] += delta
            h_new[j] -= delta
            if h_new[i] < 0 or h_new[i] > 1 or h_new[j] < 0 or h_new[j] > 1:
                continue
            score_new = _max_overlap(h_new)
            if score_new < current_score:
                h = h_new
                current_score = score_new
                if current_score < best_score:
                    best_score = current_score
                    best_h = h.copy()

    return best_h
