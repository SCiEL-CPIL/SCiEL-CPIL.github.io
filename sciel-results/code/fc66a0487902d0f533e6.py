import numpy as np
import time

N = 50

def _ratio(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * sup|h*h|).
    This is the objective to maximize.
    """
    h = np.asarray(h, dtype=float)
    if np.all(h < 1e-15):
        return 0.0
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-15 or l2_sq < 1e-15:
        return 0.0
    # Autoconvolution
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(conv)
    if sup_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_conv)

def _best_response_update(h, conv, peak_idx, step_size=0.5):
    """
    Perform a best-response update for each coordinate.
    The idea is to reduce coordinates that contribute significantly to the peak
    of the autoconvolution, and potentially increase others if it helps L2.
    
    For a fixed L1 norm, we want to maximize L2^2 / sup_conv.
    Let S = sum h_i = 1. Maximize sum h_i^2 / C, where C = max_k conv_k.
    This is equivalent to minimizing C / sum h_i^2.
    
    The gradient of the peak convolution w.r.t h_i is:
    g_i = h_{k-i} + h_{i-k} (with boundary checks)
    
    We use a multiplicative update to decrease h_i where g_i is large.
    """
    new_h = h.copy()
    
    # Compute gradient of the peak convolution
    g = np.zeros(N)
    k = peak_idx
    for i in range(N):
        val = 0.0
        idx1 = k - i
        if 0 <= idx1 < N:
            val += h[idx1]
        idx2 = i - k
        if 0 <= idx2 < N:
            val += h[idx2]
        g[i] = val
    
    # Multiplicative update: h_i <- h_i * exp(-eta * g_i)
    # This reduces h_i where it contributes to the peak.
    # We also want to maintain L1 = 1.
    
    # To balance, we can also consider the contribution to L2.
    # d/dh_i (sum h_j^2) = 2 h_i.
    # So we want to increase h_i where it contributes to L2 but not to the peak.
    # However, since the peak is the bottleneck, reducing peak contributors is primary.
    
    # Let's use a soft update:
    # h_i <- h_i * (1 - eta * g_i / max(g))
    # But this can go negative. Use exp.
    
    eta = step_size
    # Normalize g to be in [0, 1] roughly
    max_g = np.max(g)
    if max_g > 1e-15:
        g_norm = g / max_g
    else:
        g_norm = g
        
    new_h = h * np.exp(-eta * g_norm)
    
    # Renormalize to sum=1
    s = np.sum(new_h)
    if s < 1e-15:
        return h
    new_h /= s
    
    return new_h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_h = np.zeros(N)
    best_ratio = 0.0
    
    # Initialize with uniform distribution
    h = np.ones(N) / N
    r = _ratio(h)
    if r > best_ratio:
        best_ratio = r
        best_h = h.copy()
        
    # Also check delta as baseline
    h_delta = np.zeros(N)
    h_delta[0] = 1.0
    r_delta = _ratio(h_delta)
    if r_delta > best_ratio:
        best_ratio = r_delta
        best_h = h_delta.copy()
        
    # Strategy: Best-Response Dynamics
    # Each coordinate is a player. The payoff is the ratio.
    # We iteratively update h_i to its best response given the current peak.
    # This is similar to Sinkhorn/KM but for this specific game.
    
    current_h = best_h.copy()
    current_ratio = best_ratio
    
    max_iters = 5000
    patience = 100
    stale = 0
    eta = 0.5
    
    for i in range(max_iters):
        if time.time() > deadline - 5.0:
            break
            
        # 1. Compute autoconvolution
        conv = np.convolve(current_h, current_h, mode='full')
        sup_conv = np.max(conv)
        
        if sup_conv < 1e-15:
            break
            
        # 2. Identify the peak lag
        k_star = np.argmax(conv)
        
        # 3. Best-response update
        new_h = _best_response_update(current_h, conv, k_star, step_size=eta)
        
        # 4. Evaluate
        r_new = _ratio(new_h)
        
        if r_new > current_ratio:
            current_ratio = r_new
            current_h = new_h
            if r_new > best_ratio:
                best_ratio = r_new
                best_h = current_h.copy()
                stale = 0
            else:
                stale += 1
        else:
            # No improvement, decay eta and try again
            stale += 1
            eta *= 0.95
            if eta < 1e-4:
                # Try a random restart or different initialization
                # Sparse random initialization
                mask = np.random.rand(N) > 0.8
                if not np.any(mask):
                    mask[0] = True
                current_h = mask.astype(float)
                current_h /= np.sum(current_h)
                current_ratio = _ratio(current_h)
                if current_ratio > best_ratio:
                    best_ratio = current_ratio
                    best_h = current_h.copy()
                eta = 0.5
                stale = 0
                
        if stale > patience:
            # Try a different strategy: random perturbation
            perturbation = np.random.rand(N) * 0.1
            current_h = current_h + perturbation
            current_h = np.maximum(current_h, 0)
            current_h /= np.sum(current_h)
            current_ratio = _ratio(current_h)
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_h = current_h.copy()
            eta = 0.5
            stale = 0

    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(_ratio(h))
