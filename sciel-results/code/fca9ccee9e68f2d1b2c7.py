import numpy as np
import time

N = 50

def _compute_ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * sup|h*h|)"""
    h = np.asarray(h, dtype=float)
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

def _gradient_log_ratio(h):
    """
    Compute gradient of log(R(h)) = log(||h||_2^2) - log(||h||_1) - log(sup|h*h|)
    """
    h = np.asarray(h, dtype=float)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    
    if l1 < 1e-15 or l2_sq < 1e-15:
        return np.zeros_like(h)
        
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(conv)
    if sup_conv < 1e-15:
        return np.zeros_like(h)
        
    # Gradient of log(||h||_2^2) = 2h / ||h||_2^2
    grad_l2 = 2.0 * h / l2_sq
    
    # Gradient of log(||h||_1) = 1 / ||h||_1 for h_i > 0
    # For h_i == 0, subgradient is [0, 1], we use 0 to encourage sparsity growth? 
    # Actually, we want to maximize, so we subtract this term.
    # If h_i is 0, increasing it slightly increases L1 linearly but L2 quadratically.
    # The term -log(L1) has gradient -1/L1.
    grad_l1 = np.zeros_like(h)
    mask_pos = h > 1e-12
    grad_l1[mask_pos] = 1.0 / l1
    
    # Gradient of log(sup|h*h|)
    # Let k* be the index of max conv.
    # (h*h)_k = sum_i h_i h_{k-i}
    # d/dh_i (h*h)_{k*} = h_{k*-i} (with 0 for out of bounds)
    # We need to be careful with indices.
    # conv index j corresponds to sum_{i} h_i h_{j-i} where h is defined on 0..N-1.
    # conv length is 2N-1. Index 0..2N-2.
    # h is 0..N-1.
    # If k* is the index in conv, then for a given i in 0..N-1, the partner index is k*-i.
    # This partner must be in 0..N-1.
    
    k_star = np.argmax(conv)
    
    # Construct the vector g where g_i = h_{k_star - i} if 0 <= k_star - i < N else 0
    grad_conv = np.zeros_like(h)
    for i in range(N):
        j = k_star - i
        if 0 <= j < N:
            grad_conv[i] = h[j]
            
    # The term is - log(sup_conv), so gradient is - grad_conv / sup_conv
    grad_sup = -grad_conv / sup_conv
    
    return grad_l2 - grad_l1 + grad_sup

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    best_h = np.zeros(N)
    best_ratio = 0.0
    
    # Initialize with a few random sparse starts
    # Also include the known good Sidon set as a starting point
    sidon_S = [0, 1, 4, 10, 18, 23, 30]
    h_sidon = np.zeros(N)
    for idx in sidon_S:
        h_sidon[idx] = 1.0 / 7.0
    best_h = h_sidon.copy()
    best_ratio = _compute_ratio(h_sidon)
    
    num_restarts = 20
    steps_per_restart = 500
    
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Initialize
        if r == 0:
            h = h_sidon.copy()
        else:
            # Random sparse initialization
            k = np.random.randint(3, 10) # 3 to 9 non-zeros
            idxs = np.random.choice(N, k, replace=False)
            h = np.zeros(N)
            h[idxs] = 1.0 / k
            # Add small noise to break symmetry
            h += np.random.rand(N) * 1e-6
            h[h < 0] = 0
            
        # Gradient ascent
        lr = 0.5
        for step in range(steps_per_restart):
            if time.time() > deadline - 5.0:
                break
                
            grad = _gradient_log_ratio(h)
            
            # Adaptive step size: simple scaling
            # We want to move in the direction of increasing log-ratio
            h_new = h + lr * grad
            
            # Project to non-negative
            h_new[h_new < 0] = 0
            
            # Normalize? The ratio is scale invariant:
            # R(c*h) = (c^2 ||h||_2^2) / (c ||h||_1 * c ||h*h||_inf) = R(h)
            # So scaling doesn't matter. But numerical stability might.
            # Let's keep the L1 norm roughly constant to avoid under/overflow
            l1_new = np.sum(h_new)
            if l1_new > 1e-10:
                h_new = h_new / l1_new * np.sum(h) # Keep same total mass?
                # Actually, just normalizing to sum=1 is fine and keeps numbers small
                h_new = h_new / l1_new
                
            # Check for improvement
            ratio_new = _compute_ratio(h_new)
            
            if ratio_new > best_ratio:
                best_ratio = ratio_new
                best_h = h_new.copy()
                
            h = h_new
            
            # Decay learning rate slightly
            lr *= 0.999

    # Final check
    final_ratio = _compute_ratio(best_h)
    if final_ratio > best_ratio:
        best_ratio = final_ratio
        
    return list(best_h)

if __name__ == "__main__":
    # Quick test
    h = construct()
    print(_compute_ratio(h))
