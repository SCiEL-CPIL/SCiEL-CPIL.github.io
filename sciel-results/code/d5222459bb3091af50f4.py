import numpy as np
import time

def arithmetic_kakeya_distribution():
    n = 10
    # Initialize P
    P = np.ones((n, n), dtype=np.float64) / (n * n)
    
    # Precompute projection values
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y)
    
    # Projections
    V_m1 = X - Y
    V_0 = X
    V_1 = X + Y
    V_2 = X + 2*Y
    
    # Unique values and counts for binning
    # We need to map each cell to a bin index for each projection
    # And compute the probability of each bin
    
    def get_bins(V):
        # Returns unique values and an array of indices into unique values for each cell
        # Or just use np.unique with return_inverse
        vals, inverse = np.unique(V, return_inverse=True)
        return vals, inverse
        
    vals_m1, inv_m1 = get_bins(V_m1)
    vals_0, inv_0 = get_bins(V_0)
    vals_1, inv_1 = get_bins(V_1)
    vals_2, inv_2 = get_bins(V_2)
    
    num_bins_m1 = len(vals_m1)
    num_bins_0 = len(vals_0)
    num_bins_1 = len(vals_1)
    num_bins_2 = len(vals_2)
    
    # Helper to compute entropy and gradient
    # p_bins: array of bin probabilities
    # inv: array mapping cells to bins
    def compute_entropy_grad(p_flat, inv, num_bins):
        # p_flat is P flattened (100,)
        # bin_probs = np.zeros(num_bins)
        # np.add.at(bin_probs, inv, p_flat)
        # This is efficient enough for 100 cells
        
        bin_probs = np.bincount(inv, weights=p_flat, minlength=num_bins)
        
        # Entropy
        # Ignore zero probs
        mask = bin_probs > 0
        if not np.any(mask):
            return 0.0, np.zeros_like(p_flat)
        
        p = bin_probs[mask]
        H = -np.sum(p * np.log2(p))
        
        # Gradient
        # dH/dp_bin = -log2(p_bin) - 1/ln(2)
        # dH/dP_cell = dH/dp_bin[bin(cell)]
        
        grad_bin = np.zeros(num_bins)
        grad_bin[mask] = -np.log2(p) - 1.0/np.log(2)
        
        grad_cell = grad_bin[inv]
        
        return H, grad_cell

    start_time = time.time()
    deadline = 600.0 # seconds
    # We will run for a while, maybe 500s
    
    eta = 0.01 # learning rate
    
    while time.time() < deadline - 10:
        P_flat = P.flatten()
        
        H_m1, grad_m1 = compute_entropy_grad(P_flat, inv_m1, num_bins_m1)
        H_0, grad_0 = compute_entropy_grad(P_flat, inv_0, num_bins_0)
        H_1, grad_1 = compute_entropy_grad(P_flat, inv_1, num_bins_1)
        H_2, grad_2 = compute_entropy_grad(P_flat, inv_2, num_bins_2)
        
        M = max(H_0, H_1, H_2)
        if M < 1e-9:
            break
            
        # Determine active gradients for M
        # If H_0 is max, use grad_0. If tie, average.
        # To be robust, we can use a soft-max weighting or just pick the max.
        # Picking the max can cause oscillation. Averaging active ones is better.
        
        active_grad = np.zeros_like(P_flat)
        max_val = M
        active_count = 0
        
        if abs(H_0 - max_val) < 1e-6:
            active_grad += grad_0
            active_count += 1
        if abs(H_1 - max_val) < 1e-6:
            active_grad += grad_1
            active_count += 1
        if abs(H_2 - max_val) < 1e-6:
            active_grad += grad_2
            active_count += 1
            
        if active_count > 0:
            active_grad /= active_count
        else:
            # Fallback
            active_grad = grad_0
            
        # Objective J = H_m1 / M
        # dJ = (1/M) * dH_m1 - (H_m1 / M^2) * dM
        
        grad_J = (1.0 / M) * grad_m1 - (H_m1 / (M * M)) * active_grad
        
        # Update
        P_flat = P_flat + eta * grad_J
        
        # Project to simplex
        # Clip negatives
        P_flat[P_flat < 0] = 0
        
        # Normalize
        s = np.sum(P_flat)
        if s == 0:
            P_flat = np.ones_like(P_flat) / len(P_flat)
        else:
            P_flat /= s
            
        P = P_flat.reshape((n, n))
        
        # Decay learning rate slightly?
        # eta *= 0.999
        
    return P
