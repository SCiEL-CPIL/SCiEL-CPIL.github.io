import numpy as np
import time

N = 100
DEADLINE = time.time() + 140  # 10s margin

def _time_remaining():
    return max(0, DEADLINE - time.time())

def _compute_overlaps(h):
    # h is N x 1 or N
    # Overlap at shift k is sum h_i (1 - h_{i+k})
    # We want to minimize max overlap.
    # Overlap = sum h_i - sum h_i h_{i+k} = N/2 - C_k
    # Minimize max (N/2 - C_k) = N/2 - min C_k
    # So maximize min C_k.
    # C_k = sum h_i h_{i+k}
    C = np.zeros(N)
    for k in range(N):
        if k == 0:
            C[k] = np.sum(h**2)
        else:
            # Circular shift? The problem says "shifted". 
            # Usually in these tasks, it's circular shift or linear?
            # "shifted 1-h" usually implies circular in signal processing contexts unless specified.
            # Given N=100 and "step heights", circular is likely.
            h_shifted = np.roll(h, k)
            C[k] = np.sum(h * h_shifted)
    return C

def construct():
    # Start with h = 0.5
    h = np.full(N, 0.5)
    x = np.zeros(N)
    
    best_h = h.copy()
    best_min_C = np.min(_compute_overlaps(best_h))
    
    # Optimization loop
    # We optimize x directly. h = 0.5 + x.
    # Constraint: sum(x) = 0, -0.5 <= x <= 0.5.
    
    # Initialize x with some noise to break symmetry?
    # Or start from 0.
    
    # Gradient ascent on min(C_k)
    # C_k(x) = sum (0.5+x_i)(0.5+x_{i+k}) = 25 + R_k(x)
    # So we maximize min R_k(x).
    
    step_size = 0.01
    for step in range(1000):
        if _time_remaining() < 5:
            break
            
        # Compute C and find min
        C = _compute_overlaps(h)
        min_C = np.min(C)
        
        if min_C > best_min_C:
            best_min_C = min_C
            best_h = h.copy()
            
        # Find indices of min C
        min_idx = np.argwhere(C == min_C).flatten()
        
        # Gradient of C_k w.r.t x
        # C_k = sum (0.5+x_i)(0.5+x_{i+k})
        # dC_k/dx_j = 0.5+x_{j-k} + 0.5+x_j (from terms where j is i or i+k)
        # Actually, C_k = sum_i x_i x_{i+k} + 0.5 sum x_i + 0.5 sum x_{i+k} + 25
        # Since sum x = 0, C_k = 25 + R_k.
        # R_k = sum x_i x_{i+k}
        # dR_k/dx_j = x_{j-k} + x_j
        
        grad = np.zeros(N)
        for k in min_idx:
            # Add gradient of R_k
            # d/dx_j (sum x_i x_{i+k}) = x_{j-k} + x_j
            grad += np.roll(x, -k) + x
            
        # Average gradient
        grad /= len(min_idx)
        
        # Project gradient to sum(x)=0
        grad -= np.mean(grad)
        
        # Update x
        x += step_size * grad
        
        # Project to bounds and sum constraint
        # Simple projection:
        # 1. Clip to [-0.5, 0.5]
        x = np.clip(x, -0.5, 0.5)
        # 2. Adjust to sum 0
        # This is a projection onto the intersection of box and hyperplane.
        # Simple iterative projection or just subtract mean?
        # Subtracting mean might violate bounds.
        # If we subtract mean, we might push values out of bounds.
        # Let's do a few iterations of alternating projection.
        for _ in range(10):
            x = np.clip(x, -0.5, 0.5)
            x -= np.mean(x)
            x = np.clip(x, -0.5, 0.5)
            
        h = 0.5 + x
        
        # Check if we converged or improved
        if np.max(np.abs(x)) < 1e-6:
            break
            
    return best_h
