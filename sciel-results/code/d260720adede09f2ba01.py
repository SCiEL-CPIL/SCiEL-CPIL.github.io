import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    # Initial guess: constant
    h = np.full(N, 0.5)
    
    # Precompute shift indices for linear correlation
    # L_k = sum_{i} h_i * (1 - h_{i-k})
    # Valid i for shift k: max(0, k) <= i <= N-1-k
    # Actually, if we define shift k as h_{i-k}, then i-k must be in [0, N-1].
    # So i in [k, N-1].
    # The term is h_i * (1 - h_{i-k}).
    
    def compute_overlaps(h):
        overlaps = np.zeros(N)
        for k in range(N):
            # i ranges from k to N-1
            if k == 0:
                # h_i * (1 - h_i)
                overlaps[k] = np.sum(h * (1 - h))
            else:
                # h[k:] * (1 - h[:-k])
                overlaps[k] = np.sum(h[k:] * (1 - h[:-k]))
        return overlaps

    def compute_grad(h, k):
        # Gradient of L_k w.r.t h
        # L_k = sum_{i=k}^{N-1} h_i (1 - h_{i-k})
        # dL_k / dh_j:
        # 1. If j is in [k, N-1], term h_j (1 - h_{j-k}) exists. Deriv: (1 - h_{j-k})
        # 2. If j-k is in [k, N-1] i.e. j in [2k, N-1+k], term h_{j+k} (1 - h_j) exists.
        #    Wait, the sum is over i. Term is h_i (1 - h_{i-k}).
        #    If we vary h_j, it appears as h_i when i=j, and as h_{i-k} when i-k=j => i=j+k.
        #    So contributions:
        #    From i=j: (1 - h_{j-k}) if j >= k and j-k >= 0.
        #    From i=j+k: -h_{j+k} if j+k < N and j >= 0.
        #    Note: if k=0, these are the same term? No, h_j(1-h_j). Deriv 1-2h_j.
        #    My formula: (1-h_j) - h_j = 1-2h_j. Correct.
        
        grad = np.zeros(N)
        if k == 0:
            grad[:] = 1 - 2 * h
        else:
            # Part 1: h_j is the first factor. j in [k, N-1]. h_{j-k} is second.
            # Valid if j-k >= 0.
            # j >= k.
            # grad[j] += (1 - h[j-k])
            
            # Part 2: h_j is the second factor (inside 1-h). i = j+k.
            # Term is h_{j+k} (1 - h_j). Deriv w.r.t h_j is -h_{j+k}.
            # Valid if j+k < N.
            # j < N-k.
            
            # Combine
            # Indices for Part 1: k to N-1
            # Indices for Part 2: 0 to N-1-k
            
            # Vectorized update
            # Part 1
            grad[k:] += (1 - h[:N-k])
            
            # Part 2
            grad[:N-k] -= h[k:]
            
        return grad

    best_h = h.copy()
    best_score = np.inf
    
    step_size = 0.1
    
    while time.time() < deadline:
        overlaps = compute_overlaps(h)
        current_max = np.max(overlaps)
        
        if current_max < best_score:
            best_score = current_max
            best_h = h.copy()
            
        # Find the shift(s) with max overlap
        max_idx = np.argmax(overlaps)
        
        # Compute gradient for that shift
        grad = compute_grad(h, max_idx)
        
        # We want to decrease the max overlap.
        # Move h in direction -grad.
        # But we must keep sum(h) = N/2.
        # Project -grad onto sum=0 hyperplane.
        # v = -grad - mean(-grad)
        v = -grad
        v -= np.mean(v)
        
        # Update h
        h_new = h + step_size * v
        
        # Clip to [0, 1]
        h_new = np.clip(h_new, 0, 1)
        
        # Re-normalize sum to N/2
        # Simple adjustment: add/subtract constant
        # But this breaks [0,1] potentially.
        # Better: iterative projection or just accept small error and fix at end?
        # The constraint is strict.
        # Let's do a simple scaling if sum is out of bounds, but sum is additive.
        # If sum(h_new) != N/2, we need to adjust.
        # Since v has mean 0, sum(h_new) should be close to sum(h).
        # Due to clipping, it might drift.
        # Let's correct by distributing the error evenly, respecting bounds.
        target_sum = N / 2.0
        current_sum = np.sum(h_new)
        diff = target_sum - current_sum
        if abs(diff) > 1e-6:
            # Distribute diff
            # If diff > 0, we need to increase some h_i.
            # We can add diff/N to all, then clip, then repeat?
            # Or just add to those with room.
            # Simple: add diff/N to all, clip.
            h_new += diff / N
            h_new = np.clip(h_new, 0, 1)
            # This might not sum exactly to N/2 after clipping.
            # For the sake of the loop, this is okay.
            # At the end, we will project best_h exactly.
            
        h = h_new
        
        # Adaptive step size?
        # If score didn't improve, maybe reduce step size?
        # Or just keep going.
        
    # Final projection of best_h to ensure sum is exactly N/2 and bounds [0,1]
    # We can use a simple water-filling or just assume the loop kept it close.
    # To be safe, let's do a final exact adjustment on best_h.
    # Since best_h was a local min, small adjustments are fine.
    
    # Ensure sum is exactly N/2
    # Find indices where we can adjust.
    # This is a linear programming problem: minimize ||h - best_h||_2 subject to sum h = N/2, 0<=h<=1.
    # We can solve this quickly.
    
    def project_to_sum(h, target_sum):
        # Simple iterative projection
        for _ in range(100):
            s = np.sum(h)
            diff = target_sum - s
            if abs(diff) < 1e-9:
                break
            # Add diff/N to all
            h += diff / N
            h = np.clip(h, 0, 1)
        return h

    best_h = project_to_sum(best_h, N/2.0)
    
    # Verify
    # overlaps = compute_overlaps(best_h)
    # return np.max(overlaps)
    
    return best_h
