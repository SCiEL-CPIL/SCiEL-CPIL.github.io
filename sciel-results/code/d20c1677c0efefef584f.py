import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    timeout = 110.0 # Safety margin
    
    def compute_overlaps(h):
        # h is a vector of length N
        # Overlap k is sum_i h_i * (1 - h_{(i+k)%N})
        # We can compute this using FFT for speed, but N=100 is small.
        # Let's use direct computation for simplicity and clarity, or numpy roll.
        overlaps = np.zeros(N)
        for k in range(1, N):
            shifted = np.roll(h, -k)
            overlaps[k] = np.sum(h * (1 - shifted))
        return overlaps

    def max_overlap(h):
        overlaps = compute_overlaps(h)
        return np.max(overlaps), np.argmax(overlaps)

    def project(h):
        # Project h onto {x : 0 <= x_i <= 1, sum x_i = N/2}
        # 1. Clip to [0, 1]
        h = np.clip(h, 0, 1)
        # 2. Adjust sum
        target_sum = N / 2.0
        current_sum = np.sum(h)
        diff = target_sum - current_sum
        
        if abs(diff) < 1e-9:
            return h
        
        # We need to add diff/N to each element, but respecting bounds.
        # This is equivalent to finding lambda such that sum(clip(h_i + lambda, 0, 1)) = target_sum
        # Since h is already clipped, we can just distribute the difference to elements not at bounds.
        
        # Simple iterative adjustment
        # If diff > 0, we need to increase some h_i.
        # If diff < 0, we need to decrease some h_i.
        
        # Better: Use the water-filling projection directly on the original h before clipping?
        # No, the projection onto the intersection of box and hyperplane is:
        # Find lambda such that sum(clip(h_i + lambda, 0, 1)) = target_sum
        
        # Let's implement the bisection for lambda
        def sum_clipped(h, lam):
            return np.sum(np.clip(h + lam, 0, 1))
        
        # We need sum_clipped(h, lam) = target_sum
        # sum_clipped is monotonic in lam.
        
        # Range of lambda:
        # min possible sum is sum(clip(h, 0, 1)) which is current_sum after clip.
        # Wait, if we start from h, and add lambda, the range of lambda is [-1, 1].
        
        lo = -1.0
        hi = 1.0
        
        # If current_sum (after clip) is already target_sum, lambda=0.
        # But we want to project the *updated* h.
        
        # Let's just find lambda
        for _ in range(50): # Bisection
            mid = (lo + hi) / 2
            s = sum_clipped(h, mid)
            if s < target_sum:
                lo = mid
            else:
                hi = mid
        
        lam = (lo + hi) / 2
        h_new = np.clip(h + lam, 0, 1)
        return h_new

    # Initial point: Random binary
    rng = np.random.default_rng(42)
    h = np.zeros(N)
    indices = rng.permutation(N)
    h[indices[:N//2]] = 1.0
    
    best_h = h.copy()
    best_score, _ = max_overlap(h)
    
    alpha = 0.01
    for it in range(5000):
        if time.time() - start_time > timeout:
            break
            
        score, k_star = max_overlap(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
            
        # Gradient of overlap k_star
        # d/dh_j sum_i h_i (1 - h_{i+k}) = (1 - h_{j-k}) - h_{j+k}
        # Indices are modulo N
        shifted_minus = np.roll(h, k_star) # h_{j-k}
        shifted_plus = np.roll(h, -k_star) # h_{j+k}
        
        grad = (1 - shifted_minus) - shifted_plus
        
        h = h - alpha * grad
        h = project(h)
        
        # Adaptive step size?
        # If score increased, reduce alpha.
        # For now, keep it simple.
        
    # Final polish: try small random perturbations
    for _ in range(100):
        if time.time() - start_time > timeout:
            break
        h_pert = best_h + rng.normal(0, 0.01, N)
        h_pert = project(h_pert)
        score_pert, _ = max_overlap(h_pert)
        if score_pert < best_score:
            best_score = score_pert
            best_h = h_pert
    
    return best_h
