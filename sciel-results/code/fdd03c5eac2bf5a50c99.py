import numpy as np

N = 100

def construct():
    # Initial random guess
    np.random.seed(42)
    h = np.random.rand(N)
    
    # Normalize to sum N/2
    h = h / h.sum() * (N / 2.0)
    
    best_h = h.copy()
    best_val = float('inf')
    
    # Precompute shifts
    shifts = np.arange(N)
    
    def compute_overlap(h):
        # Compute overlap for all shifts k
        # overlap_k = sum_i h_i * (1 - h_{i+k})
        # We can do this with rolls
        overlaps = np.zeros(N)
        for k in shifts:
            h_shifted = np.roll(h, k)
            overlaps[k] = np.sum(h * (1 - h_shifted))
        return overlaps

    def project(h, target_sum):
        # Project h to [0,1]^N with sum target_sum
        # Using bisection on lambda
        # h_i = clip(h_i - lambda, 0, 1)
        lo = h.min() - 1.0
        hi = h.max() + 1.0
        
        for _ in range(50):
            mid = (lo + hi) / 2.0
            h_proj = np.clip(h - mid, 0.0, 1.0)
            s = h_proj.sum()
            if s > target_sum:
                lo = mid
            else:
                hi = mid
        return np.clip(h - (lo + hi) / 2.0, 0.0, 1.0)

    # Soft max parameter
    tau = 10.0
    lr = 0.05
    
    # Optimization loop
    max_iters = 5000
    import time
    start_time = time.time()
    
    for i in range(max_iters):
        if time.time() - start_time > 110:
            break
            
        overlaps = compute_overlap(h)
        
        # Soft max to approximate max
        # We want to minimize max(overlaps)
        # Using log-sum-exp for smoothness
        m = np.max(overlaps)
        log_sum_exp = m + np.log(np.sum(np.exp(overlaps - m)))
        soft_max_val = log_sum_exp / tau
        
        # Gradient of soft max
        # d/dh [ (1/tau) log sum exp(tau * overlap_k) ]
        # = (1/tau) * (sum exp(tau * overlap_k) * tau * grad(overlap_k)) / sum exp(tau * overlap_k)
        # = sum (w_k * grad(overlap_k)) where w_k = exp(tau * overlap_k) / sum exp(tau * overlap_k)
        
        exp_overlaps = np.exp(tau * (overlaps - m))
        weights = exp_overlaps / np.sum(exp_overlaps)
        
        # Compute gradient
        # overlap_k = sum_i h_i (1 - h_{i+k})
        # grad_i = (1 - h_{i+k}) - h_{i-k} (from the term where i is the shifted index? No)
        # Let's derive grad of sum_j h_j (1 - h_{j+k}) w.r.t h_i.
        # Term 1: h_i appears as h_j with j=i. Deriv: (1 - h_{i+k}).
        # Term 2: h_i appears as h_{j+k} with j+k=i => j=i-k. Deriv: -h_{i-k}.
        # So grad_i = (1 - h_{i+k}) - h_{i-k}.
        
        grad = np.zeros(N)
        # Efficiently compute grad for all k weighted by weights
        # grad = sum_k weights[k] * ( (1 - roll(h, k)) - roll(h, -k) )
        
        # We can vectorize this or just loop since N=100
        for k in shifts:
            w = weights[k]
            if w < 1e-6: continue
            term1 = 1.0 - np.roll(h, k)
            term2 = np.roll(h, -k)
            grad += w * (term1 - term2)
            
        # Update
        h_new = h - lr * grad
        h_new = project(h_new, N / 2.0)
        
        # Check improvement
        overlaps_new = compute_overlap(h_new)
        val_new = np.max(overlaps_new)
        
        if val_new < best_val:
            best_val = val_new
            best_h = h_new.copy()
            
        h = h_new
        
    # Final check and return
    overlaps_final = compute_overlap(best_h)
    final_val = np.max(overlaps_final)
    
    # The score is raw value. 
    # The task says "score is the upper bound on C5... evaluator measures the raw value".
    # The seed returns 0.5.
    # We should return the array h.
    
    return best_h
