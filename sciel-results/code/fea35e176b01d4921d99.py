import numpy as np

N = 100

def construct():
    # Initialize with uniform
    h = np.full(N, 0.5)
    
    # Time budget
    import time
    start_time = time.time()
    max_time = 110.0 # Safety margin
    
    # Subgradient descent
    eta = 0.01
    # Adaptive learning rate or fixed?
    
    for step in range(5000):
        if time.time() - start_time > max_time:
            break
            
        # Compute overlaps O_k for k=1 to N-1
        # O_k = sum_{i=0}^{N-k-1} h[i] * (1 - h[i+k])
        # We only need the max one for the gradient, but 
        # to be robust, we can track the top few.
        
        # Efficient computation of max overlap
        # Naive is O(N^2) = 10000 ops. Very fast.
        max_overlap = 0
        max_k = 1
        overlaps = np.zeros(N)
        
        for k in range(1, N):
            # Vectorized
            term = h[:N-k] * (1 - h[k:])
            o_k = np.sum(term)
            overlaps[k] = o_k
            if o_k > max_overlap:
                max_overlap = o_k
                max_k = k
                
        # Compute gradient for max_k
        k_star = max_k
        grad = np.zeros(N)
        
        # Part 1: derivative of h[i] in h[i](1-h[i+k])
        # for i in 0..N-k-1
        # d/dh[i] = 1 - h[i+k]
        grad[:N-k_star] += (1 - h[k_star:])
        
        # Part 2: derivative of h[i+k] in h[i](1-h[i+k])
        # Let j = i+k. i = j-k.
        # Term is -h[j-k] * h[j].
        # d/dh[j] = -h[j-k].
        # j ranges from k to N-1.
        # Condition: i < N-k => j-k < N-k => j < N. Always true.
        # Also i >= 0 => j >= k.
        grad[k_star:] += -h[:N-k_star]
        
        # Update
        h = h - eta * grad
        
        # Clip
        h = np.clip(h, 0, 1)
        
        # Normalize
        current_sum = np.sum(h)
        if current_sum > 0:
            h = h * (N / 2) / current_sum
            h = np.clip(h, 0, 1)
            # Re-normalize after clip?
            # If clip changes sum, we need to adjust.
            # Simple iteration:
            for _ in range(10):
                s = np.sum(h)
                if s == 0: break
                h = h * (N/2) / s
                h = np.clip(h, 0, 1)
                
    return h
