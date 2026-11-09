import numpy as np
import time

N = 50

def compute_score(h):
    # h is a numpy array of length N
    # Compute convolution
    # The problem implies the function is defined on [-1/4, 1/4] with 50 steps.
    # The convolution support will be larger.
    # We need to compute the discrete convolution.
    # Standard np.convolve(h, h, mode='full') gives length 2N-1.
    # Does the evaluator use this?
    # "autoconvolution" usually means f*f.
    # The supremum norm is the max of the convolution values.
    # We assume the discrete approximation is sufficient.
    
    c = np.convolve(h, h, mode='full')
    max_c = np.max(c)
    norm2_sq = np.sum(h**2)
    norm1 = np.sum(h)
    
    if norm1 == 0 or max_c == 0:
        return 0.0
        
    return norm2_sq / (norm1 * max_c)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Initialize with a random binary vector
    # Let's try to find a good k first
    best_score = 0.0
    best_h = np.zeros(N)
    
    # Try a few random initializations
    for _ in range(10):
        if time.time() > deadline:
            break
        # Random k between 5 and 15
        k = np.random.randint(5, 16)
        h = np.zeros(N)
        indices = np.random.choice(N, k, replace=False)
        h[indices] = 1.0
        
        score = compute_score(h)
        if score > best_score:
            best_score = score
            best_h = h.copy()
            
        # Local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Randomly pick an index to flip
            idx = np.random.randint(N)
            h_old = h[idx]
            h[idx] = 1.0 - h_old
            
            new_score = compute_score(h)
            if new_score > best_score:
                best_score = new_score
                best_h = h.copy()
                improved = True
            else:
                h[idx] = h_old
                
            # Occasionally try adding or removing a random element
            # This is covered by flipping if we start with binary.
            # But what if we need to change k? Flipping does that.
            
    # Final polish: try to optimize heights?
    # For binary, the score is k / max_c.
    # If we allow non-binary, we might smooth the max_c.
    # Let's try a simple gradient step on the continuous version.
    # But binary might be optimal for this specific ratio?
    # Let's stick to binary search for now, it's fast.
    
    # Convert to list of floats
    return [float(x) for x in best_h]
