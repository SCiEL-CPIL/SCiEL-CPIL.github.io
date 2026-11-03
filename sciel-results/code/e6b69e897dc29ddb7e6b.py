import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for 120s limit
    
    # Initial guess: Barker-like or random
    # Let's start with a random sequence with mean 0, then shift to mean 1/N?
    # Or start with ones and perturb.
    # Starting with ones gives high sidelobes (linear decay).
    # Random signs give low sidelobes but sum ~ 0.
    # We need sum = 1.
    
    # Strategy:
    # 1. Generate random h
    # 2. Normalize sum to 1
    # 3. Minimize sum of squared sidelobes
    
    np.random.seed(42)
    h = np.random.randn(N)
    
    # Normalize to sum 1
    h = h / np.sum(h)
    
    best_h = h.copy()
    best_score = np.inf
    
    # Optimization loop
    # We minimize J = sum_{k != 0} A[k]^2
    # Gradient descent
    
    # Precompute indices for k != 0
    # A[k] = conv(h, h)[N-1+k] for k in 0..N-1 (linear conv length 2N-1)
    # Actually, np.convolve(h, h, 'full') gives length 2N-1.
    # Index 0 is k = -(N-1). Index N-1 is k=0.
    # Sidelobes are all indices except N-1.
    
    for i in range(1000):
        if time.time() > deadline:
            break
            
        # Compute autocorrelation
        A = np.convolve(h, h, 'full')
        
        # Extract sidelobes (exclude center)
        center = N - 1
        # Sidelobes are A[0:center] and A[center+1:]
        # Note: A is symmetric.
        # We can just take the first half to save time, or all.
        # Let's take all k != 0.
        # Mask for k != 0
        # Indices: 0 to 2N-2. Center is N-1.
        
        # Objective: sum of squares of sidelobes
        # grad_h = 2 * sum_{k != 0} A[k] * (h_{j+k} + h_{j-k})
        
        # Efficient gradient calculation using FFT?
        # Or just use the property that grad is related to convolution.
        # Let B = A with center removed (set to 0).
        # Then grad = 2 * convolve(B, h, 'full')?
        # Let's verify.
        # A[k] = sum_i h[i] h[i+k].
        # d/dh[j] A[k] = h[j+k] + h[j-k].
        # sum_k A[k] (h[j+k] + h[j-k])
        # = sum_k A[k] h[j+k] + sum_k A[k] h[j-k]
        # Let m = j+k => k = m-j. Sum_m A[m-j] h[m].
        # This is a cross-correlation.
        # Since A is symmetric, A[m-j] = A[j-m].
        # So it's conv(A, h) at j?
        # conv(A, h)[j] = sum_m A[m] h[j-m].
        # Yes.
        # So grad = 2 * (conv(A, h, 'full') + conv(A, h, 'full'))? No.
        # The two sums are identical due to symmetry of A and reversal in conv?
        # Let's just compute the gradient numerically if analytical is tricky, 
        # or use the conv trick.
        
        B = A.copy()
        B[center] = 0.0
        
        # grad = 2 * np.convolve(B, h, 'full')
        # But we need the part corresponding to h indices.
        # conv(B, h) has length 2N-1 + 2N-1 - 1? No.
        # B is length 2N-1. h is length N.
        # conv length 3N-2?
        # We only need the middle N values corresponding to h[0..N-1].
        # The gradient w.r.t h[j] is the j-th element of the vector G.
        # G = 2 * (B * h) where * is appropriate correlation.
        
        # Let's use a simpler approach: finite difference or just use the 
        # "gradient" of the max function directly on the values?
        # No, let's stick to sum of squares.
        
        # Actually, for N=400, we can just compute the gradient explicitly 
        # using the formula:
        # grad[j] = 2 * sum_{k != 0} A[k] * (h[j+k] + h[j-k])
        # This is O(N^2). 160k ops. 1000 iters = 160M ops. 
        # NumPy can do this in < 1s if vectorized.
        
        # Vectorized gradient:
        # A is length 2N-1.
        # We want to compute for each j in 0..N-1:
        # sum_{k} A[k] * (h[j+k] + h[j-k])
        
        # This is equivalent to the convolution of A (padded?) and h?
        # Let's just use np.convolve on B and h?
        # B is length 2N-1. h is length N.
        # conv(B, h, 'full') length 3N-2.
        # The gradient for h[j] corresponds to index j + (N-1) in the convolution?
        # Let's test with small N.
        
        # Alternative: Use FFT for speed if needed, but N=400 is small.
        
        # Let's try the convolve trick.
        # G_full = np.convolve(B, h, 'full')
        # The term for h[j] is G_full[j + N - 1]?
        # Let's assume this and verify logic.
        
        G_full = np.convolve(B, h, 'full')
        # The relevant part is G_full[N-1 : N-1+N]
        grad = 2.0 * G_full[N-1 : N-1+N]
        
        # Gradient Descent Step
        # We need to maintain sum(h) = 1.
        # Project gradient onto the hyperplane sum(h)=1.
        # grad_proj = grad - mean(grad)
        
        grad_mean = np.mean(grad)
        grad_proj = grad - grad_mean
        
        # Learning rate
        # Scale by current objective value?
        obj_val = np.sum(B**2)
        if obj_val == 0:
            break
            
        lr = 0.01 / (obj_val + 1e-9)
        
        h_new = h - lr * grad_proj
        
        # Re-normalize
        h_new = h_new / np.sum(h_new)
        
        # Check improvement
        # Compute new objective
        A_new = np.convolve(h_new, h_new, 'full')
        B_new = A_new.copy()
        B_new[center] = 0.0
        obj_new = np.sum(B_new**2)
        
        if obj_new < obj_val:
            h = h_new
            if obj_new < best_score:
                best_score = obj_new
                best_h = h.copy()
        else:
            # If it didn't improve, maybe reduce learning rate or skip
            # For simplicity, just keep old h
            pass
            
    # Final normalization to ensure sum is exactly 1 (floating point drift)
    best_h = best_h / np.sum(best_h)
    
    return best_h

# EVOLVE-BLOCK-END
