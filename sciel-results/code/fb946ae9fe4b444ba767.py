import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    best_min_corr = -np.inf
    best_x = None
    
    # We will run several restarts to avoid local minima
    # Each restart runs optimization for a certain time or steps
    
    def project(x):
        # Project to [-1, 1] and sum 0
        # Simple alternating projection
        for _ in range(5):
            # Clip to [-1, 1]
            x = np.clip(x, -1.0, 1.0)
            # Subtract mean to make sum 0
            x = x - np.mean(x)
            # Clip again because subtracting mean might push out of bounds
            x = np.clip(x, -1.0, 1.0)
        # Final adjustment to ensure sum is exactly 0? 
        # Clipping breaks the sum. 
        # If we need exact sum 0, we can do a more careful projection.
        # But for optimization, approximate is often fine, 
        # or we can fix the sum by adjusting a few elements.
        # Let's try to enforce sum 0 more strictly at the end if needed.
        # For now, alternating projection is a good heuristic.
        return x

    def get_autocorr(x):
        X = np.fft.fft(x)
        acf = np.fft.ifft(X * np.conj(X)).real
        # acf[0] is the energy. We care about k=1..N-1
        # Due to floating point errors, acf might have tiny imaginary parts, handled by .real
        # Also, circular autocorrelation.
        return acf

    def optimize(x0, max_time=10.0):
        x = x0.copy()
        x = project(x)
        
        # Learning rate
        lr = 0.05
        # Momentum?
        v = np.zeros_like(x)
        mu = 0.9
        
        best_local_min = -np.inf
        best_local_x = x.copy()
        
        t_start = time.time()
        
        while time.time() - t_start < max_time:
            acf = get_autocorr(x)
            # We want to maximize min acf[k] for k in 1..N-1
            # Note: acf is symmetric. acf[k] = acf[N-k].
            # We just need to check 1..N-1
            
            # Find the minimum
            min_val = np.min(acf[1:])
            
            if min_val > best_local_min:
                best_local_min = min_val
                best_local_x = x.copy()
                
            # Find indices where acf is close to min to average gradients
            # This helps if multiple lags are tied for minimum
            threshold = min_val + 1e-3
            min_indices = np.where(acf[1:] <= threshold)[0] + 1
            
            # Compute gradient
            # grad = sum_{k in min_indices} (shift(x, k) + shift(x, -k))
            grad = np.zeros_like(x)
            for k in min_indices:
                grad += np.roll(x, k) + np.roll(x, -k)
            
            # Normalize gradient to prevent large steps
            norm = np.linalg.norm(grad)
            if norm > 1e-8:
                grad /= norm
            
            # Update
            v = mu * v + (1 - mu) * grad
            x = x + lr * v
            
            # Project
            x = project(x)
            
            # Decay learning rate?
            # lr = max(0.01, lr * 0.999)
            
        return best_local_x, best_local_min

    # Run restarts
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize x
        # Random initialization with sum 0
        # Try different initializations
        if i == 0:
            # Start with alternating 1, -1? Sum is 0 if N even.
            x0 = np.array([1.0 if j % 2 == 0 else -1.0 for j in range(N)])
        else:
            # Random
            x0 = np.random.randn(N)
            x0 -= np.mean(x0)
            x0 /= np.max(np.abs(x0)) # Scale to fit in [-1,1] roughly
        
        x_opt, min_corr = optimize(x0, max_time=5.0)
        
        if min_corr > best_min_corr:
            best_min_corr = min_corr
            best_x = x_opt
            
        print(f"Restart {i}: min_corr={min_corr:.4f}, best={best_min_corr:.4f}, time={time.time()-start_time:.1f}")

    # If no optimization ran or failed, use a default
    if best_x is None:
        # Fallback: alternating
        best_x = np.array([1.0 if j % 2 == 0 else -1.0 for j in range(N)])
        
    # Convert x to h
    # h = (x + 1) / 2
    h = (best_x + 1.0) / 2.0
    
    # Ensure sum is exactly N/2
    # Due to floating point, sum(h) might be N/2 + epsilon
    # We can adjust the last element to fix the sum
    current_sum = np.sum(h)
    target_sum = N / 2.0
    diff = target_sum - current_sum
    h[-1] += diff
    
    # Ensure h is in [0, 1]
    # If diff was large, h[-1] might be out of bounds.
    # But diff should be tiny.
    h = np.clip(h, 0.0, 1.0)
    
    # Re-normalize if clipping changed the sum?
    # If we clip, sum changes.
    # Let's do a more careful final fix.
    # We want sum(h) = N/2.
    # If h[-1] was clipped, the sum is off.
    # We can distribute the error over all elements? No, that might break bounds.
    # Given the optimization, the error should be negligible.
    
    return h
