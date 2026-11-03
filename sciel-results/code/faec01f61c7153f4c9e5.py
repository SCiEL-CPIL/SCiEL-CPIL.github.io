import numpy as np
import time

N = 600
M = 2048 # Next power of 2 > 2*N

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_score = float('inf')
    best_h = None
    
    # Multiple restarts to escape local minima
    n_restarts = 50
    
    for r in range(n_restarts):
        # Initialize with random non-negative values
        # Using a mix of random and smooth to start
        h = np.random.rand(N)
        # Ensure non-negative
        h = np.abs(h)
        
        # Normalize
        norm = np.linalg.norm(h)
        if norm == 0:
            h = np.random.rand(N)
            norm = np.linalg.norm(h)
        h = h / norm
        
        # Local optimization
        # We want to minimize max(c[1:N]) / c[0]
        
        # Learning rate schedule
        lr = 0.1
        
        for i in range(200): # Iterations per restart
            if time.time() > deadline:
                break
                
            # Compute autoconvolution
            H = np.fft.fft(h, M)
            C = np.fft.ifft(H * H).real
            c0 = C[0]
            
            if c0 < 1e-10:
                break
                
            # Sidelobes are C[1:N] (positive lags)
            # Note: C[0] is the main lobe
            sidelobes = C[1:N]
            
            # Find the index of the maximum sidelobe
            k_max = np.argmax(sidelobes)
            max_val = sidelobes[k_max]
            
            # Gradient of c_k with respect to h
            # c_k = sum h[j] h[j+k]
            # d c_k / d h[j] = h[j+k] + h[j-k]
            
            # Construct gradient vector
            g = np.zeros(N)
            
            # Term 1: h[j+k] -> shift h by -k
            # Term 2: h[j-k] -> shift h by +k
            
            # Using np.roll for circular approximation is risky for linear, 
            # but for k < N, we can construct it carefully.
            
            # h[j+k] is non-zero for j such that 0 <= j+k < N => -k <= j < N-k
            # So for j in [0, N-k-1], g[j] += h[j+k]
            if N - k_max > 0:
                g[0:N-k_max] += h[k_max:N]
                
            # h[j-k] is non-zero for j such that 0 <= j-k < N => k <= j < N+k
            # So for j in [k_max, N-1], g[j] += h[j-k_max]
            if k_max < N:
                g[k_max:N] += h[0:N-k_max]
                
            # Update h
            h = h - lr * g
            
            # Project to non-negative
            h = np.maximum(h, 0)
            
            # Normalize
            norm = np.linalg.norm(h)
            if norm < 1e-10:
                # Failed, restart this inner loop
                h = np.random.rand(N)
                h = np.abs(h)
                norm = np.linalg.norm(h)
                h = h / norm
                continue
            h = h / norm
            
            # Track best
            current_score = max_val / c0
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
                
        # Optional: Simulated annealing step? 
        # For now, just restart with new random init.
        
        if time.time() > deadline - 10:
            break
            
    if best_h is None:
        # Fallback to seed
        return [1.0] * N
        
    return list(best_h)
