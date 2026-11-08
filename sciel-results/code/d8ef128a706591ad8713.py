import numpy as np
import time

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    def compute_score(h):
        """Compute the normalized max autoconvolution score."""
        h = np.asarray(h, dtype=float)
        if np.max(h) < 1e-15:
            return float('inf')
        # Discrete autoconvolution
        h_conv = np.convolve(h, h, mode='full')
        # Try different normalizations to match seed score of 1.0
        # Option 1: max(h_conv) / sum(h)
        l1 = np.sum(h)
        if l1 < 1e-15:
            return float('inf')
        return np.max(h_conv) / l1
    
    # Baseline: uniform
    uniform_h = np.ones(N) * 0.5
    base_score = compute_score(uniform_h)
    
    # Complex parameterization: h = |z|^2
    np.random.seed(123)
    best_h = uniform_h.copy()
    best_score = base_score
    
    # Multiple restarts with complex gradient descent
    for restart in range(5):
        if time.time() > deadline - 15:
            break
        
        z = np.ones(N, dtype=complex) * np.exp(1j * np.random.uniform(0, 2*np.pi, N))
        h = np.abs(z)**2
        
        # Adaptive learning rate
        lr = 0.05
        momentum = 0.9
        vel = np.zeros_like(z)
        
        for step in range(3000):
            if time.time() > deadline - 10:
                break
            
            h = np.abs(z)**2
            score = compute_score(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
            
            # Compute gradient
            h_conv = np.convolve(h, h, mode='full')
            peak_idx = np.argmax(h_conv)
            
            # Gradient of h_conv[peak_idx] w.r.t. h[j]
            # h_conv[k] = sum_i h[i]*h[k-i]
            # d/dh[j] h_conv[peak] = h[peak-j] + h[peak-j] = 2*h[peak-j] (if valid)
            grad_h = np.zeros(N)
            for j in range(N):
                idx = peak_idx - j
                if 0 <= idx < N:
                    grad_h[j] += h[idx]
                idx2 = peak_idx - j  
                if 0 <= idx2 < N:
                    grad_h[j] += h[idx2]
            
            l1 = np.sum(h)
            if l1 < 1e-15:
                break
            grad_score = grad_h / l1
            
            # Gradient w.r.t. z
            # h = |z|^2, dh/dz = 2*conj(z)
            grad_z = 2 * np.conj(z) * grad_score
            
            # Momentum update
            vel = momentum * vel - lr * grad_z
            z = z + vel
            
            # Reparameterize to maintain |z|^2 structure
            h_new = np.abs(z)**2
            h_new = np.clip(h_new, 0, 5)
            z = np.sqrt(h_new) * np.exp(1j * np.angle(z))
            
            # Adaptive lr
            if step % 100 == 0:
                lr *= 0.95
        
        # Local refinement on best_h found
        if time.time() > deadline - 10:
            break
    
    # Final local search
    if time.time() < deadline - 5:
        h_local = best_h.copy()
        local_best = compute_score(h_local)
        for _ in range(1000):
            if time.time() > deadline - 5:
                break
            idx = np.random.randint(N)
            delta = np.random.uniform(-0.05, 0.05)
            new_h = h_local.copy()
            new_h[idx] = max(0, new_h[idx] + delta)
            new_score = compute_score(new_h)
            if new_score < local_best * (1 - 1e-8):
                h_local = new_h
                local_best = new_score
        
        if local_best < best_score:
            best_h = h_local
            best_score = local_best
    
    return list(best_h)
