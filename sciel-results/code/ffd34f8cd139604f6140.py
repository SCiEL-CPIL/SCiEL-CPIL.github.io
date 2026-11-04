import numpy as np
import time

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_h = None
    best_score = float('inf')
    
    # Number of restarts
    n_restarts = 20
    
    # Precompute indices for autocorrelation? 
    # FFT is easier.
    
    for r in range(n_restarts):
        # Random initialization
        # Uniform random [0, 1]
        h = np.random.rand(N)
        
        # Or start with binary
        # h = np.random.randint(0, 2, N).astype(float)
        
        # Learning rate
        lr = 0.01
        momentum = 0.9
        v = np.zeros(N)
        
        for step in range(5000):
            if time.time() > deadline:
                break
            
            # Compute autocorrelation
            # Zero-pad to 2N for linear convolution/correlation
            h_pad = np.zeros(2*N)
            h_pad[:N] = h
            f = np.fft.fft(h_pad)
            c = np.fft.ifft(f * np.conj(f)).real
            c = c[:N] # c[0] is c0, c[1..N-1] are c_k
            
            c0 = c[0]
            if c0 == 0:
                break
                
            # Find max sidelobe
            # c[1:]
            k_star = np.argmax(c[1:]) + 1
            c_max = c[k_star]
            
            score = c_max / c0
            if score < best_score:
                best_score = score
                best_h = h.copy()
                
            # Gradient calculation
            # d(c_k)/d(h_i) = h_{i+k} + h_{i-k}
            # We need to construct the gradient for c_k_star
            grad_ck = np.zeros(N)
            # h_{i+k} term: for i in 0..N-1-k, grad[i] += h[i+k]
            # h_{i-k} term: for i in k..N-1, grad[i] += h[i-k]
            # Using slicing
            grad_ck[:N-k_star] += h[k_star:]
            grad_ck[k_star:] += h[:N-k_star]
            
            # d(c0)/d(h_i) = 2*h_i
            grad_c0 = 2.0 * h
            
            # d(score)/d(h_i) = (c0 * grad_ck - c_max * grad_c0) / c0^2
            grad_score = (c0 * grad_ck - c_max * grad_c0) / (c0 * c0)
            
            # Update momentum
            v = momentum * v - lr * grad_score
            h = h + v
            
            # Project to non-negative
            h[h < 0] = 0.0
            
            # Optional: Normalize? No, ratio is scale invariant.
            # But keeping magnitude stable helps.
            # Let's normalize to sum of squares = 1?
            # If we normalize, we change the gradient direction?
            # No, we can just normalize after update.
            norm = np.linalg.norm(h)
            if norm > 0:
                h = h / norm
            else:
                break

        if time.time() > deadline:
            break

    if best_h is None:
        return [1.0] * N
        
    return list(best_h)
