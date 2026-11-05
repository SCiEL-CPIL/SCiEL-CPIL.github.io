import numpy as np

N = 400

def construct():
    best_h = np.ones(N)
    best_val = np.inf
    
    def compute_val(h):
        S = np.sum(h)
        if np.abs(S) < 1e-9:
            return np.inf
        C = np.convolve(h, h)
        M = np.max(C)
        return abs(2 * N * M / (S * S))

    # Try several random initializations
    for _ in range(20):
        # Random initialization
        h = np.random.randn(N)
        
        # Ensure sum is not zero and normalize to sum=1
        S = np.sum(h)
        if np.abs(S) < 1e-9:
            h[0] = 1
            S = np.sum(h)
        h = h / S
        
        # Optimize using projected gradient descent on sum(C^2)
        eta = 0.05
        for step in range(1000):
            # Compute autocorrelation
            C = np.convolve(h, h)
            
            # Calculate gradient of sum(C^2) with respect to h
            # grad_j = 4 * sum_l h_l C_{j+l}
            grad = np.zeros(N)
            for j in range(N):
                max_l = min(N, 2*N - 1 - j)
                if max_l > 0:
                    grad[j] = 4.0 * np.dot(h[:max_l], C[j:j+max_l])
            
            # Project gradient to sum 0 to maintain constraint sum(h)=1
            grad = grad - np.mean(grad)
            
            h_new = h - eta * grad
            
            # Renormalize
            S_new = np.sum(h_new)
            if np.abs(S_new) < 1e-9:
                break
            h_new = h_new / S_new
            
            # Simple line search / step size adjustment
            val_new = abs(2 * N * np.max(np.convolve(h_new, h_new)))
            val_old = abs(2 * N * np.max(C))
            
            if val_new < val_old:
                h = h_new
            else:
                eta *= 0.5
                if eta < 1e-6:
                    break
                    
        # Evaluate
        val = compute_val(h)
        
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
    return best_h
