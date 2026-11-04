import numpy as np
import time

N = 400

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds margin
    
    # Initialize with random signs
    x = np.random.choice([-1, 1], size=N).astype(float)
    
    # Normalize to unit energy
    x = x / np.sqrt(np.sum(x**2))
    
    best_x = x.copy()
    best_val = np.inf
    
    # Soft-max parameter
    beta = 10.0
    
    # Gradient Descent
    lr = 0.01
    for i in range(500):
        if time.time() > deadline:
            break
            
        # Compute linear auto-correlation
        c_full = np.correlate(x, x, mode='full')
        center_idx = N - 1
        
        # Sidelobes are everything except the center
        sidelobes = np.concatenate([c_full[:center_idx], c_full[center_idx+1:]])
        
        abs_sidelobes = np.abs(sidelobes)
        
        # Soft max
        m = np.max(abs_sidelobes)
        if m == 0:
            grad = np.zeros(N)
        else:
            log_sum_exp = np.log(np.sum(np.exp(beta * (abs_sidelobes - m)))) + beta * m
            
            # Weights for soft max
            weights = np.exp(beta * (abs_sidelobes - m))
            weights /= np.sum(weights)
            
            # Construct h vector for gradient
            # h[k] for k=1..N-1 is weight * sign(c_k)
            # c_full[center_idx+1:] are lags 1..N-1
            lags_pos = c_full[center_idx+1:]
            w_pos = weights[N-1:] # weights are aligned with sidelobes: [neg, pos]
            
            h = np.zeros(N)
            h[1:] = w_pos * np.sign(lags_pos)
            # Symmetric part
            h[:0:-1] = h[1:]
            
            # Compute gradient g = sum_k h[k] (x_{n-k} + x_{n+k})
            grad = np.zeros(N)
            for k in range(1, N):
                w_val = h[k]
                if w_val != 0:
                    # x_{n-k}: x[0:N-k] placed at indices k:N
                    grad[k:N] += w_val * x[0:N-k]
                    # x_{n+k}: x[k:N] placed at indices 0:N-k
                    grad[0:N-k] += w_val * x[k:N]
        
        # Normalize gradient
        norm = np.linalg.norm(grad)
        if norm > 0:
            x -= lr * grad / norm
            
            # Renormalize x to unit energy
            x_norm = np.linalg.norm(x)
            if x_norm > 1e-6:
                x /= x_norm
            else:
                x = np.random.choice([-1, 1], size=N).astype(float)
                x /= np.sqrt(N)
        
        # Track best
        c_check = np.correlate(x, x, mode='full')
        max_side = np.max(np.abs(np.concatenate([c_check[:N-1], c_check[N:]])))
        if max_side < best_val:
            best_val = max_side
            best_x = x.copy()
            
    return best_x
