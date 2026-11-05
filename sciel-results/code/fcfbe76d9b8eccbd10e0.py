import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initial guess
    # Random or structured
    best_h = np.ones(N)
    best_val = np.inf
    
    # Try a few initializations
    inits = [
        np.ones(N),
        np.random.randn(N),
        np.sign(np.random.randn(N)),
        np.cos(np.linspace(0, 10*np.pi, N))
    ]
    
    for h0 in inits:
        h = h0.copy()
        # Normalize to sum=1
        s = np.sum(h)
        if np.abs(s) < 1e-9:
            h += 1e-9
            s = np.sum(h)
        h /= s
        
        beta = 20.0
        eta = 0.05
        patience = 0
        best_local_val = np.inf
        
        while time.time() < deadline:
            c = np.convolve(h, h)
            
            # Soft max
            c_max = np.max(c)
            exp_c = np.exp(beta * (c - c_max))
            p = exp_c / np.sum(exp_c)
            
            # Gradient
            G = 2.0 * np.array([np.dot(h, p[j:j+N]) for j in range(N)])
            
            h_new = h - eta * G
            
            # Project to sum=1
            h_new = h_new - np.mean(h_new) + 1.0/N
            
            c_new = np.convolve(h_new, h_new)
            val_new = 2.0 * N * np.max(c_new)
            
            if val_new < best_local_val:
                best_local_val = val_new
                patience = 0
            else:
                patience += 1
                # Reduce learning rate if stuck
                if patience > 100:
                    eta *= 0.9
                    patience = 0
                    
            h = h_new
            
            if val_new < best_val:
                best_val = val_new
                best_h = h_new.copy()
                
    return best_h
