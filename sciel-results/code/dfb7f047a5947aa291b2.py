import numpy as np

N = 400

def construct():
    def get_val(h):
        s = np.sum(h)
        if s == 0:
            return np.inf
        conv = np.convolve(h, h)
        m = np.max(conv)
        return abs(2 * N * m / (s * s))

    rng = np.random.default_rng(42)
    
    best_h = np.ones(N)
    best_val = get_val(best_h)
    
    # Simulated Annealing with Restarts
    num_restarts = 5
    max_iters_per_restart = 10000
    
    for r in range(num_restarts):
        # Initialization
        if r == 0:
            h = np.ones(N)
        else:
            h = rng.normal(size=N)
            s_init = np.sum(h)
            if abs(s_init) < 1e-9:
                continue
            h = h / s_init
            
        current_val = get_val(h)
        
        T = 0.5
        cooling_rate = 0.9998
        min_T = 1e-6
        
        for i in range(max_iters_per_restart):
            # Generate perturbation
            # Scale noise by current T to allow larger jumps early on
            noise_scale = 0.5 * T + 0.01
            noise = rng.normal(0, noise_scale, size=N)
            h_new = h + noise
            
            s_new = np.sum(h_new)
            if abs(s_new) < 1e-9:
                continue
            
            # Normalize to keep sum constant (scale invariance)
            h_new = h_new / s_new
            
            val_new = get_val(h_new)
            
            if val_new < best_val:
                best_val = val_new
                best_h = h_new.copy()
                
            # Acceptance
            delta = val_new - current_val
            if delta < 0 or rng.random() < np.exp(-delta / T):
                h = h_new
                current_val = val_new
                
            T *= cooling_rate
            if T < min_T:
                break

    return best_h
