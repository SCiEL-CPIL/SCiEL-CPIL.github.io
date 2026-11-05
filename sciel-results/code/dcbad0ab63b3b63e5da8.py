import numpy as np

N = 400

def compute_value(h):
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    c = np.convolve(h, h)
    max_c = np.max(c)
    return abs(2 * N * max_c / (s ** 2))

def construct():
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    # Random restarts
    for restart in range(20):
        if restart == 0:
            h = np.ones(N)
        else:
            # Mix of positive and negative
            h = np.random.randn(N)
            # Ensure sum is not zero
            if abs(np.sum(h)) < 1e-3:
                h += 0.1
        
        val = compute_value(h)
        
        # Coordinate Descent / Local Search
        # Iterate a few times
        for i in range(1000):
            # Pick a random index
            idx = np.random.randint(N)
            
            # Try a few small steps
            best_step_val = val
            best_step_h = h
            
            for delta in [-0.1, -0.01, 0.01, 0.1]:
                h_new = h.copy()
                h_new[idx] += delta
                val_new = compute_value(h_new)
                if val_new < best_step_val:
                    best_step_val = val_new
                    best_step_h = h_new
            
            if best_step_val < val:
                h = best_step_h
                val = best_step_val
                if val < best_val:
                    best_val = val
                    best_h = h.copy()
            else:
                # Stagnation, maybe break or continue
                pass
                
    return best_h
