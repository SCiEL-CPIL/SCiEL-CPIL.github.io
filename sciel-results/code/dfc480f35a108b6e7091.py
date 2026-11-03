import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # We suspect Linear shifts are intended to allow < 0.5
    # Objective: minimize max_k C_lin(k)
    # C_lin(k) = (2/N) * sum_{i=0}^{N-k-1} h_i (1 - h_{i+k})
    
    h = np.full(N, 0.5)
    
    # Use a local search / gradient step on h
    # Since h is in [0,1], we can project.
    
    def compute_max_overlap(h):
        max_c = 0
        for k in range(N):
            # Linear shift
            # h_shifted = h[k:]
            # h_part = h[:-k]
            if k == 0:
                c = np.sum(h * (1 - h))
            else:
                c = np.sum(h[:-k] * (1 - h[k:]))
            c *= 2.0 / N
            if c > max_c:
                max_c = c
        return max_c

    best_h = h.copy()
    best_score = compute_max_overlap(best_h)
    
    # Random restarts with binary-like initializations
    # and local optimization
    
    for restart in range(20):
        if time.time() > deadline:
            break
            
        # Initialize with random binary or mixed
        if restart % 2 == 0:
            h = np.random.choice([0, 1], N)
            # Ensure sum is N/2
            # Sort and swap?
            # Just random permutation of N/2 ones and N/2 zeros
            h = np.zeros(N)
            h[:N//2] = 1
            np.random.shuffle(h)
        else:
            h = np.random.rand(N)
            # Normalize to sum N/2
            h = h / np.sum(h) * (N/2)
            h = np.clip(h, 0, 1)
            # Re-normalize
            h = h / np.sum(h) * (N/2)
            
        # Local search: perturb and project
        current_score = compute_max_overlap(h)
        
        step_size = 0.1
        for i in range(500):
            if time.time() > deadline:
                break
                
            # Perturb
            h_pert = h + np.random.randn(N) * step_size
            h_pert = np.clip(h_pert, 0, 1)
            # Adjust sum
            diff = N/2 - np.sum(h_pert)
            h_pert += diff / N
            h_pert = np.clip(h_pert, 0, 1)
            # Final normalize
            h_pert = h_pert / np.sum(h_pert) * (N/2)
            
            new_score = compute_max_overlap(h_pert)
            
            if new_score < current_score:
                h = h_pert
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_h = h.copy()
            else:
                # Reduce step size if stuck?
                if i > 100:
                    step_size *= 0.95
                    
    return best_h
