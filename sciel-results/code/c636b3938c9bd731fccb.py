import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial guess: Asymmetric 2-point distribution
    # Based on analysis, a ~ 0.65, b ~ 0.35
    h = np.zeros(N)
    h[0] = 0.65
    h[1] = 0.35
    
    best_h = h.copy()
    best_score = -1.0
    
    def score(h):
        # Normalize to L1 = 1
        s = np.sum(h)
        if s == 0:
            return 0.0
        h_norm = h / s
        l2_sq = np.sum(h_norm**2)
        conv = np.convolve(h_norm, h_norm, mode='full')
        max_conv = np.max(conv)
        if max_conv == 0:
            return 0.0
        return l2_sq / max_conv

    best_score = score(best_h)
    
    # Local Search
    eps = 0.01
    while time.time() < deadline:
        improved = False
        # Try moving mass between existing non-zero bins
        non_zero_indices = np.nonzero(h > 1e-6)[0]
        if len(non_zero_indices) < 2:
             # Add a new bin if needed
             if len(non_zero_indices) == 1:
                 idx = non_zero_indices[0]
                 # Try adding to a random other index
                 other = np.random.choice([i for i in range(N) if i != idx], 1)[0]
                 h[other] += 0.1
                 h[idx] -= 0.1
                 if h[idx] < 0: h[idx] = 0
                 new_score = score(h)
                 if new_score > best_score:
                     best_score = new_score
                     best_h = h.copy()
                     improved = True
                 else:
                     h = best_h.copy()
        
        for _ in range(100): # Fixed number of steps per outer loop to avoid infinite loops
            if time.time() > deadline:
                break
            i = np.random.choice(N)
            j = np.random.choice(N)
            if i == j:
                continue
            # Move mass from i to j
            move = np.random.uniform(0.01, 0.1)
            if h[i] < move:
                move = h[i]
            
            h[i] -= move
            h[j] += move
            
            new_score = score(h)
            if new_score > best_score:
                best_score = new_score
                best_h = h.copy()
                improved = True
            else:
                h[i] += move
                h[j] -= move
        
        if not improved and time.time() > start_time + 60:
            # If stuck for a while, try a random restart with different structure
            # e.g. 3 points
            if np.random.rand() < 0.1:
                h = np.zeros(N)
                h[0] = 0.5
                h[1] = 0.3
                h[2] = 0.2
                h = h / np.sum(h)
                # Reset best if this is better
                new_score = score(h)
                if new_score > best_score:
                    best_score = new_score
                    best_h = h.copy()
                else:
                    h = best_h.copy()

    return list(best_h)
