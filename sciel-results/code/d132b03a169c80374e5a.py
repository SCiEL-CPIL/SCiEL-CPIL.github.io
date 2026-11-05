import numpy as np
import time

def priority(adjacency):
    # Ensure adjacency is a numpy array
    A = np.array(adjacency, dtype=float)
    N = A.shape[0]
    
    # Compute features
    deg = A.sum(axis=1)
    w1 = A @ deg  # Sum of degrees of neighbors
    w2 = A @ w1   # Sum of degrees of 2-hop neighbors
    
    # Define candidate score functions
    # We want to maximize the independent set size.
    # Greedy picks highest score first.
    # So we want "good" vertices to have high scores.
    # "Good" vertices are those that block few others.
    
    candidates = []
    
    # 1. Seed: -degree
    candidates.append(-deg)
    
    # 2. -w1
    candidates.append(-w1)
    
    # 3. -(deg + w1)
    candidates.append(-(deg + w1))
    
    # 4. -(deg + 0.5*w1)
    candidates.append(-(deg + 0.5 * w1))
    
    # 5. -(0.5*deg + w1)
    candidates.append(-(0.5 * deg + w1))
    
    # 6. -w2
    candidates.append(-w2)
    
    # 7. -(deg + w1 + w2)
    candidates.append(-(deg + w1 + w2))
    
    # 8. -1/(1+deg)
    candidates.append(-1.0 / (1.0 + deg))
    
    # 9. -1/(1+w1)
    candidates.append(-1.0 / (1.0 + w1))
    
    # 10. Randomized degree (simulating multiple random orders is hard here, 
    #     but we can try a few random perturbations)
    #     However, deterministic is safer.
    
    best_score_vec = None
    best_size = -1
    
    # Pre-compute neighbor masks for fast greedy simulation?
    # For N=200, simple loop is fine.
    
    def simulate_greedy(scores):
        # Add tiny jitter to break ties deterministically by index
        # This ensures unique ordering
        s = scores.copy()
        s += np.arange(N) * 1e-9
        
        # Sort indices by descending score
        order = np.argsort(s)[::-1]
        
        selected = np.zeros(N, dtype=bool)
        blocked = np.zeros(N, dtype=bool)
        
        count = 0
        for v in order:
            if not blocked[v]:
                selected[v] = True
                count += 1
                # Block neighbors
                # A[v, :] is the row of neighbors
                blocked[A[v, :] > 0] = True
        
        return count, s # Return count and the jittered scores used

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 10.0 # 10s is plenty, limit is 120s
    
    for i, scores in enumerate(candidates):
        if time.time() > deadline:
            break
            
        size, jittered_scores = simulate_greedy(scores)
        
        if size > best_size:
            best_size = size
            best_score_vec = jittered_scores

    # If no candidate was evaluated (should not happen), fallback to seed
    if best_score_vec is None:
        s = -deg.copy()
        s += np.arange(N) * 1e-9
        return s

    return best_score_vec
