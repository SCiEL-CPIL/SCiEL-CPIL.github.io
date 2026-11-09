import numpy as np
import time

def greedy_mis_size(adjacency, scores):
    """
    Returns the size of the independent set produced by the greedy algorithm
    using the given scores.
    """
    n = len(scores)
    # Get vertices in descending score order
    order = np.argsort(-scores)
    
    selected = np.zeros(n, dtype=bool)
    blocked = np.zeros(n, dtype=bool)
    
    count = 0
    for v in order:
        if not blocked[v]:
            selected[v] = True
            count += 1
            # Block neighbors
            neighbors = np.where(adjacency[v])[0]
            blocked[neighbors] = True
            # Also block v itself? No, v is already selected.
            # But we should ensure we don't pick v again (not possible in loop)
            # And we should mark v as blocked for future checks? 
            # The 'blocked' array tracks if a vertex is forbidden from selection.
            # Selected vertices are obviously forbidden.
            blocked[v] = True
            
    return count

def priority(adjacency):
    start_time = time.time()
    timeout = 140.0 # Safety margin
    
    adjacency = np.asarray(adjacency)
    n = adjacency.shape[0]
    
    # Initial scores: -degree
    degrees = adjacency.sum(axis=1)
    best_scores = -degrees.astype(float)
    
    # Evaluate initial
    best_size = greedy_mis_size(adjacency, best_scores)
    
    # Simulated Annealing / Local Search
    # We will perturb the scores.
    # A good perturbation is swapping the scores of two vertices.
    # Or adding small noise.
    # Swapping is discrete and maintains the relative "strength" of the degree signal somewhat.
    
    current_scores = best_scores.copy()
    current_size = best_size
    
    # Temperature parameters
    T = 1.0
    cooling = 0.99
    min_T = 0.01
    
    rng = np.random.default_rng(42)
    
    while time.time() - start_time < timeout:
        if T < min_T:
            break
            
        # Pick a random move: Swap two random indices
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        
        if i == j:
            continue
            
        # Apply move
        current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
        
        new_size = greedy_mis_size(adjacency, current_scores)
        
        delta = new_size - current_size
        
        if delta > 0:
            # Better, always accept
            current_size = new_size
            if new_size > best_size:
                best_size = new_size
                best_scores = current_scores.copy()
        else:
            # Worse, accept with probability exp(delta / T)
            # delta is negative, so exp(negative / T) < 1
            if rng.random() < np.exp(delta / T):
                current_size = new_size
            else:
                # Revert
                current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
                
        T *= cooling
        
    return best_scores
