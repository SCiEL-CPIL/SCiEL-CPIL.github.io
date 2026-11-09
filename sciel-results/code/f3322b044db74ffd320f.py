import numpy as np
import time

# EVOLVE-BLOCK-START
import numpy as np
import time

def _two_opt_tour_length(tour, d):
    """
    Optimizes a given tour using 2-opt and returns the final length.
    tour: list of node indices, length n+1, tour[0] == tour[-1]
    d: distance matrix
    """
    n = len(tour) - 1
    if n <= 2:
        return sum(d[tour[i], tour[i+1]] for i in range(n))
    
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Edges (i-1, i) and (j, j+1)
                # New edges (i-1, j) and (i, j+1)
                # Note: tour indices are 0..n
                # i is index in tour (1..n-1)
                # j is index in tour (i+1..n-1)
                
                a = tour[i-1]
                b = tour[i]
                c = tour[j]
                dd = tour[j+1]
                
                # Original length contribution
                old_len = d[a, b] + d[c, dd]
                # New length contribution
                new_len = d[a, c] + d[b, dd]
                
                if new_len < old_len - 1e-10:
                    # Reverse the segment tour[i] to tour[j]
                    # tour[i...j] becomes tour[j...i]
                    tour[i:j+1] = reversed(tour[i:j+1])
                    improved = True
    
    # Calculate final length
    total = 0.0
    for i in range(n):
        total += d[tour[i], tour[i+1]]
    return total

def _evaluate_heuristic(d, h, time_limit=5.0):
    """
    Evaluates a heuristic by constructing a greedy tour and applying 2-opt.
    This is a proxy for how well the heuristic guides the ACO.
    ACO ants are stochastic, but if the heuristic strongly favors short edges,
    the greedy construction (which picks the most attractive edge) should produce
    a decent starting point for 2-opt.
    """
    start_time = time.time()
    n = d.shape[0]
    
    # Construct greedy tour based on heuristic
    # Attractiveness = pheromone * heuristic. Initially pheromone=1.
    # So attractiveness is proportional to h[i][j].
    # We want to pick the node j that maximizes h[current][j].
    
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    current = 0
    tour = [0]
    
    for _ in range(n - 1):
        # Get attractiveness of unvisited nodes
        # h[current, :]
        # Set visited to very low value (or -inf)
        # Since h should be positive, we can just set visited to 0 or -1
        # But wait, h might have 1e-9. We want to avoid visited nodes.
        # We can create a mask.
        
        # Efficient way:
        # scores = h[current, :].copy()
        # scores[visited] = -1  # Ensure they are not picked
        # next_node = np.argmax(scores)
        
        # Even faster:
        # Use argmax on masked array?
        # Let's just use a simple loop or vectorized mask.
        
        scores = h[current, :].copy()
        scores[visited] = -1
        next_node = np.argmax(scores)
        
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    
    tour.append(0)
    
    # Apply 2-opt
    # Check time
    if time.time() - start_time < time_limit:
        final_length = _two_opt_tour_length(tour, d)
    else:
        # Fallback: just return greedy length
        final_length = sum(d[tour[i], tour[i+1]] for i in range(n))
        
    return final_length

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin
    
    d = distance_matrix.copy()
    n = d.shape[0]
    
    candidates = []
    
    # 1. Inverse distance variants
    candidates.append(1.0 / (d + 1e-9))
    candidates.append(1.0 / (d**1.5 + 1e-9))
    candidates.append(1.0 / (d**2 + 1e-9))
    candidates.append(1.0 / (d**4 + 1e-9))
    
    # 2. Offset variants
    candidates.append(1.0 / (d + 0.05))
    candidates.append(1.0 / (d + 0.1))
    candidates.append(1.0 / (d + 0.2))
    
    # 3. Exponential decay
    candidates.append(np.exp(-d))
    
    # 4. Neighbor-based boosts
    h_base = 1.0 / (d + 1e-9)
    
    # Top 3
    h_top3 = h_base.copy()
    for i in range(n):
        row = d[i].copy()
        row[i] = np.inf
        top3_idx = np.argpartition(row, 3)[:3]
        h_top3[i, top3_idx] *= 10.0
    candidates.append(h_top3)
    
    # Top 5
    h_top5 = h_base.copy()
    for i in range(n):
        row = d[i].copy()
        row[i] = np.inf
        top5_idx = np.argpartition(row, 5)[:5]
        h_top5[i, top5_idx] *= 5.0
    candidates.append(h_top5)
    
    # Top 1 (Strict NN)
    h_top1 = h_base.copy()
    for i in range(n):
        row = d[i].copy()
        row[i] = np.inf
        top1_idx = np.argmin(row)
        h_top1[i, top1_idx] *= 20.0
    candidates.append(h_top1)
    
    # Evaluate candidates
    best_heuristic = candidates[0]
    best_score = float('inf')
    
    for idx, h in enumerate(candidates):
        # Check time
        if time.time() > deadline:
            break
        
        # Evaluate using 2-opt proxy
        # This is deterministic and fast.
        # We can run it once. If we have time, we could run it with different random restarts
        # for the initial greedy tour? No, greedy is deterministic.
        # But we can try different "greedy" strategies if we had more candidates.
        
        score = _evaluate_heuristic(d, h)
        
        if score < best_score:
            best_score = score
            best_heuristic = h
            
    return best_heuristic
# EVOLVE-BLOCK-END
