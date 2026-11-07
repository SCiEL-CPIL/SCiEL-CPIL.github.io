# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Return an edge-attractiveness matrix for ant colony orienteering.
    
    Strategy: Monte Carlo Efficiency-Greedy Heuristic.
    We simulate many paths starting from node 0. 
    At each step, we choose the next node that maximizes the "prize efficiency"
    (prize / total additional distance to go there and return to start).
    This highlights edges that lead to high-value nodes with low cost.
    """
    n = len(prizes)
    eps = 1e-9
    
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    # Initialize score matrix
    edge_scores = np.zeros((n, n), dtype=float)
    
    # Number of greedy simulations
    num_sims = 50
    
    # For each simulation
    for _ in range(num_sims):
        visited = [False] * n
        visited[0] = True
        current = 0
        path_len = 0.0
        
        while True:
            # Find feasible next nodes
            feasible = []
            for j in range(n):
                if visited[j]:
                    continue
                # Check if going to j and returning to 0 is within max_len
                new_len = path_len + distance_matrix[current, j] + distance_matrix[j, 0]
                if new_len <= max_len:
                    feasible.append(j)
            
            if not feasible:
                break
            
            # Efficiency-Greedy selection: maximize prizes[j] / (d[current, j] + d[j, 0])
            # Add a small randomization to avoid determinism:
            # Pick the best, but with 10% chance pick the second best if exists?
            # Or just pure efficiency.
            
            best_j = None
            best_score = -1.0
            second_best_j = None
            second_best_score = -1.0
            
            for j in feasible:
                cost = distance_matrix[current, j] + distance_matrix[j, 0]
                if cost < eps:
                    score = prizes[j] / eps
                else:
                    score = prizes[j] / cost
                if score > best_score:
                    if best_j is not None:
                        second_best_j = best_j
                        second_best_score = best_score
                    best_j = j
                    best_score = score
                elif score > second_best_score:
                    second_best_j = j
                    second_best_score = score
            
            # Randomization: 80% best, 20% second best (if exists)
            import random
            if second_best_j is not None and random.random() < 0.2:
                chosen_j = second_best_j
            else:
                chosen_j = best_j
            
            # Accumulate score
            # Use the efficiency score as the weight? Or just 1?
            # Using the score reinforces good choices.
            edge_scores[current, chosen_j] += best_score if chosen_j == best_j else second_best_score
            
            # Update state
            visited[chosen_j] = True
            path_len = path_len + distance_matrix[current, chosen_j]
            current = chosen_j
            
            # Optimization: If we've visited all nodes, break
            if all(visited):
                break
    
    # Convert scores to heuristic
    heuristic = edge_scores + 1.0
    
    # Ensure positivity and finiteness
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, eps)
    
    # Normalize to have mean 1
    mean_val = np.mean(heuristic)
    if mean_val > eps:
        heuristic = heuristic / mean_val
    
    return heuristic
# EVOLVE-BLOCK-END
