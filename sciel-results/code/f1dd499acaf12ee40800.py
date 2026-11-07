# EVOLVE-BLOCK-START
import numpy as np
import time

def _construct_tour(distance_matrix, heuristic_matrix, demands, capacity):
    """
    Constructs a tour using a greedy strategy driven by the heuristic matrix.
    The heuristic matrix H[i,j] is interpreted as an attractiveness score.
    We pick the next node j maximizing H[current, j] among feasible nodes.
    If no feasible node exists, return to depot.
    """
    n = len(demands)
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    current = 0
    remaining_capacity = capacity
    total_distance = 0.0
    
    unvisited_count = n - 1
    while unvisited_count > 0:
        best_j = -1
        best_score = -np.inf
        
        # Vectorized search for best next node
        # Mask out visited and infeasible nodes
        feasible_mask = ~visited
        feasible_mask[0] = False # Never go to depot as a customer
        # Check capacity
        # demands is a numpy array
        feasible_mask[1:] = feasible_mask[1:] & (demands[1:] <= remaining_capacity)
        
        if not np.any(feasible_mask):
            # Return to depot
            total_distance += distance_matrix[current, 0]
            current = 0
            remaining_capacity = capacity
            continue
            
        # Get scores for feasible nodes
        scores = heuristic_matrix[current].copy()
        scores[~feasible_mask] = -np.inf
        
        best_j = np.argmax(scores)
        best_score = scores[best_j]
        
        if best_score == -np.inf:
            # Should be caught by the mask check above, but safety
            total_distance += distance_matrix[current, 0]
            current = 0
            remaining_capacity = capacity
            continue

        # Move to best_j
        total_distance += distance_matrix[current, best_j]
        remaining_capacity -= demands[best_j]
        visited[best_j] = True
        unvisited_count -= 1
        current = best_j
        
    # Return to depot at the end
    total_distance += distance_matrix[current, 0]
    
    return total_distance

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Generates an edge-attractiveness matrix.
    
    Strategy: Savings-based heuristic.
    The classical Clarke-Wright savings heuristic defines a savings value for merging
    two nodes i and j into a single route: S(i,j) = d(0,i) + d(0,j) - d(i,j).
    A large S(i,j) means that visiting i and j consecutively saves a lot of distance
    compared to visiting them separately from the depot.
    
    We use this savings value directly as the heuristic attractiveness:
    H[i,j] = max(S(i,j), 0) + epsilon
    
    This is a structurally different approach from distance-based heuristics (like 1/d^p).
    It explicitly encodes the "benefit" of linking two nodes, which is more aligned with
    the goal of minimizing total route distance in a VRP context.
    
    We will evaluate this heuristic using the greedy tour constructor and compare it
    against a few variations (e.g., normalized savings, savings + 1/d) to see if it
    yields a better greedy tour, which should correlate with better ACO performance.
    """
    n = distance_matrix.shape[0]
    start_time = time.time()
    deadline = start_time + 290.0 # Safety margin for 300s limit
    
    # Pre-compute safe distances
    eps = 1e-9
    safe_dist = distance_matrix + eps
    depot_dist = distance_matrix[0, :]
    
    # Candidate heuristics to test
    candidates = []
    
    # 1. Pure Savings: H[i,j] = max(S(i,j), 0) + eps
    # S(i,j) = d(0,i) + d(0,j) - d(i,j)
    savings = depot_dist[np.newaxis, :] + depot_dist[:, np.newaxis] - distance_matrix
    # Ensure positive
    savings = np.maximum(savings, 0) + eps
    candidates.append(savings)
    
    # 2. Savings + Inverse Distance: H[i,j] = max(S(i,j), 0) + eps + 1/d(i,j)
    inv_dist = 1.0 / safe_dist
    candidates.append(savings + inv_dist)
    
    # 3. Normalized Savings: H[i,j] = S(i,j) / (d(0,i) + d(0,j) + eps)
    # This normalizes the savings by the cost of serving them separately.
    norm_savings = savings / (depot_dist[np.newaxis, :] + depot_dist[:, np.newaxis] + eps)
    candidates.append(norm_savings)
    
    # 4. Savings * Inverse Distance: H[i,j] = max(S(i,j), 0) / d(i,j)
    # This might emphasize short links with high savings.
    cand4 = savings / safe_dist
    candidates.append(cand4)
    
    best_h = None
    best_cost = float('inf')
    
    for h in candidates:
        if time.time() > deadline:
            break
        
        try:
            # Ensure finite and positive
            if not np.all(np.isfinite(h)):
                continue
            h = np.where(h > 0, h, 1e-9)
            
            cost = _construct_tour(distance_matrix, h, demands, capacity)
            
            if cost < best_cost:
                best_cost = cost
                best_h = h.copy()
                
        except Exception:
            continue
    
    # If no valid heuristic was found, fallback to 1/d
    if best_h is None:
        best_h = 1.0 / safe_dist
        best_h = np.where(best_h > 0, best_h, 1e-9)
        
    # Final sanity check
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones_like(distance_matrix) * 1e-9
        
    return best_h
# EVOLVE-BLOCK-END
