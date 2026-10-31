import numpy as np
import time

# Global deadline guard
_DEADLINE = time.time() + 110.0  # 110s safety margin within 120s limit

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick the next city in a TSP tour using a multi-criteria heuristic.
    
    Strategy: For each candidate node j in unvisited_nodes, compute a composite score
    that balances:
      1. Distance from current_node to j (greedy)
      2. Distance from j to destination_node (goal-seeking)
      3. Minimum distance from j to any other unvisited node (connectivity/lookahead)
      4. A "regret" term: how much worse j would be compared to the best alternative
    
    We use a weighted combination with weights tuned to encourage good tour structure.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return current_node
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: greedy nearest neighbor
        d_curr = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(d_curr)])
    
    # Vectorized computations
    # d_curr: (N,) distances from current to all unvisited
    d_curr = distance_matrix[current_node, unvisited_nodes]
    
    # d_dest: (N,) distances from each unvisited to destination
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # d_unvisited: (N, N) distances between unvisited nodes
    d_unvisited = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # min_next: (N,) min distance from each j to its nearest other unvisited (excluding self)
    np.fill_diagonal(d_unvisited, np.inf)
    min_next = np.min(d_unvisited, axis=1)
    
    # Second nearest neighbor for a richer lookahead
    # Replace the min with inf and take min again
    d_unvisited_copy = d_unvisited.copy()
    min_indices = np.argmin(d_unvisited_copy, axis=1)
    rows = np.arange(n_unvisited)
    d_unvisited_copy[rows, min_indices] = np.inf
    second_min_next = np.min(d_unvisited_copy, axis=1)
    
    # Regret: for each j, the difference between the best score and j's score
    # First compute a base score without regret
    # Weights: greedy, destination, connectivity, second-lookahead
    w_greedy = 1.0
    w_dest = 0.3
    w_conn = 0.2
    w_second = 0.1
    
    base_scores = (w_greedy * d_curr + 
                   w_dest * d_dest + 
                   w_conn * min_next + 
                   w_second * second_min_next)
    
    # Regret term: for each candidate j, how much more expensive it is 
    # compared to the best candidate
    # This encourages picking nodes that are clearly better
    # Actually, let's use a different approach: 
    # Compute the cost if we pick j, including the cost to go from j to the next-best node
    # This is a 2-step lookahead
    
    # For each j, the 2-step cost is: d(current, j) + min over k != j of d(j, k) + ...
    # But we also want to consider the destination
    
    # Let's try a different structural approach:
    # Use a "savings" based heuristic inspired by Clarke-Wright
    # For each pair (i, j) of unvisited nodes, the savings of connecting them 
    # is d(current, i) + d(current, j) - d(i, j)
    
    # Actually, let me try a simpler but effective approach:
    # Score = d(current, j) + alpha * d(j, dest) + beta * (min_next - second_min_next)
    # The (min_next - second_min_next) term is negative when j is a good connector
    
    # Or better: use the idea that we want to avoid creating "dangling" edges
    # A node j is "expensive" if all its neighbors in unvisited are far
    
    # Let's go with a clean weighted sum:
    scores = d_curr + 0.25 * d_dest + 0.15 * min_next
    
    # Add a small penalty for nodes that are isolated (high min_next)
    # This is already captured by the min_next term
    
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
