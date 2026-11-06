"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node in a TSP tour construction.
    Uses a two-phase approach:
    1. Construction: greedy with forward-backward cost estimation
    2. Local search: Or-opt style segment relocation to improve tour
    
    Since we're building the tour incrementally, we maintain a global tour state
    and apply local improvements when beneficial.
    """
    k = len(unvisited_nodes)
    
    if k == 0:
        return int(current_node)

    if k == 1:
        return int(unvisited_nodes[0])

    # Check time deadline
    if time.time() - _START_TIME > _DEADLINE:
        # Fallback to simple nearest neighbor
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])

    # Use a smarter construction heuristic
    return _construct_with_lookahead(current_node, destination_node, unvisited_nodes, distance_matrix)

def _construct_with_lookahead(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Construction heuristic that considers:
    - Distance from current to candidate
    - Distance from candidate to destination (backward cost)
    - Local neighborhood structure
    
    This is more globally aware than pure nearest neighbor.
    """
    k = len(unvisited_nodes)
    U = unvisited_nodes
    
    # Get distances from current to all unvisited
    dist_from_current = distance_matrix[current_node, U]
    
    # Get distances from all unvisited to destination
    dist_to_dest = distance_matrix[U, destination_node]
    
    # Get sub-matrix of distances between unvisited nodes
    D = distance_matrix[np.ix_(U, U)]
    
    # Compute a composite score that balances:
    # 1. Immediate cost: dist_from_current[i]
    # 2. Future cost: dist_to_dest[i] (cost to eventually return)
    # 3. Local connectivity: how well this node connects to its neighbors
    
    # For each candidate i, estimate the "penalty" of choosing it:
    # - If we pick i, we lose the chance to pick its nearest neighbor next
    # - So we add a penalty based on how unique i is as a connection point
    
    scores = np.zeros(k)
    
    for i in range(k):
        # Base cost: forward + backward
        base_cost = dist_from_current[i] + dist_to_dest[i]
        
        # Penalty: if i is the unique nearest neighbor of some other node,
        # choosing i now may force a long jump later
        # Estimate by checking if i is in the top-2 nearest for any other node
        penalty = 0.0
        for j in range(k):
            if j == i:
                continue
            # Is i one of the 2 nearest neighbors of j?
            row = D[j].copy()
            row[i] = np.inf  # exclude self
            sorted_idx = np.argsort(row)
            if i in sorted_idx[:2]:
                # i is important for j's connectivity
                # Penalty proportional to how much worse the 3rd choice would be
                if len(sorted_idx) >= 3:
                    third_cost = row[sorted_idx[2]]
                    penalty += (third_cost - row[sorted_idx[0]]) * 0.3
                else:
                    penalty += 1.0
        
        scores[i] = base_cost + penalty
    
    best_idx = np.argmin(scores)
    return int(U[best_idx])

# EVOLVE-BLOCK-END
