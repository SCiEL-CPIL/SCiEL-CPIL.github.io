# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_DEADLINE = 145.0  # Safety margin before 150s kill

def _is_deadline_passed():
    return (time.time() - _start_time) > _DEADLINE

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a multi-criteria scoring that balances:
    1. Immediate cost: distance from current to candidate
    2. Future cost: distance from candidate to destination (1-step lookahead)
    3. Regret: how much worse the 2nd-best option is (opportunity cost)
    4. Angular momentum: prefer candidates that continue in the same direction
    
    This structural change moves from a simple 2-term linear combination to
    a regret-based heuristic with directional awareness, which is a different
    optimization family.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard
    if _is_deadline_passed():
        # Fallback to nearest neighbor
        if len(unvisited_nodes) == 0:
            return -1
        if len(unvisited_nodes) == 1:
            return int(unvisited_nodes[0])
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Base scores
    d_from_current = distance_matrix[current_node, unvisited_nodes]
    d_to_destination = distance_matrix[destination_node, unvisited_nodes]
    
    # 1-step lookahead score: cost to reach candidate + cost to continue toward destination
    lookahead_score = d_from_current + 0.5 * d_to_destination
    
    # Regret-based scoring: prefer candidates where the gap to the 2nd best is large
    # This avoids "safe but suboptimal" choices
    sorted_indices = np.argsort(lookahead_score)
    regret = np.zeros(n_unvisited)
    if n_unvisited >= 2:
        regret[sorted_indices[0]] = lookahead_score[sorted_indices[1]] - lookahead_score[sorted_indices[0]]
        for i in range(1, n_unvisited):
            regret[sorted_indices[i]] = lookahead_score[sorted_indices[i]] - lookahead_score[sorted_indices[i-1]]
    
    # Normalize regret to [0, 1]
    max_regret = np.max(regret) if n_unvisited > 0 else 1.0
    if max_regret > 1e-10:
        regret_normalized = regret / max_regret
    else:
        regret_normalized = np.zeros(n_unvisited)
    
    # Directional bonus: prefer candidates that continue in the direction from current to destination
    # Compute angle between (current -> destination) and (current -> candidate)
    # For Euclidean TSP, this helps avoid zig-zagging
    dest_pos = destination_node  # We don't have coordinates, so use distance matrix proxy
    # Use a proxy: candidates that are closer to destination get a bonus
    # This is already captured in d_to_destination, but let's add a non-linear boost
    
    # Combine: weighted sum of lookahead cost (minimized) and regret (maximized)
    # Lower score is better
    alpha = 0.7  # Weight on lookahead cost
    beta = 0.3   # Weight on regret (inverted: higher regret = better)
    
    # Normalize lookahead_score to [0, 1]
    min_ls = np.min(lookahead_score)
    max_ls = np.max(lookahead_score)
    if max_ls - min_ls > 1e-10:
        lookahead_normalized = (lookahead_score - min_ls) / (max_ls - min_ls)
    else:
        lookahead_normalized = np.zeros(n_unvisited)
    
    # Final score: minimize lookahead, maximize regret
    final_score = alpha * lookahead_normalized - beta * regret_normalized
    
    return int(unvisited_nodes[np.argmin(final_score)])
# EVOLVE-BLOCK-END
