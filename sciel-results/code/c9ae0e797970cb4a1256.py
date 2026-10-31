"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = None
_TIME_BUDGET = 140.0  # Safety margin: 150s limit, leave 10s for overhead/termination

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _TIME_BUDGET
    return _DEADLINE

def _two_opt_improve(seq, dist_matrix, deadline, max_passes=100):
    """
    Perform 2-Opt improvement on the sequence 'seq'.
    'seq' is a numpy array of node indices.
    Returns the improved sequence.
    """
    n = len(seq)
    if n < 4:
        return seq
    
    # Copy the sequence to avoid modifying the original
    best_seq = seq.copy()
    
    improved = True
    passes = 0
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        
        # Check deadline
        if time.time() > deadline - 2.0:
            break
            
        for i in range(n - 2):
            if time.time() > deadline - 2.0:
                break
            for j in range(i + 1, n - 1):
                # Calculate delta
                d_ui = dist_matrix[best_seq[i], best_seq[i+1]]
                d_xj = dist_matrix[best_seq[j], best_seq[j+1]]
                d_ux = dist_matrix[best_seq[i], best_seq[j+1]]
                d_vy = dist_matrix[best_seq[i+1], best_seq[j]]
                
                delta = d_ux + d_vy - d_ui - d_xj
                
                if delta < -1e-10:
                    # Perform reversal
                    best_seq[i+1:j+1] = best_seq[i+1:j+1][::-1]
                    improved = True
                    
    return best_seq

def _nearest_neighbor_path(start, end, unvisited, dist_matrix):
    """Construct a path using nearest neighbor heuristic."""
    remaining = list(unvisited)
    order = []
    curr = start
    
    while remaining:
        dists = dist_matrix[curr, remaining]
        min_idx = np.argmin(dists)
        next_node = remaining[min_idx]
        order.append(next_node)
        remaining.pop(min_idx)
        curr = next_node
        
    return np.array(order, dtype=np.int64)

def _regret_based_order(start, end, unvisited, dist_matrix):
    """
    Construct a path using regret-based heuristic.
    For each step, select the node that minimizes the regret:
    regret(node) = second_best_cost - best_cost
    This balances local proximity with global structure.
    """
    remaining = list(unvisited)
    order = []
    curr = start
    
    while remaining:
        if len(remaining) == 1:
            order.append(remaining[0])
            break
            
        # Calculate distances from current node to all remaining
        dists = dist_matrix[curr, remaining]
        
        # Sort remaining nodes by distance from current
        sorted_indices = np.argsort(dists)
        sorted_nodes = [remaining[i] for i in sorted_indices]
        sorted_dists = dists[sorted_indices]
        
        # Calculate regret for each node
        # Regret = (distance from second-best to node) - (distance from best to node)
        # We want to pick nodes where the "cost of waiting" is high
        
        regrets = np.zeros(len(remaining))
        
        # For the best node (closest to current), regret is 0
        # For others, regret is how much worse they are compared to best
        # Actually, a better regret definition:
        # For each node, consider the cost if we pick it now vs. if we pick the best node now
        # Simpler: just use distance to destination as a tie-breaker/weight
        
        # Alternative: use a combination of distance from current and distance to destination
        # score = alpha * dist(current, node) + (1-alpha) * dist(node, destination)
        
        alpha = 0.7
        dest_dists = dist_matrix[remaining, end]
        scores = alpha * dists + (1 - alpha) * dest_dists
        
        min_idx = np.argmin(scores)
        next_node = remaining[min_idx]
        order.append(next_node)
        remaining.pop(min_idx)
        curr = next_node
        
    return np.array(order, dtype=np.int64)

def _path_cost_open(seq, dist_matrix):
    """Calculate cost of an open path (no return to start)."""
    if len(seq) < 2:
        return 0.0
    return np.sum(dist_matrix[seq[:-1], seq[1:]])

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a hybrid strategy:
    1. Generate multiple candidate orderings (NN, regret-based).
    2. Refine the best candidate with 2-Opt.
    3. Return the first node in the refined path.
    """
    deadline = _get_deadline()
    
    # Fast fallback if time is critical
    if time.time() > deadline - 5.0:
        if len(unvisited_nodes) == 0:
            return -1
        if len(unvisited_nodes) == 1:
            return int(unvisited_nodes[0])
        # Nearest Neighbor fallback
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
        
    if n_unvisited == 2:
        # Exact solution for 2 nodes
        u0 = unvisited_nodes[0]
        u1 = unvisited_nodes[1]
        cost_0_1 = distance_matrix[current_node, u0] + distance_matrix[u0, u1] + distance_matrix[u1, destination_node]
        cost_1_0 = distance_matrix[current_node, u1] + distance_matrix[u1, u0] + distance_matrix[u0, destination_node]
        if cost_0_1 <= cost_1_0:
            return int(u0)
        else:
            return int(u1)

    # Generate candidate orderings
    candidates = []
    
    # Candidate 1: Nearest Neighbor
    nn_order = _nearest_neighbor_path(current_node, destination_node, unvisited_nodes, distance_matrix)
    candidates.append(nn_order)
    
    # Candidate 2: Regret-based
    if time.time() < deadline - 10.0:
        regret_order = _regret_based_order(current_node, destination_node, unvisited_nodes, distance_matrix)
        candidates.append(regret_order)
    
    # Evaluate candidates and pick the best initial path
    best_seq = None
    best_cost = float('inf')
    
    for order in candidates:
        seq = np.array([current_node] + list(order) + [destination_node], dtype=np.int64)
        cost = _path_cost_open(seq, distance_matrix)
        if cost < best_cost:
            best_cost = cost
            best_seq = seq
    
    if best_seq is None:
        # Fallback
        return int(unvisited_nodes[0])
    
    # Refine with 2-Opt
    optimized_seq = _two_opt_improve(best_seq, distance_matrix, deadline)
    
    # The next node is the second element in the sequence (index 1)
    next_node_idx = 1
    if next_node_idx < len(optimized_seq):
        return int(optimized_seq[next_node_idx])
    else:
        # Fallback
        return int(unvisited_nodes[0])

# EVOLVE-BLOCK-END
