"""
Combined TSP constructive heuristic: Fast Reference + Dynamic Lookahead + Exact Tail.

Strategy:
  1. On detecting a new instance, build a fast reference tour using 
     Multi-Start Nearest-Neighbor (no 2-Opt to save time).
  2. Pre-compute the successor for each node in that reference tour.
  3. During construction:
     a. If the reference successor is unvisited, follow it.
     b. If not, use a 2-step lookahead heuristic to select the next node, 
        minimizing immediate cost plus the minimum cost to the next-next node.
     c. For small remaining sets (<=8), use exact brute force to optimize the tail.
  4. Deadline guard: if time is running out, fall back to nearest neighbor.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import permutations

_start_time = time.time()
_LIMIT = 145.0  # Safety margin before 150s limit

# Per-instance state
_last_dm_checksum = None
_reference_successor = None  # array: reference_successor[i] = next node after i in ref tour
_instance_initialized = False
_instance_counter = 0


def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT


def _nn_tour(distance_matrix, start):
    """Build a tour using nearest neighbor from a given start node."""
    n = distance_matrix.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    tour = [start]
    current = start
    
    for _ in range(n - 1):
        dists = distance_matrix[current].copy()
        dists[visited] = np.inf
        next_node = int(np.argmin(dists))
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    
    return tour


def _build_reference_tour(distance_matrix):
    """
    Build a fast reference tour using Multi-Start NN.
    Returns an array where reference_successor[i] is the node visited after i in the tour.
    """
    global _reference_successor
    
    n = distance_matrix.shape[0]
    if n <= 1:
        return np.array([], dtype=int)
    if n == 2:
        return np.array([1, 0])
    
    best_tour = None
    best_len = float('inf')
    
    # Multi-start: try all starting nodes
    for start in range(n):
        if _is_deadline_passed():
            break
        
        # Build NN tour from this start
        tour = _nn_tour(distance_matrix, start)
        
        # Compute tour length
        tour_len = sum(distance_matrix[tour[i]][tour[(i + 1) % n]] for i in range(n))
        
        if tour_len < best_len:
            best_len = tour_len
            best_tour = tour
    
    if best_tour is None:
        # Fallback: simple NN from node 0
        best_tour = _nn_tour(distance_matrix, 0)
    
    # Build successor array
    successor = np.zeros(n, dtype=int)
    for i in range(n):
        successor[best_tour[i]] = best_tour[(i + 1) % n]
    
    _reference_successor = successor
    return successor


def _detect_new_instance(distance_matrix, unvisited_nodes):
    """Detect if we're starting a new instance."""
    global _last_dm_checksum
    
    n_total = distance_matrix.shape[0]
    n_unvisited = len(unvisited_nodes)
    
    # A new instance starts when all (or nearly all) nodes are unvisited
    if n_unvisited >= n_total - 1:
        # Quick checksum
        step = max(1, n_total // 10)
        checksum = np.sum(distance_matrix[::step, ::step])
        
        if _last_dm_checksum is None or abs(checksum - _last_dm_checksum) > 1e-6:
            _last_dm_checksum = checksum
            return True
    
    return False


def _exact_tail_optimization(current_node, destination_node, unvisited_nodes, distance_matrix):
    """For small remaining sets, use exact brute force to optimize the tail."""
    nodes = list(unvisited_nodes)
    best_cost = np.inf
    best_first_node = nodes[0]
    
    # Limit permutations if n is slightly larger, but 8! is 40320 which is fine.
    # If n > 8, this function should not be called.
    for perm in permutations(nodes):
        # Cost: current -> perm[0] -> ... -> perm[-1] -> destination
        cost = distance_matrix[current_node, perm[0]]
        for i in range(len(perm) - 1):
            cost += distance_matrix[perm[i], perm[i+1]]
        cost += distance_matrix[perm[-1], destination_node]
        
        if cost < best_cost:
            best_cost = cost
            best_first_node = perm[0]
            
    return int(best_first_node)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Follows the pre-computed reference tour's successor when possible,
    falling back to a 2-step lookahead scoring rule when unavailable.
    """
    global _instance_initialized, _reference_successor, _instance_counter
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # If few nodes left, use exact brute force to optimize the tail
    # 8! = 40320 permutations, which is fast.
    if n_unvisited <= 8:
        return _exact_tail_optimization(current_node, destination_node, unvisited_nodes, distance_matrix)
    
    # Detect new instance
    is_new_instance = _detect_new_instance(distance_matrix, unvisited_nodes)
    
    if is_new_instance:
        _instance_initialized = False
        _reference_successor = None
        # Build reference tour
        _build_reference_tour(distance_matrix)
        _instance_initialized = True
        _instance_counter += 1
    
    if _is_deadline_passed():
        # Fallback: pure nearest neighbor
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists_from_current)
        return int(unvisited_nodes[best_idx])
    
    # Try to follow the reference tour
    if _reference_successor is not None:
        ref_next = int(_reference_successor[current_node])
        # Check if ref_next is in unvisited_nodes
        if ref_next in unvisited_nodes:
            return ref_next
    
    # Fallback: 2-step lookahead scoring
    # Score = dist(current, c) + min_{d in unvisited, d!=c} dist(c, d)
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Vectorized 2-step lookahead
    # sub_dist is n_unvisited x n_unvisited
    sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    np.fill_diagonal(sub_dist, np.inf)
    
    # For each candidate c, find the min distance to any other unvisited node d
    min_next_dist = np.min(sub_dist, axis=1)
    
    # Combine: immediate cost + penalty for future isolation
    # We also add a small bias towards destination to prevent ending up far from dest
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    scores = dists_from_current + min_next_dist + 0.1 * dists_to_dest
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
