"""Combined TSP constructive heuristic: Reference-tour guidance with local fallback.

Strategy:
  1. On detecting a new instance, build a high-quality reference tour using
     Multi-Start Nearest-Neighbor (all starts) followed by 2-Opt to convergence.
  2. Pre-compute the successor for each node in that reference tour.
  3. During construction, strongly prefer following the reference tour's successor
     when it is still unvisited. If not available, fall back to a local scoring rule
     combining distance to current, distance to destination, and a small lookahead.
  4. For small remaining sets (<=8), use exact brute force to optimize the tail.
  5. Deadline guard: if time is running out, fall back to nearest neighbor.
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
        # Find nearest unvisited
        dists = distance_matrix[current].copy()
        dists[visited] = np.inf
        next_node = int(np.argmin(dists))
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    
    return tour


def _two_opt(distance_matrix, tour):
    """Improve a tour using 2-opt until no improvement is found."""
    n = len(tour)
    if n < 4:
        return tour
    
    tour = list(tour)
    improved = True
    
    while improved and not _is_deadline_passed():
        improved = False
        for i in range(n - 2):
            if _is_deadline_passed():
                return tour
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                # Calculate delta
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                delta = distance_matrix[a][c] + distance_matrix[b][d] - distance_matrix[a][b] - distance_matrix[c][d]
                if delta < -1e-10:
                    # Reverse segment
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
    return tour


def _build_reference_tour(distance_matrix):
    """
    Build a high-quality reference tour using Multi-Start NN + 2-Opt.
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
        
        # Apply 2-opt to this tour
        tour_2opt = _two_opt(distance_matrix, tour)
        tour_len_2opt = sum(distance_matrix[tour_2opt[i]][tour_2opt[(i + 1) % n]] for i in range(n))
        
        if tour_len_2opt < best_len:
            best_len = tour_len_2opt
            best_tour = tour_2opt
    
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
    """
    For small sets of unvisited nodes (<=8), use brute force to find the optimal
    ordering to visit them and return to destination.
    """
    n = len(unvisited_nodes)
    if n == 0:
        return -1
    if n == 1:
        return int(unvisited_nodes[0])
    if n > 8:
        return None  # Signal to use heuristic fallback
    
    unvisited_arr = np.asarray(unvisited_nodes)
    best_order = None
    best_len = float('inf')
    
    # Brute force all permutations
    for perm in permutations(range(n)):
        # Calculate total length: current -> perm[0] -> ... -> perm[-1] -> destination
        total = distance_matrix[current_node, unvisited_arr[perm[0]]]
        for i in range(n - 1):
            total += distance_matrix[unvisited_arr[perm[i]], unvisited_arr[perm[i + 1]]]
        total += distance_matrix[unvisited_arr[perm[-1]], destination_node]
        
        if total < best_len:
            best_len = total
            best_order = perm
    
    return int(unvisited_arr[best_order[0]])


def select_next_node(current_node, destination_node, unvisited_nodes,
                     distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Follows the pre-computed reference tour's successor when possible,
    falling back to local scoring when the reference successor is unavailable.
    """
    global _instance_initialized, _reference_successor, _instance_counter
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Detect new instance
    is_new_instance = _detect_new_instance(distance_matrix, unvisited_nodes)
    
    if is_new_instance:
        _instance_initialized = False
        _reference_successor = None
        _instance_counter += 1
        # Build reference tour
        _build_reference_tour(distance_matrix)
        _instance_initialized = True
    
    if _is_deadline_passed():
        # Fallback: pure nearest neighbor
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists_from_current)
        return int(unvisited_nodes[best_idx])
    
    # For small remaining sets, use exact optimization
    if n_unvisited <= 8:
        exact_result = _exact_tail_optimization(current_node, destination_node, 
                                                 unvisited_nodes, distance_matrix)
        if exact_result is not None:
            return exact_result
    
    # Try to follow the reference tour
    if _reference_successor is not None:
        ref_next = int(_reference_successor[current_node])
        # Check if ref_next is in unvisited_nodes
        if ref_next in unvisited_nodes:
            return ref_next
    
    # Fallback: local scoring with small lookahead
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    scores = dists_from_current.copy()
    
    # Add a small penalty for nodes that are far from the destination
    if n_unvisited > 2:
        dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
        scores += 0.1 * dists_to_dest
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
