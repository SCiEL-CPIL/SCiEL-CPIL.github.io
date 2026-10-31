"""
Multi-Start NN + 2-Opt guided TSP constructive heuristic with exact tail optimization.

Strategy:
  1. On detecting a new instance, run Multi-Start Nearest-Neighbor (all starts)
     followed by 2-Opt to convergence to build a high-quality reference tour.
  2. Pre-compute the successor for each node in that reference tour.
  3. During construction, follow the reference tour's successor when possible;
     if the current node's reference successor is already visited, fall back to
     a local scoring rule (distance + small lookahead).
  4. For small remaining sets (<=5), use exact brute force to optimize the tail.
  5. Deadline guard: if time is running out, fall back to nearest neighbor.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import permutations

_start_time = time.time()
_LIMIT = 142.0  # Safety margin before 150s limit

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
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                delta = distance_matrix[a][c] + distance_matrix[b][d] - distance_matrix[a][b] - distance_matrix[c][d]
                if delta < -1e-10:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
    return tour


def _or_opt(distance_matrix, tour):
    """Improve a tour using Or-Opt (1-3 city segment relocations)."""
    n = len(tour)
    if n < 5:
        return tour
    
    tour = list(tour)
    improved = True
    
    while improved and not _is_deadline_passed():
        improved = False
        for seg_len in [1, 2, 3]:
            for i in range(n):
                if _is_deadline_passed():
                    return tour
                if i + seg_len >= n:
                    continue
                # Segment to relocate: tour[i:i+seg_len]
                seg = tour[i:i+seg_len]
                # Try inserting after each position j (not adjacent to original)
                for j in range(n):
                    if j in range(i - 1, i + seg_len + 1):
                        continue
                    # Remove segment and insert after position j
                    new_tour = tour[:i] + tour[i+seg_len:]
                    insert_pos = j if j < i else j - seg_len
                    if insert_pos < 0 or insert_pos > len(new_tour) - seg_len:
                        continue
                    # Compute delta
                    # Original: ...a - seg[0] ... seg[-1] - b...
                    # New: ...a - c...d - seg[0]...seg[-1] - ...
                    a_idx = (i - 1) % n
                    b_idx = (i + seg_len) % n
                    # In new_tour, find neighbors at insertion point
                    # This is complex; use full recomputation for small n
                    candidate = new_tour[:insert_pos] + seg + new_tour[insert_pos:]
                    if len(candidate) != n:
                        continue
                    cand_len = sum(distance_matrix[candidate[k]][candidate[(k+1) % n]] for k in range(n))
                    curr_len = sum(distance_matrix[tour[k]][tour[(k+1) % n]] for k in range(n))
                    if cand_len < curr_len - 1e-10:
                        tour = candidate
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return tour


def _build_reference_tour(distance_matrix):
    """
    Build a high-quality reference tour using Multi-Start NN + 2-Opt + Or-Opt.
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
        
        # Apply Or-Opt after 2-opt
        tour_oropt = _or_opt(distance_matrix, tour_2opt)
        
        tour_len_oropt = sum(distance_matrix[tour_oropt[i]][tour_oropt[(i + 1) % n]] for i in range(n))
        
        if tour_len_oropt < best_len:
            best_len = tour_len_oropt
            best_tour = tour_oropt
    
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
    falling back to local scoring when the reference successor is unavailable.
    """
    global _instance_initialized, _reference_successor, _instance_counter
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # If few nodes left, use exact brute force to optimize the tail
    if n_unvisited <= 5:
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
    
    # Fallback: local scoring with lookahead
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    scores = dists_from_current.copy()
    
    # Add pull towards destination
    if n_unvisited > 2:
        dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
        scores += 0.05 * dists_to_dest
    
    # Add 2-step lookahead
    if n_unvisited > 2:
        sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        np.fill_diagonal(sub_dist, np.inf)
        min_next = np.min(sub_dist, axis=1)
        scores += 0.10 * min_next
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
