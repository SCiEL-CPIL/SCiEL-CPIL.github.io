"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state
_DEADLINE = None
_CACHE = {}
_START_TIME = None

def _get_instance_id(distance_matrix):
    """Create a lightweight fingerprint for the distance matrix."""
    n = distance_matrix.shape[0]
    # Use a subset of entries as a quick hash to avoid hashing the whole matrix every time
    if n > 10:
        # Take a deterministic sample of the matrix
        sample = distance_matrix[::max(1, n//10), :].tobytes()
    else:
        sample = distance_matrix.tobytes()
    return hash(sample)

def _two_opt_path(seq, distance_matrix, deadline):
    """
    Apply 2-Opt improvement to a path (not a cycle).
    seq is an array of node indices.
    The first and last elements are fixed anchors.
    Returns the optimized sequence.
    """
    n_seq = len(seq)
    if n_seq <= 3:
        return seq
    
    seq = seq.copy()
    current_cost = np.sum(distance_matrix[seq[:-1], seq[1:]])
    
    improved = True
    passes = 0
    max_passes = 1000 # Safety limit, usually converges much faster
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        
        # Check deadline
        if time.time() > deadline - 0.1:
            break
            
        for i in range(n_seq - 2):
            # Early exit if deadline is close
            if time.time() > deadline - 0.1:
                break
                
            for j in range(i + 2, n_seq - 1):
                # 2-Opt move: reverse segment seq[i+1 : j+1]
                # Edges removed: (seq[i], seq[i+1]) and (seq[j], seq[j+1])
                # Edges added: (seq[i], seq[j]) and (seq[i+1], seq[j+1])
                
                u = seq[i]
                v = seq[i+1]
                x = seq[j]
                y = seq[j+1]
                
                d_uv = distance_matrix[u, v]
                d_xy = distance_matrix[x, y]
                d_ux = distance_matrix[u, x]
                d_vy = distance_matrix[v, y]
                
                delta = d_ux + d_vy - d_uv - d_xy
                
                if delta < -1e-10:
                    seq[i+1:j+1] = seq[i+1:j+1][::-1]
                    current_cost += delta
                    improved = True
                    # Restart inner loop to re-evaluate from the same i with new structure?
                    # Or just break and restart outer?
                    # Breaking outer is safer to ensure we don't miss improvements caused by the reversal.
                    break
            
            if improved:
                # Restart outer loop
                break
                
    return seq

def _construct_tour(distance_matrix, deadline):
    """
    Construct a high-quality tour using Multi-Start Nearest Neighbor + 2-Opt.
    Returns the best tour found as a numpy array of node indices.
    """
    n = distance_matrix.shape[0]
    best_tour = None
    best_len = np.inf
    
    # Multi-Start NN
    # For small n (50), we can try all starting nodes.
    # Each NN construction is O(n^2). 50 * 2500 = 125,000 ops. Very fast.
    # Then 2-Opt on each.
    
    for start in range(n):
        if time.time() > deadline - 5.0:
            break
            
        # Construct NN tour starting from 'start'
        remaining = set(range(n))
        remaining.remove(start)
        tour = [start]
        curr = start
        
        while remaining:
            # Find nearest neighbor in remaining
            # Vectorized distance calculation
            dists = distance_matrix[curr, list(remaining)]
            min_idx = np.argmin(dists)
            next_node = list(remaining)[min_idx]
            tour.append(next_node)
            remaining.remove(next_node)
            curr = next_node
        
        # Convert to numpy array for 2-Opt
        tour_arr = np.array(tour)
        
        # Apply 2-Opt to the full cycle
        # Standard 2-Opt for TSP cycle
        # We can use the same logic but treat it as a cycle.
        # However, _two_opt_path is for paths. Let's write a specific 2-opt for cycles or adapt.
        # Actually, let's just use a simple 2-opt loop for the cycle here.
        
        tour_arr = _two_opt_cycle(tour_arr, distance_matrix, deadline)
        
        # Calculate length
        tour_len = np.sum(distance_matrix[tour_arr[:-1], tour_arr[1:]]) + distance_matrix[tour_arr[-1], tour_arr[0]]
        
        if tour_len < best_len:
            best_len = tour_len
            best_tour = tour_arr
            
        # Check deadline
        if time.time() > deadline - 5.0:
            break
            
    if best_tour is None:
        # Fallback: just return 0..n-1
        best_tour = np.arange(n)
        
    return best_tour

def _two_opt_cycle(tour, distance_matrix, deadline):
    """
    Apply 2-Opt to a full TSP cycle.
    """
    n = len(tour)
    if n <= 3:
        return tour
        
    tour = tour.copy()
    improved = True
    passes = 0
    max_passes = 1000
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        
        if time.time() > deadline - 0.1:
            break
            
        for i in range(n - 1):
            if time.time() > deadline - 0.1:
                break
                
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                    
                u = tour[i]
                v = tour[i+1]
                x = tour[j]
                y = tour[(j+1) % n]
                
                d_uv = distance_matrix[u, v]
                d_xy = distance_matrix[x, y]
                d_ux = distance_matrix[u, x]
                d_vy = distance_matrix[v, y]
                
                delta = d_ux + d_vy - d_uv - d_xy
                
                if delta < -1e-10:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    improved = True
                    break
            
            if improved:
                break
                
    return tour

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _DEADLINE, _CACHE, _START_TIME
    
    if _START_TIME is None:
        _START_TIME = time.time()
        _DEADLINE = _START_TIME + 145.0  # 5s safety margin before 150s kill
        
    # Deadline guard: fall back to nearest neighbor if time is up
    if time.time() > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if n_unvisited == 2:
        # If only two left, pick the one that creates the shorter edge from current
        d0 = distance_matrix[current_node, unvisited_nodes[0]]
        d1 = distance_matrix[current_node, unvisited_nodes[1]]
        d_dest_0 = distance_matrix[unvisited_nodes[0], destination_node]
        d_dest_1 = distance_matrix[unvisited_nodes[1], destination_node]
        d_cross = distance_matrix[unvisited_nodes[0], unvisited_nodes[1]]
        
        # If we pick 0: path is ...->current->0->1->dest
        # Cost contribution: dist(current,0) + dist(0,1) + dist(1,dest)
        cost_pick_0 = d0 + d_cross + d_dest_1
        
        # If we pick 1: path is ...->current->1->0->dest
        # Cost contribution: dist(current,1) + dist(1,0) + dist(0,dest)
        cost_pick_1 = d1 + d_cross + d_dest_0
        
        if cost_pick_0 <= cost_pick_1:
            return int(unvisited_nodes[0])
        else:
            return int(unvisited_nodes[1])

    # Check cache
    instance_id = _get_instance_id(distance_matrix)
    if instance_id in _CACHE:
        full_tour = _CACHE[instance_id]
    else:
        # Compute the full tour
        full_tour = _construct_tour(distance_matrix, _DEADLINE)
        _CACHE[instance_id] = full_tour
        
    # Find the next node in the full tour after current_node
    # The full tour is a cycle.
    # We need to find current_node in full_tour, and then return the next node in the tour.
    # However, we must ensure that the next node is in unvisited_nodes.
    # Since we are building the tour step-by-step, the "full tour" computed at the start
    # might not match the partial tour built so far if the heuristic was different.
    # BUT, the strategy is:
    # 1. Compute a high-quality full tour ONCE at the beginning (when unvisited is large).
    # 2. At each step, find where current_node is in this pre-computed tour.
    # 3. The next node in the pre-computed tour should be our candidate.
    # 4. If that candidate is in unvisited_nodes, return it.
    # 5. If not, it means the pre-computed tour diverged from the actual path taken?
    #    No, the evaluator calls select_next_node sequentially.
    #    Step 1: current=start, unvisited=all others. We compute tour. Return next.
    #    Step 2: current=next_from_step_1, unvisited=rest.
    #    The pre-computed tour is fixed.
    #    If we always follow the pre-computed tour, we will visit nodes in that order.
    #    So the next node in the pre-computed tour after current_node SHOULD be in unvisited_nodes.
    #    Unless there's a bug or the tour wraps around incorrectly.
    
    # Find index of current_node in full_tour
    try:
        idx = np.where(full_tour == current_node)[0][0]
        next_node = full_tour[(idx + 1) % len(full_tour)]
    except IndexError:
        # Fallback
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # Verify next_node is in unvisited_nodes
    if next_node in unvisited_nodes:
        return int(next_node)
    else:
        # This case should not happen if we are strictly following the pre-computed tour.
        # But if it does, fallback to nearest neighbor among unvisited.
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

# EVOLVE-BLOCK-END
