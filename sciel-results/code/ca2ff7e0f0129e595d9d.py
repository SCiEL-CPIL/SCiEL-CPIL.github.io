"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate W1 vectors (weight 1, entries in {-2,0,2}, norm^2=4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
        
    # Compute conflict matrix: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists as sorted arrays for efficient operations
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    # Precompute conflict bitmask for fast membership testing using numpy
    # conflicts is already a boolean matrix, we'll use it directly
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    
    def greedy_min_degree_fast():
        """Fast greedy independent set using heap for min-degree selection."""
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.astype(np.int32).copy()
        
        # Use a heap for efficient min-degree extraction
        # Heap entries: (degree, vertex_id)
        heap = [(int(degrees[i]), i) for i in range(N4)]
        heapq.heapify(heap)
        
        # Track removed vertices to skip stale heap entries
        removed = np.zeros(N4, dtype=bool)
        
        while heap and time.time() < deadline - 5:
            deg, v = heapq.heappop(heap)
            
            # Skip if already removed or not alive
            if removed[v] or not alive[v]:
                continue
            
            # Select this vertex
            in_set[v] = True
            removed[v] = True
            alive[v] = False
            
            # Remove all neighbors
            nbrs = neighbors[v]
            for u in nbrs:
                if alive[u]:
                    alive[u] = False
                    removed[u] = True
                    # Update degree of remaining alive vertices
                    # We don't need exact degrees for greedy, just relative ordering
                    # But we do need to update the heap for remaining vertices
                    # Actually, for min-degree greedy, we just need to remove neighbors
                    # The degree updates are for the "improved" version, but basic min-degree
                    # just picks the current minimum and removes its neighbors
            
            # Push updated degrees for vertices that lost a neighbor
            # Actually, in basic min-degree greedy, we don't update degrees of non-removed vertices
            # We just keep removing the min-degree alive vertex
            # Let's re-push alive vertices with their current degree
            # This is expensive. Let's use a simpler approach:
            # Just re-heapify periodically or use a different strategy.
            
            # For now, let's just continue - the heap may have stale entries
            # but we skip them with the removed check
        
        return in_set
    
    def greedy_min_degree_simple():
        """Simple greedy: repeatedly pick min-degree alive vertex."""
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.astype(np.int32).copy()
        
        while True:
            if not np.any(alive):
                break
            if time.time() > deadline - 5:
                break
            
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            # Find min degree among alive
            min_idx = alive_indices[np.argmin(degrees[alive_indices])]
            
            in_set[min_idx] = True
            alive[min_idx] = False
            
            # Remove neighbors
            nbrs = neighbors[min_idx]
            alive_nbrs = nbrs[alive[nbrs]]
            if len(alive_nbrs) > 0:
                alive[alive_nbrs] = False
                degrees[alive_nbrs] -= 1  # Not strictly needed for simple greedy
            
        return in_set
    
    def sa_local_search_fast(in_set_init, max_time=10.0):
        """Fast SA with precomputed conflict counts."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        T = 4.0
        cooling = 0.9993
        min_T = 0.05
        
        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        in_set_indices = np.where(in_set)[0]
        for i in in_set_indices:
            count_conflicts[neighbors[i]] += 1
        
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        
        # Precompute not_in_set for faster access
        not_in_set_mask = ~in_set
        
        while time.time() < sa_deadline:
            not_in_set_indices = np.where(not_in_set_mask)[0]
            in_set_indices = np.where(in_set)[0]
            
            if len(not_in_set_indices) == 0 and len(in_set_indices) == 0:
                break
            
            move_type = rng.random()
            
            if move_type < 0.5 and len(not_in_set_indices) > 0:
                # ADD move
                counts = count_conflicts[not_in_set_indices]
                min_count = np.min(counts)
                min_cands = not_in_set_indices[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]
                
                k = min_count
                delta = 1 - k
                
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.4
                else:
                    accept = rng.random() < np.exp(delta / T)
                
                if accept:
                    # Remove conflicting vertices from in_set
                    to_remove = np.where(conflicts[v] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        not_in_set_mask[to_remove] = True
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                    in_set[v] = True
                    not_in_set_mask[v] = False
                    current_size += delta
                    count_conflicts[neighbors[v]] += 1
                else:
                    stuck_counter += 1
                    
            elif move_type < 0.9 and len(in_set_indices) > 0 and len(not_in_set_indices) > 0:
                # SWAP move
                out_conf_counts = count_conflicts[not_in_set_indices]
                min_out_count = np.min(out_conf_counts)
                min_out_cands = not_in_set_indices[out_conf_counts == min_out_count]
                v_add = min_out_cands[rng.integers(len(min_out_cands))]
                
                v_remove = in_set_indices[rng.integers(len(in_set_indices))]
                
                conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                other_conflicts = conflicts_with[conflicts_with != v_remove]
                
                if len(other_conflicts) == 0:
                    in_set[v_remove] = False
                    not_in_set_mask[v_remove] = True
                    in_set[v_add] = True
                    not_in_set_mask[v_add] = False
                    count_conflicts[neighbors[v_remove]] -= 1
                    count_conflicts[neighbors[v_add]] += 1
                else:
                    to_remove = np.unique(np.concatenate(([v_remove], other_conflicts)))
                    delta = 1 - len(to_remove)
                    accept = delta > 0 or (rng.random() < np.exp(delta / T))
                    if accept:
                        in_set[to_remove] = False
                        not_in_set_mask[to_remove] = True
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                        in_set[v_add] = True
                        not_in_set_mask[v_add] = False
                        current_size += delta
                        count_conflicts[neighbors[v_add]] += 1
                    else:
                        stuck_counter += 1
                        
            else:
                # REMOVE move
                if len(in_set_indices) > 0:
                    v = in_set_indices[rng.integers(len(in_set_indices))]
                    delta = -1
                    accept = rng.random() < np.exp(delta / T)
                    
                    if accept:
                        in_set[v] = False
                        not_in_set_mask[v] = True
                        current_size += delta
                        count_conflicts[neighbors[v]] -= 1
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1
            
            if current_size > local_best_size:
                local_best_size = current_size
                local_best_in_set = in_set.copy()
                stuck_counter = 0
            
            if stuck_counter > 1500:
                T = max(T, 3.0)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size
    
    def polish_greedy_fast(in_set_init, max_time=5.0):
        """Fast polish using iterative vectorized conflict checking."""
        polish_set = in_set_init.copy()
        polish_size = int(polish_set.sum())
        polish_deadline = time.time() + max_time
        
        while time.time() < polish_deadline:
            not_in_mask = ~polish_set
            if not np.any(not_in_mask):
                break
                
            not_in_indices = np.where(not_in_mask)[0]
            
            # Check which candidates have no conflicts with current set
            counts = conflicts[not_in_indices] @ polish_set
            addable_mask = counts == 0
            
            if not np.any(addable_mask):
                break
                
            add_indices = not_in_indices[addable_mask]
            polish_set[add_indices] = True
            polish_size += len(add_indices)
            
            if len(add_indices) == 0:
                break
                
        return polish_set, polish_size
    
    # Multiple restarts
    n_restarts = 0
    max_restarts = 150  # More restarts since each is faster
    
    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1
        
        # 1. Greedy Init (simple version is faster)
        in_set = greedy_min_degree_simple()
        current_size = int(in_set.sum())
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 2. SA Local Search
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break
        
        sa_time = min(remaining_time * 0.4, 8.0)
        if sa_time < 1.5:
            sa_time = 1.5
            
        sa_result, sa_size = sa_local_search_fast(in_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()
        
        # 3. Polish
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 2:
            polish_time = min(remaining_time * 0.3, 4.0)
            if polish_time > 0.5:
                polish_result, polish_size = polish_greedy_fast(sa_result if sa_size >= current_size else in_set, max_time=polish_time)
                
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()
                
                # 4. Second SA on polished result
                remaining_time2 = deadline - 3 - time.time()
                if remaining_time2 > 3 and polish_size >= current_size:
                    sa2_time = min(remaining_time2 * 0.5, 5.0)
                    if sa2_time > 1:
                        sa2_result, sa2_size = sa_local_search_fast(polish_result, max_time=sa2_time)
                        if sa2_size > best_size:
                            best_size = sa2_size
                            best_in_set = sa2_result.copy()
                        
                        # Final polish
                        remaining_time3 = deadline - 2 - time.time()
                        if remaining_time3 > 1.5:
                            final_polish, final_size = polish_greedy_fast(sa2_result, max_time=min(remaining_time3 * 0.5, 3.0))
                            if final_size > best_size:
                                best_size = final_size
                                best_in_set = final_polish.copy()
    
    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
