"""Rewritten program for kissing number in dimension 11."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115

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

    # Compute conflict matrix for W4 vectors
    # Conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbors for fast updates
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)

    rng = np.random.default_rng(42)

    def greedy_init():
        """Greedy initialization for SA using min-degree heuristic."""
        x = np.zeros(N4, dtype=bool)
        degrees = neighbor_counts.copy().astype(np.int32)
        alive = np.ones(N4, dtype=bool)
        
        while np.any(alive):
            alive_idx = np.where(alive)[0]
            if len(alive_idx) == 0: break
            min_deg = np.min(degrees[alive_idx])
            cands = alive_idx[degrees[alive_idx] == min_deg]
            i = cands[rng.integers(len(cands))]
            
            x[i] = True
            # Remove i and its neighbors from alive
            to_remove = np.unique(np.concatenate(([i], neighbors[i])))
            alive[to_remove] = False
            
            # Update degrees for remaining alive nodes
            rem = np.where(alive)[0]
            if len(rem) > 0:
                # Subtract 1 for each neighbor of i that is in rem
                # conflicts[rem, i] is a boolean vector indicating if rem[j] is neighbor of i
                deg_update = conflicts[rem, i].astype(np.int32)
                degrees[rem] -= deg_update
                
        return x

    def repair(x_res):
        """Iterative repair: remove vertex with highest conflict degree in current set."""
        x_final = x_res.copy()
        active_idx = np.where(x_final)[0]
        
        # Repeat until no conflicts
        changed = True
        while changed:
            changed = False
            if len(active_idx) == 0:
                break
            
            # Calculate conflict count for each active vertex within the active set
            # We can do this by checking neighbors
            # For efficiency, if set is small, just loop
            # active_idx is list of indices
            # Create a set for O(1) lookup
            active_set = set(active_idx.tolist())
            
            max_conf = -1
            victim = -1
            
            for i in active_idx:
                # Count neighbors of i that are in active_set
                nbrs = neighbors[i]
                count = 0
                for n in nbrs:
                    if n in active_set:
                        count += 1
                if count > max_conf:
                    max_conf = count
                    victim = i
            
            if max_conf > 0:
                # Remove victim
                x_final[victim] = False
                active_idx = np.array([i for i in active_idx if i != victim])
                active_set.remove(victim)
                changed = True
        
        return x_final

    def soft_sa(init_x, max_time, lambda_pen=2.0):
        """
        Soft-constraint Simulated Annealing with 1-flips and 2-swaps.
        Energy E = -size + lambda * num_conflicts
        """
        x = init_x.copy()
        # conflict_counts[i] = number of neighbors of i that are in x
        conflict_counts = np.zeros(N4, dtype=np.int32)
        for i in np.where(x)[0]:
            conflict_counts[neighbors[i]] += 1
        
        # Calculate initial energy
        size = x.sum()
        num_conflicts = np.sum(conflict_counts) // 2
        current_E = -size + lambda_pen * num_conflicts
        
        best_E = current_E
        best_x = x.copy()
        
        T = 5.0
        T_min = 0.01
        cooling = 0.995
        
        sa_deadline = time.time() + max_time
        
        # Precompute list of all vertices for random selection
        all_vertices = np.arange(N4)
        
        while time.time() < sa_deadline:
            # Choose move type: 0 for flip, 1 for swap
            # Swaps are more effective for MIS
            move_type = rng.integers(2)
            
            if move_type == 0:
                # Flip move
                i = rng.integers(N4)
                if x[i]:
                    # Remove i
                    delta_E = 1 - lambda_pen * conflict_counts[i]
                    if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                        x[i] = False
                        conflict_counts[neighbors[i]] -= 1
                        current_E += delta_E
                        if current_E < best_E:
                            best_E = current_E
                            best_x = x.copy()
                else:
                    # Add i
                    delta_E = -1 + lambda_pen * conflict_counts[i]
                    if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                        x[i] = True
                        conflict_counts[neighbors[i]] += 1
                        current_E += delta_E
                        if current_E < best_E:
                            best_E = current_E
                            best_x = x.copy()
            else:
                # Swap move: Remove i (in set), Add j (not in set)
                # Pick random i in set
                in_set = np.where(x)[0]
                if len(in_set) == 0:
                    continue
                i = in_set[rng.integers(len(in_set))]
                
                # Pick random j not in set
                # To avoid bias, pick random j from all, if in set, pick another?
                # Or just pick random j from all, if x[j] is True, skip or retry.
                # Efficient way: pick random j from all. If x[j] is True, we can't swap (or it's a flip).
                # Let's just pick random j from all.
                j = rng.integers(N4)
                if x[j]:
                    # If j is in set, this is not a valid swap (i and j both in set).
                    # We can treat it as a flip of i? No, let's just skip or do a flip.
                    # For simplicity, if j is in set, do a flip of i.
                    delta_E = 1 - lambda_pen * conflict_counts[i]
                    if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                        x[i] = False
                        conflict_counts[neighbors[i]] -= 1
                        current_E += delta_E
                        if current_E < best_E:
                            best_E = current_E
                            best_x = x.copy()
                    continue

                # Valid swap: i in set, j not in set
                # Delta size = 0
                # Delta conflicts:
                # i removed: conflicts decrease by conflict_counts[i]
                # j added: conflicts increase by conflict_counts[j]
                # BUT: if i and j are neighbors, the interaction changes.
                # Current conflicts involving i: conflict_counts[i] (neighbors in set)
                # Current conflicts involving j: conflict_counts[j] (neighbors in set)
                # If i and j are neighbors:
                #   Before: i contributes 1 to j's count, j contributes 1 to i's count.
                #   After: i removed, j added.
                #   The edge (i,j) was a conflict. It is removed from i's count (since i is gone) and from j's count (since i is gone).
                #   Wait, conflict_counts[i] counts neighbors of i in x. Since j is NOT in x, j is NOT in conflict_counts[i].
                #   conflict_counts[j] counts neighbors of j in x. Since i IS in x, i IS in conflict_counts[j].
                #   So when we remove i, conflict_counts[j] decreases by 1 (if they are neighbors).
                #   When we add j, conflict_counts[i] would increase by 1 (if they are neighbors), but i is removed.
                
                # Let's calculate delta_conflicts precisely.
                # Old conflicts involving i or j:
                # 1. Edges from i to other nodes in x (excluding j since j not in x): conflict_counts[i]
                # 2. Edges from j to other nodes in x (including i since i in x): conflict_counts[j]
                # Note: The edge (i,j) is counted in conflict_counts[j] (as i is neighbor of j and in x).
                # It is NOT counted in conflict_counts[i] (as j is not in x).
                
                # New state: i removed, j added.
                # 1. Edges from j to other nodes in x (excluding i since i removed): 
                #    conflict_counts[j] - (1 if i is neighbor of j else 0)
                # 2. Edges from i to other nodes in x: 0 (i removed)
                
                # Total old conflicts involving i or j = conflict_counts[i] + conflict_counts[j]
                # Total new conflicts involving i or j = conflict_counts[j] - (1 if i~j else 0)
                
                # Wait, are there other changes?
                # Neighbors of i (other than j) lose i from their count.
                # Neighbors of j (other than i) gain j from their count.
                # This doesn't change the TOTAL number of conflicts, only the distribution.
                # The total number of conflicts changes by:
                # - (conflicts of i in old set) + (conflicts of j in new set)
                # Conflicts of i in old set = conflict_counts[i]
                # Conflicts of j in new set = (neighbors of j in x \ {i})
                # = conflict_counts[j] - (1 if i~j else 0)
                
                # So delta_conflicts = (conflict_counts[j] - (1 if i~j else 0)) - conflict_counts[i]
                
                is_neighbor = conflicts[i, j]
                delta_conf = conflict_counts[j] - conflict_counts[i] - (1 if is_neighbor else 0)
                
                delta_E = lambda_pen * delta_conf # Size change is 0
                
                if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                    # Apply swap
                    # Remove i
                    conflict_counts[neighbors[i]] -= 1
                    x[i] = False
                    # Add j
                    conflict_counts[neighbors[j]] += 1
                    x[j] = True
                    
                    current_E += delta_E
                    if current_E < best_E:
                        best_E = current_E
                        best_x = x.copy()
            
            T *= cooling
            if T < T_min:
                T = T_min

        return best_x

    best_size = 0
    best_x = np.zeros(N4, dtype=bool)

    n_restarts = 0
    max_restarts = 500
    
    while time.time() < deadline - 5 and n_restarts < max_restarts:
        n_restarts += 1
        
        # Initialize
        x_init = greedy_init()
        
        # Run SA
        # Adjust time per restart based on remaining time
        remaining = deadline - time.time() - 5
        if remaining <= 0: break
        sa_time = min(0.5, remaining / (max_restarts - n_restarts + 1) * 1.5) # Adaptive
        if sa_time < 0.1: sa_time = 0.1
        
        x_res = soft_sa(x_init, sa_time)
        
        # Repair to ensure independence
        x_final = repair(x_res)
        
        size_final = x_final.sum()
        if size_final > best_size:
            best_size = size_final
            best_x = x_final.copy()
        
        # Optional: Local search polish on best_x
        # If time permits, do a quick hill climb on best_x
        if time.time() < deadline - 10 and n_restarts % 10 == 0:
            # Simple hill climb: try to add any vertex that doesn't conflict
            # and remove any vertex that conflicts with >1 others?
            # Let's just rely on SA restarts for now.
            pass

    final_v4_indices = np.where(best_x)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])

    return final_vectors

# EVOLVE-BLOCK-END
