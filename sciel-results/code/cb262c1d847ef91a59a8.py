"""Combined program for Kissing Number in dimension 11."""
import numpy as np
import itertools
import time

D = 11

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
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    N4 = len(vectors_v4)
    
    # Compute conflict matrix: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Convert to neighbor lists
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_in_set = np.zeros(N4, dtype=bool)
    best_size = 0

    def greedy_min_degree():
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set

    def greedy_fill(in_set, counts):
        added = 0
        # Repeatedly find nodes with count 0 and not in set
        # Since adding a node only increases counts of neighbors, 
        # a node that is blocked remains blocked.
        # So a single pass over all nodes is sufficient?
        # No, if A is added, B (neighbor of A) is blocked.
        # But C (not neighbor of A) might be unblocked? No, counts only increase.
        # So if counts[v] > 0, it stays > 0.
        # If counts[v] == 0, we add it.
        # So we just need to iterate through all v where counts[v]==0 and ~in_set[v].
        # Order doesn't matter for final set size? 
        # If A and B are both candidates (count 0), and A, B are neighbors.
        # If we add A first, B becomes blocked.
        # If we add B first, A becomes blocked.
        # So order matters for WHICH nodes are added, but not necessarily the count?
        # Actually, it can affect the count.
        # But greedy fill is usually done in a fixed order or random order.
        # For deterministic behavior in a pass, we use fixed order.
        
        candidates = np.where((counts == 0) & (~in_set))[0]
        
        for v in candidates:
            if counts[v] == 0 and not in_set[v]:
                in_set[v] = True
                added += 1
                for u in neighbors[v]:
                    counts[u] += 1
        return added

    def local_search(in_set):
        in_set = in_set.copy()
        counts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            counts[neighbors[i]] += 1
            
        current_size = in_set.sum()
        
        # Initial fill
        added = greedy_fill(in_set, counts)
        current_size += added
        
        while time.time() < deadline - 10:
            improved = False
            
            # 1. Single node removal + fill
            in_set_indices = np.where(in_set)[0]
            rng.shuffle(in_set_indices)
            
            for v in in_set_indices:
                if not in_set[v]:
                    continue
                
                # Remove v
                in_set[v] = False
                for u in neighbors[v]:
                    counts[u] -= 1
                    
                # Fill
                added_nodes = []
                candidates = np.where((counts == 0) & (~in_set))[0]
                for u in candidates:
                    if counts[u] == 0 and not in_set[u]:
                        in_set[u] = True
                        added_nodes.append(u)
                        for w in neighbors[u]:
                            counts[w] += 1
                            
                new_size = in_set.sum()
                if new_size > current_size:
                    current_size = new_size
                    improved = True
                else:
                    # Revert
                    for u in added_nodes:
                        in_set[u] = False
                        for w in neighbors[u]:
                            counts[w] -= 1
                    in_set[v] = True
                    for u in neighbors[v]:
                        counts[u] += 1
                    
            if improved:
                continue # Restart the loop to try more improvements

            # 2. Double node removal + fill (if single failed)
            # Try removing 2 random nodes and filling
            if time.time() < deadline - 10:
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) >= 2:
                    to_remove = rng.choice(in_set_indices, size=2, replace=False)
                    saved_in_set = in_set.copy()
                    saved_counts = counts.copy()
                    
                    for v in to_remove:
                        in_set[v] = False
                        for u in neighbors[v]:
                            counts[u] -= 1
                            
                    # Fill
                    added_nodes = []
                    candidates = np.where((counts == 0) & (~in_set))[0]
                    for u in candidates:
                        if counts[u] == 0 and not in_set[u]:
                            in_set[u] = True
                            added_nodes.append(u)
                            for w in neighbors[u]:
                                counts[w] += 1
                                
                    new_size = in_set.sum()
                    if new_size > current_size:
                        current_size = new_size
                        improved = True
                    else:
                        # Revert
                        in_set[:] = saved_in_set
                        counts[:] = saved_counts

            if not improved:
                # Perturbation: Remove 3-5 random nodes and fill
                if time.time() < deadline - 10:
                    in_set_indices = np.where(in_set)[0]
                    if len(in_set_indices) > 5:
                        to_remove = rng.choice(in_set_indices, size=rng.integers(3, 6), replace=False)
                        saved_in_set = in_set.copy()
                        saved_counts = counts.copy()
                        
                        for v in to_remove:
                            in_set[v] = False
                            for u in neighbors[v]:
                                counts[u] -= 1
                                
                        # Fill
                        added_nodes = []
                        candidates = np.where((counts == 0) & (~in_set))[0]
                        for u in candidates:
                            if counts[u] == 0 and not in_set[u]:
                                in_set[u] = True
                                added_nodes.append(u)
                                for w in neighbors[u]:
                                    counts[w] += 1
                                    
                        new_size = in_set.sum()
                        if new_size > current_size:
                            current_size = new_size
                            improved = True
                        else:
                            in_set[:] = saved_in_set
                            counts[:] = saved_counts

                if not improved:
                    break
                    
        return in_set

    # Main Loop
    n_restarts = 0
    max_restarts = 100
    
    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1
        
        # 1. Greedy Init
        in_set = greedy_min_degree()
        
        # 2. Local Search
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 2:
            break
            
        in_set = local_search(in_set)
        
        current_size = in_set.sum()
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 3. Perturb best
        if n_restarts > 1 and time.time() < deadline - 5:
            perturbed = best_in_set.copy()
            indices = np.where(perturbed)[0]
            if len(indices) > 3:
                to_remove = rng.choice(indices, size=3, replace=False)
                perturbed[to_remove] = False
            perturbed = local_search(perturbed)
            current_size = perturbed.sum()
            if current_size > best_size:
                best_size = current_size
                best_in_set = perturbed.copy()

    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    
    # Add W1 vectors
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors
