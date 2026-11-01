"""Rewritten program for Kissing Number in dimension 11."""
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
    
    # Convert to neighbor lists for faster local search updates
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
        """
        Add all possible nodes to in_set, updating counts.
        Returns number of nodes added.
        """
        added = 0
        candidates = np.where((counts == 0) & (~in_set))[0]
        
        for v in candidates:
            if counts[v] == 0 and not in_set[v]:
                in_set[v] = True
                added += 1
                for u in neighbors[v]:
                    counts[u] += 1
        return added

    def local_search(in_set):
        """
        Simulated annealing with add/remove/swap moves for maximum independent set.
        """
        in_set = in_set.copy()
        counts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            counts[neighbors[i]] += 1
            
        current_size = in_set.sum()
        
        # First, greedy fill to maximize current set
        added = greedy_fill(in_set, counts)
        current_size += added
        
        # Simulated annealing parameters
        t0 = 1.0
        t_end = 0.01
        n_steps = 50000
        alpha = (t_end / t0) ** (1.0 / n_steps)
        
        temp = t0
        best_size_sa = current_size
        best_in_set_sa = in_set.copy()
        
        in_set_indices = np.where(in_set)[0]
        not_in_set_indices = np.where(~in_set)[0]
        
        for step in range(n_steps):
            if time.time() >= deadline - 5:
                break
                
            temp *= alpha
            
            # Choose a random move type
            move_type = rng.integers(0, 3)  # 0: add, 1: remove, 2: swap
            
            if move_type == 0:
                # Try to add a random node not in set
                if len(not_in_set_indices) == 0:
                    continue
                v = not_in_set_indices[rng.integers(len(not_in_set_indices))]
                if counts[v] > 0:
                    continue  # Can't add, would violate independence
                
                # Add v
                in_set[v] = True
                for u in neighbors[v]:
                    counts[u] += 1
                
                new_size = current_size + 1
                # Accept (always beneficial)
                current_size = new_size
                # Update indices
                not_in_set_indices = np.where(~in_set)[0]
                in_set_indices = np.where(in_set)[0]
                
            elif move_type == 1:
                # Try to remove a random node from set
                if len(in_set_indices) == 0:
                    continue
                v = in_set_indices[rng.integers(len(in_set_indices))]
                
                # Remove v
                in_set[v] = False
                for u in neighbors[v]:
                    counts[u] -= 1
                
                new_size = current_size - 1
                delta = -1
                
                # Accept with probability exp(delta/temp)
                if delta >= 0 or rng.random() < np.exp(delta / temp):
                    current_size = new_size
                    # Update indices
                    not_in_set_indices = np.where(~in_set)[0]
                    in_set_indices = np.where(in_set)[0]
                else:
                    # Revert
                    in_set[v] = True
                    for u in neighbors[v]:
                        counts[u] += 1
                    
            else:
                # Swap: remove one from set, add one not in set
                if len(in_set_indices) == 0 or len(not_in_set_indices) == 0:
                    continue
                
                v_remove = in_set_indices[rng.integers(len(in_set_indices))]
                v_add = not_in_set_indices[rng.integers(len(not_in_set_indices))]
                
                if v_remove == v_add:
                    continue
                
                # Check if v_add can be added after removing v_remove
                # v_add is valid if its only conflicts are with v_remove (which we're removing)
                v_add_neighbors = neighbors[v_add]
                # Count how many of v_add's neighbors are in the set (excluding v_remove)
                conflicts_after_removal = 0
                for u in v_add_neighbors:
                    if in_set[u] and u != v_remove:
                        conflicts_after_removal += 1
                        break
                
                if conflicts_after_removal > 0:
                    continue  # Can't do this swap
                
                # Perform swap
                in_set[v_remove] = False
                for u in neighbors[v_remove]:
                    counts[u] -= 1
                
                in_set[v_add] = True
                for u in neighbors[v_add]:
                    counts[u] += 1
                
                new_size = current_size  # Size unchanged
                delta = 0
                
                # Always accept size-neutral moves that maintain independence
                # But this might get stuck, so accept with some probability
                if rng.random() < 0.5 or temp > 0.1:
                    current_size = new_size
                    # Update indices
                    not_in_set_indices = np.where(~in_set)[0]
                    in_set_indices = np.where(in_set)[0]
                else:
                    # Revert
                    in_set[v_add] = False
                    for u in neighbors[v_add]:
                        counts[u] -= 1
                    
                    in_set[v_remove] = True
                    for u in neighbors[v_remove]:
                        counts[u] += 1
            
            # Track best
            if current_size > best_size_sa:
                best_size_sa = current_size
                best_in_set_sa = in_set.copy()
        
        return best_in_set_sa

    # Main Loop: Multiple Restarts
    n_restarts = 0
    max_restarts = 20
    
    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1
        
        # 1. Greedy Init
        in_set = greedy_min_degree()
        
        # 2. Local Search (Simulated Annealing)
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 2:
            break
            
        in_set = local_search(in_set)
        
        current_size = in_set.sum()
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 3. Optional: Try a random perturbation of the best set found so far
        if n_restarts > 1 and time.time() < deadline - 5:
            perturbed = best_in_set.copy()
            indices = np.where(perturbed)[0]
            if len(indices) > 2:
                # Remove 3-5 random nodes for stronger perturbation
                n_remove = min(5, len(indices) // 2)
                to_remove = rng.choice(indices, size=n_remove, replace=False)
                perturbed[to_remove] = False
            perturbed = local_search(perturbed)
            current_size = perturbed.sum()
            if current_size > best_size:
                best_size = current_size
                best_in_set = perturbed.copy()

    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    
    # Add W1 vectors (norm 2, weight 1)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors
