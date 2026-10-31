import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
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

    # Compute conflict matrix
    # Conflict if dot product > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    best_selected = np.zeros(N4, dtype=bool)
    best_count = 0
    
    rng = np.random.default_rng(42)
    
    # Run multiple greedy MIS attempts
    # Each attempt: Greedy + Local Search (Add if possible)
    max_restarts = 200 # Increase restarts to improve quality
    
    for r in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        alive = np.ones(N4, dtype=bool)
        selected_indices = []
        degrees = conflicts.sum(axis=1).astype(np.int32).copy()
        
        # Greedy MIS
        while np.any(alive):
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            # Find min degree among alive
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            selected_indices.append(chosen)
            
            # Remove chosen and its neighbors
            neighbors = np.where(conflicts[chosen])[0]
            to_remove = np.unique(np.concatenate(([chosen], neighbors)))
            
            alive[to_remove] = False
            
            # Update degrees for remaining
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
        
        # Local Search: Try to add back any node that has no neighbors in selected
        selected_mask = np.zeros(N4, dtype=bool)
        selected_mask[selected_indices] = True
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Find unselected nodes
            unselected = np.where(~selected_mask)[0]
            if len(unselected) == 0:
                break
            
            # For each unselected node, check if it has any neighbor in selected
            # conflicts[unselected] is (len(unselected), N4)
            # We want to know if any column in selected_mask is True
            # neighbor_count = (conflicts[unselected] & selected_mask).sum(axis=1)
            # This is O(N^2) per pass. 5280^2 = 28M ops. Fast in NumPy.
            
            neighbor_counts = (conflicts[unselected] & selected_mask).sum(axis=1)
            
            # Nodes with 0 neighbors in selected can be added
            addable = unselected[neighbor_counts == 0]
            
            if len(addable) > 0:
                # Add all of them?
                # Adding one might make another not addable?
                # No, if they are both addable, they have 0 neighbors in S.
                # Do they conflict with each other?
                # If u, v are addable, they are not in S.
                # If they conflict, adding both is invalid.
                # So we must add them one by one or find an independent set among them.
                # Since they are "addable", they have no neighbors in S.
                # But they might have neighbors in `addable`.
                # So we need to find an independent set in the subgraph induced by `addable`.
                # However, `addable` nodes have 0 neighbors in S.
                # They might have neighbors in `addable`.
                # Simple strategy: Add them greedily.
                
                # Sort by degree in the remaining graph?
                # Or just add one by one, updating selected_mask.
                
                for u in addable:
                    # Re-check if still addable (might have been blocked by a previously added node in this batch)
                    # Since we update selected_mask, we can just check again.
                    # But `addable` was computed with old selected_mask.
                    # Let's just pick one, add it, and restart the inner loop.
                    # This is slow.
                    
                    # Better: Pick one, add it, set improved=True, break inner loop to recompute.
                    selected_mask[u] = True
                    selected_indices.append(u)
                    improved = True
                    break # Restart the while loop to recompute neighbor_counts
                    
        count = len(selected_indices)
        if count > best_count:
            best_count = count
            best_selected = np.zeros(N4, dtype=bool)
            best_selected[selected_indices] = True

    result_v4 = vectors_v4[best_selected]
    final_vectors = np.vstack([result_v4, vectors_v1])
    return final_vectors
