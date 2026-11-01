import numpy as np
import itertools
import time

def construct():
    D = 11
    deadline = time.time() + 130  # safety margin from 150s
    
    candidates = []
    
    # Type 1: Weight 1, value +/- 1
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
            
    # Type 2: Weight 1, value +/- 2
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            candidates.append(v)
            
    # Type 3: Weight 2, values +/- 1
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
                
    # Type 4: Weight 3, values +/- 1
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
            
    # Type 5: Weight 4, values +/- 1
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    norms_sq = np.sum(candidates**2, axis=1)
    max_norm_sq = 4
    
    dots = candidates @ candidates.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    conflict = (dist_sq < max_norm_sq)
    np.fill_diagonal(conflict, False)
    
    # Precompute conflict as bitsets for faster operations
    # Use numpy boolean matrix
    conflict_bool = conflict.astype(np.int8)
    
    def greedy_independent_set(conflict_mat, n, rng):
        remaining = np.ones(n, dtype=bool)
        degrees = conflict_mat.sum(axis=1).astype(np.int64).copy()
        selected = []
        
        while remaining.any():
            if time.time() > deadline:
                break
            rem_indices = np.where(remaining)[0]
            if len(rem_indices) == 0:
                break
            
            rem_degrees = degrees[rem_indices]
            min_deg = np.min(rem_degrees)
            
            min_mask = rem_degrees == min_deg
            min_indices_local = np.where(min_mask)[0]
            
            chosen_local = rng.integers(len(min_indices_local))
            chosen_global = rem_indices[chosen_local]
            
            selected.append(chosen_global)
            
            neigh_mask = conflict_mat[chosen_global]
            
            to_remove = neigh_mask | (np.zeros(n, dtype=bool))
            to_remove[chosen_global] = True
            
            actual_remove = to_remove & remaining
            
            if np.any(actual_remove):
                # Update degrees for remaining nodes
                rem_still = remaining & ~to_remove
                if np.any(rem_still):
                    deg_change = conflict_mat[actual_remove][:, rem_still].sum(axis=0)
                    degrees[rem_still] -= deg_change
                
                remaining[actual_remove] = False
        
        return selected
    
    def local_search(selected_set, conflict_mat, n, rng, max_iters=500):
        """Try to add more vertices or swap to improve."""
        selected_set = set(selected_set)
        selected_arr = np.array(sorted(selected_set), dtype=np.int64)
        
        # Build adjacency for selected
        in_selected = np.zeros(n, dtype=bool)
        in_selected[selected_arr] = True
        
        # For each non-selected node, count conflicts with selected
        # A node can be added if it has 0 conflicts with selected
        # Try swaps: remove one selected, add up to two non-selected
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            # Try adding a single non-selected node with no conflicts
            non_selected = np.where(~in_selected)[0]
            if len(non_selected) > 0:
                # Count conflicts for each non-selected with selected
                conflict_counts = conflict_mat[non_selected][:, in_selected].sum(axis=1)
                addable = np.where(conflict_counts == 0)[0]
                if len(addable) > 0:
                    idx = addable[rng.integers(len(addable))]
                    node = non_selected[idx]
                    selected_arr = np.append(selected_arr, node)
                    in_selected[node] = True
                    improved = True
                    continue
            
            # Try swap: remove one, try to add two
            if len(selected_arr) > 2:
                for _ in range(10):  # limited random swaps
                    if time.time() > deadline:
                        break
                    rem_idx = rng.integers(len(selected_arr))
                    rem_node = selected_arr[rem_idx]
                    
                    # Temporarily remove
                    temp_in_selected = in_selected.copy()
                    temp_in_selected[rem_node] = False
                    
                    # Find non-selected nodes with no conflicts to temp selected
                    non_sel = np.where(~in_selected)[0]
                    if len(non_sel) > 0:
                        cc = conflict_mat[non_sel][:, temp_in_selected].sum(axis=1)
                        addable = np.where(cc == 0)[0]
                        if len(addable) >= 2:
                            # Pick two
                            if len(addable) == 2:
                                picks = addable
                            else:
                                picks = rng.choice(addable, size=2, replace=False)
                            new_nodes = non_sel[picks]
                            # Check they don't conflict with each other
                            if not conflict_mat[new_nodes[0], new_nodes[1]]:
                                # Do the swap
                                selected_arr = np.delete(selected_arr, rem_idx)
                                selected_arr = np.concatenate([selected_arr, new_nodes])
                                in_selected[rem_node] = False
                                in_selected[new_nodes] = True
                                improved = True
                                break
        
        return selected_arr
    
    best_selected = np.array([], dtype=np.int64)
    best_size = 0
    
    rng = np.random.default_rng(42)
    
    num_restarts = 30
    
    for attempt in range(num_restarts):
        if time.time() > deadline - 30:
            break
            
        selected = greedy_independent_set(conflict_bool, n, rng)
        selected_arr = np.array(selected, dtype=np.int64)
        
        if len(selected_arr) > best_size:
            best_size = len(selected_arr)
            best_selected = selected_arr.copy()
        
        # Local search
        improved_arr = local_search(selected_arr, conflict_bool, n, rng, max_iters=200)
        
        if len(improved_arr) > best_size:
            best_size = len(improved_arr)
            best_selected = improved_arr.copy()
    
    result = candidates[best_selected]
    return result
