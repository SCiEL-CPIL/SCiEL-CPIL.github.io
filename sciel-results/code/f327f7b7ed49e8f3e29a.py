import numpy as np
import time
import itertools
import heapq

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 620  # safety margin before 630s kill

    # Generate all 4-norm vectors (±1 on exactly 4 coordinates)
    candidates_4 = []
    for combo in itertools.combinations(range(d), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            for idx, val in zip(combo, signs):
                v[idx] = val
            candidates_4.append(v)
    
    cand_arr = np.array(candidates_4, dtype=np.int64)
    n_cand = len(cand_arr)  # 5280

    # Precompute dot product matrix
    if time.time() < deadline - 500:
        dot_mat = cand_arr @ cand_arr.T
    else:
        return np.array([[1,1,0,0,0,0,0,0,0,0,0],[0,0,1,1,0,0,0,0,0,0,0]], dtype=np.int64)

    # Build adjacency: edge (i,j) if dot_mat[i,j] > 2
    # We work with the complement: non-conflict means dot <= 2
    # For independent set, we want vertices with no edges among them.
    # adjacency_mask[i] = set of j where dot_mat[i,j] > 2 (i != j)
    
    # Efficient representation: for each vertex, store conflict count and use numpy for checks
    # conflict_count[i] = number of j != i with dot_mat[i,j] > 2
    conflict_count = np.sum(dot_mat > 2, axis=1) - 1  # exclude self (dot=4>2)

    # Phase 1: Greedy initialization using priority queue (pick vertex with min conflict among remaining)
    np.random.seed(42)
    
    best_selected = None
    best_size = 0
    
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline - 200:
            break
        
        # Greedy: repeatedly pick the vertex with minimum current conflict degree
        # among non-selected vertices
        selected = []
        selected_set = set()
        in_conflict = np.zeros(n_cand, dtype=bool)  # True if vertex conflicts with some selected
        
        # For speed, maintain for each non-selected vertex the count of conflicts with selected
        # conflict_with_selected[i] = number of selected s where dot_mat[i,s] > 2
        conflict_with_selected = np.zeros(n_cand, dtype=np.int32)
        
        # Use a simple approach: at each step, find non-selected vertex with min conflict_with_selected
        # This is O(n_cand * |selected|) per step which is too slow for 5280 * 5280
        # Better: use the dot matrix directly
        
        # Alternative greedy: random order, pick if no conflict
        # But let's try: sort by initial conflict_count, pick from lowest first
        order = np.argsort(conflict_count)
        # Add some randomness
        for i in range(n_cand):
            j = np.random.randint(i, n_cand)
            order[i], order[j] = order[j], order[i]
        
        for idx in order:
            if time.time() > deadline - 180:
                break
            idx = int(idx)
            if idx in selected_set:
                continue
            # Check if idx conflicts with any selected
            # dot_mat[idx, selected] should all be <= 2
            if selected:
                sel_arr = np.array(selected)
                dots = dot_mat[idx, sel_arr]
                if np.max(dots) > 2:
                    continue
            selected.append(idx)
            selected_set.add(idx)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected[:]
    
    if best_selected is None:
        best_selected = [0, 1]
        best_size = 2

    # Phase 2: Local search using 2-opt moves with degree-based priority
    # Representation: selected as a list, selected_set as a set
    selected_list = best_selected[:]
    selected_set = set(selected_list)
    
    # Precompute: for each candidate, its dot with all selected (for fast conflict check)
    # conflict_with_selected[i] = max(dot_mat[i, s] for s in selected) - we need this <= 2
    
    # Build a priority queue of "promising" candidates: those with fewest conflicts
    # For each non-selected candidate, compute number of conflicts with selected
    # conflicts_count[i] = sum(1 for s in selected if dot_mat[i,s] > 2)
    
    def compute_conflict_counts(sel_list):
        """For each non-selected candidate, count conflicts with sel_list."""
        if not sel_list:
            return np.zeros(n_cand, dtype=np.int32)
        sel_arr = np.array(sel_list)
        # dot_mat[:, sel_arr] has shape (n_cand, len(sel_arr))
        dots = dot_mat[:, sel_arr]  # (n_cand, k)
        conflicts = np.sum(dots > 2, axis=1)
        return conflicts.astype(np.int32)
    
    # Local search: try to improve by replacing one selected vertex with one non-selected
    # or by adding a non-selected vertex that has 0 conflicts
    # Key move: find a non-selected vertex with 0 conflicts -> add it
    # Then try 1-for-1 swaps: remove s, add c if c has no conflicts with (selected - {s})
    
    num_ls_rounds = 10
    for ls_round in range(num_ls_rounds):
        if time.time() > deadline - 60:
            break
        
        improved = False
        
        # Step 1: Try to add any non-selected vertex with 0 conflicts
        conflict_counts = compute_conflict_counts(selected_list)
        # Vertices with 0 conflicts that are not selected
        zero_conflict = np.where((conflict_counts == 0) & (~np.isin(np.arange(n_cand), list(selected_set))))[0]
        
        # Add all zero-conflict vertices
        for c in zero_conflict:
            c = int(c)
            selected_list.append(c)
            selected_set.add(c)
            improved = True
        
        # Recompute after additions
        if improved:
            conflict_counts = compute_conflict_counts(selected_list)
        
        # Step 2: Try 1-for-1 swaps
        # For each selected vertex s, for each non-selected vertex c:
        #   if c has no conflicts with (selected - {s}), then swap is valid
        #   Prefer swaps that reduce total conflict count
        # This is O(|selected| * n_cand) per round which is ~250*5280 = 1.3M, fast
        
        # Compute conflict matrix more efficiently
        # For swap (remove s, add c): c must have dot <= 2 with all selected except possibly s
        # i.e., max(dot_mat[c, selected\{s}]) <= 2
        
        # For each c not in selected, find the number of selected vertices it conflicts with
        # If that count is 1, say it conflicts only with s*, then swapping s* out and c in is valid
        # If count is 0, we already handled it above
        
        # Let's be smarter: for each non-selected c with exactly 1 conflict, identify that conflict
        one_conflict_mask = (conflict_counts == 1)
        one_conflict_candidates = np.where(one_conflict_mask & (~np.isin(np.arange(n_cand), list(selected_set))))[0]
        
        for c in one_conflict_candidates:
            if time.time() > deadline - 40:
                break
            c = int(c)
            # Find which selected vertex it conflicts with
            sel_arr = np.array(selected_list)
            dots_c = dot_mat[c, sel_arr]
            conflict_indices = np.where(dots_c > 2)[0]
            if len(conflict_indices) != 1:
                continue
            s_pos = conflict_indices[0]
            s = selected_list[s_pos]
            # Swap: remove s, add c
            selected_list[s_pos] = c
            selected_set.discard(s)
            selected_set.add(c)
            improved = True
        
        # Step 3: Try 2-for-1 moves (remove 2 selected, add 1 non-selected) - more powerful
        # For non-selected c with 2 conflicts s1, s2: if we remove both s1 and s2, c can be added
        # But we lose 2 and gain 1, so net -1. Only useful if it enables further additions.
        # Skip for now; focus on 1-for-1 and 0-conflict additions.
        
        # Step 4: Try to find a non-selected c and selected s such that:
        #   c conflicts with s, but s conflicts with NO other non-selected candidate that we'd want
        #   This is complex; skip.
        
        if not improved:
            # Try random restart of local search with perturbation
            if ls_round < num_ls_rounds - 2:
                # Perturb: randomly remove 5 selected, re-greedy fill
                if len(selected_list) > 10:
                    remove_count = min(5, len(selected_list) - 5)
                    remove_indices = np.random.choice(len(selected_list), size=remove_count, replace=False)
                    for ri in sorted(remove_indices, reverse=True):
                        s = selected_list.pop(ri)
                        selected_set.discard(s)
                    
                    # Re-greedy: add back zero-conflict and one-conflict vertices
                    conflict_counts = compute_conflict_counts(selected_list)
                    zero_conflict = np.where((conflict_counts == 0) & (~np.isin(np.arange(n_cand), list(selected_set))))[0]
                    for c in zero_conflict:
                        c = int(c)
                        selected_list.append(c)
                        selected_set.add(c)
                    
                    conflict_counts = compute_conflict_counts(selected_list)
                    one_conflict_candidates = np.where((conflict_counts == 1) & (~np.isin(np.arange(n_cand), list(selected_set))))[0]
                    for c in one_conflict_candidates:
                        if time.time() > deadline - 30:
                            break
                        c = int(c)
                        sel_arr = np.array(selected_list)
                        dots_c = dot_mat[c, sel_arr]
                        conflict_indices = np.where(dots_c > 2)[0]
                        if len(conflict_indices) != 1:
                            continue
                        s_pos = conflict_indices[0]
                        s = selected_list[s_pos]
                        selected_list[s_pos] = c
                        selected_set.discard(s)
                        selected_set.add(c)
    
    # Phase 3: Try to add 2-norm vectors
    best_points = [cand_arr[i] for i in selected_list]
    
    if time.time() < deadline - 20:
        candidates_2 = []
        for combo in itertools.combinations(range(d), 2):
            for signs in itertools.product([1, -1], repeat=2):
                v = np.zeros(d, dtype=np.int64)
                for idx, val in zip(combo, signs):
                    v[idx] = val
                candidates_2.append(v)
        
        cand2_arr = np.array(candidates_2, dtype=np.int64)
        sel_arr = np.array(best_points, dtype=np.int64)
        
        added_any = False
        for v in cand2_arr:
            if time.time() > deadline - 10:
                break
            # For 4-norm selected u: need u.v <= 1 (since ||u-v||^2 = 4+2-2uv = 6-2uv >= 4 => uv <= 1)
            # For 2-norm selected w: need w.v <= 0 (since ||w-v||^2 = 2+2-2wv = 4-2wv >= 4 => wv <= 0)
            dots = sel_arr @ v
            # Check thresholds per norm
            valid = True
            for j in range(len(sel_arr)):
                norm_j = int(np.dot(sel_arr[j], sel_arr[j]))
                threshold = 1 if norm_j == 4 else 0
                if dots[j] > threshold:
                    valid = False
                    break
            if valid:
                best_points.append(v)
                added_any = True
    
    # Phase 4: Try to add remaining 4-norm vectors
    if time.time() < deadline - 5:
        sel_arr = np.array(best_points, dtype=np.int64)
        for i in range(n_cand):
            if time.time() > deadline - 3:
                break
            if i in selected_set:
                continue
            v = cand_arr[i]
            dots = sel_arr @ v
            valid = True
            for j in range(len(sel_arr)):
                norm_j = int(np.dot(sel_arr[j], sel_arr[j]))
                threshold = 1 if norm_j == 2 else 2
                if dots[j] > threshold:
                    valid = False
                    break
            if valid:
                best_points.append(v)
    
    # Final validation
    if not best_points:
        p1 = np.zeros(d, dtype=np.int64)
        p1[0] = 1
        p1[1] = 1
        p2 = np.zeros(d, dtype=np.int64)
        p2[2] = 1
        p2[3] = 1
        best_points = [p1, p2]
    
    result = np.array(best_points, dtype=np.int64)
    result = np.around(result).astype(np.int64)
    
    return result
