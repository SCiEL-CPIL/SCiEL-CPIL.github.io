# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional integer points maximizing k
    subject to: min pairwise squared distance >= max squared norm.
    
    Strategy:
    1. Use vectors with exactly 4 non-zero entries (each +/-1), so norm_sq = 4.
       Condition: for any two selected u, v with norm_sq 4:
         ||u-v||^2 = 8-2*u.v >= 4  =>  u.v <= 2
       So we need a set of 4-weight binary vectors with pairwise dot <= 2.
    
    2. Build a large set via greedy + local search (swap-based improvement).
    
    3. Try to add 2-norm vectors and other norm vectors.
    """
    d = 11
    start_time = time.time()
    deadline = start_time + 580
    
    # Generate all 4-norm vectors
    candidates_4 = []
    for combo in itertools.combinations(range(d), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            for idx, val in zip(combo, signs):
                v[idx] = val
            candidates_4.append(v)
    
    cand_arr = np.array(candidates_4, dtype=np.int64)
    n_cand = len(cand_arr)  # 5280
    
    # Precompute dot product matrix (5280 x 5280) - this is ~28M entries, ~224MB as int16
    # Actually let's use int8 since dots range from -4 to 4
    # 5280 * 5280 * 1 byte = ~28MB, manageable
    # But computing it takes time. Let's do it in chunks.
    
    # Actually, for local search we need fast dot checks. Let's precompute the full dot matrix.
    # 5280^2 = 27.9M dots, each requiring 11 multiplies+adds = ~307M ops
    # With numpy: cand_arr @ cand_arr.T should be fast
    
    if time.time() < deadline - 500:
        dot_mat = cand_arr @ cand_arr.T  # shape (5280, 5280), int64
    else:
        dot_mat = None
    
    np.random.seed(42)
    
    # Phase 1: Greedy to get initial solution
    best_selected_indices = []
    best_size = 0
    
    num_greedy_restarts = 3
    for restart in range(num_greedy_restarts):
        if time.time() > deadline - 300:
            break
        
        order = np.random.permutation(n_cand)
        selected = []
        selected_set = set()
        
        for idx in order:
            if time.time() > deadline - 280:
                break
            idx = int(idx)
            # Check if dot with all selected <= 2
            if not selected:
                selected.append(idx)
                selected_set.add(idx)
                continue
            
            # Check against selected
            # Use dot_mat if available
            if dot_mat is not None:
                # Check max dot with selected
                max_dot = -100
                for s in selected:
                    val = dot_mat[idx, s]
                    if val > max_dot:
                        max_dot = val
                        if max_dot > 2:
                            break
                if max_dot <= 2:
                    selected.append(idx)
                    selected_set.add(idx)
            else:
                v = cand_arr[idx]
                valid = True
                for s in selected:
                    if np.dot(cand_arr[s], v) > 2:
                        valid = False
                        break
                if valid:
                    selected.append(idx)
                    selected_set.add(idx)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected_indices = selected[:]
    
    # Phase 2: Local search - try to add more points by swapping
    # For each rejected candidate, try to find a selected point to remove that would allow it
    if dot_mat is not None and time.time() < deadline - 100:
        selected_set = set(best_selected_indices)
        selected_list = list(best_selected_indices)
        
        # Build conflict map: for each candidate, which selected points conflict with it
        # A selected point s conflicts with candidate c if dot_mat[c, s] > 2
        
        # Try multiple rounds of local search
        for ls_round in range(3):
            if time.time() > deadline - 60:
                break
            
            improved = False
            # Try to add each non-selected candidate
            for c in range(n_cand):
                if time.time() > deadline - 50:
                    break
                if c in selected_set:
                    continue
                
                # Find conflicts with selected
                conflicts = []
                for s in selected_list:
                    if dot_mat[c, s] > 2:
                        conflicts.append(s)
                
                if not conflicts:
                    # Can add directly!
                    selected_list.append(c)
                    selected_set.add(c)
                    improved = True
                    continue
                
                # Try to remove one conflict to add c
                # For each conflict s, check if removing s and adding c keeps all other pairs valid
                # This is expensive. Instead, just try: remove one conflict, add c
                # But we need to ensure c doesn't conflict with any remaining selected
                # c only conflicts with the ones in 'conflicts', so if we remove one conflict,
                # c still conflicts with the rest. We need to remove ALL conflicts.
                # That's too expensive if there are many conflicts.
                # Alternative: try to replace one selected point with c if c has fewer conflicts
                pass
        
        # Alternative local search: hill climbing with swaps
        # Try replacing selected point s with candidate c if c has fewer total conflicts
        for swap_iter in range(2000):
            if time.time() > deadline - 40:
                break
            
            # Pick a random selected point to try removing
            if not selected_list:
                break
            s_idx = np.random.randint(len(selected_list))
            s = selected_list[s_idx]
            
            # Try adding a random non-selected candidate
            c = np.random.randint(n_cand)
            if c in selected_set:
                continue
            
            # Check if c conflicts with selected (excluding s)
            valid = True
            for pos, sel in enumerate(selected_list):
                if pos == s_idx:
                    continue
                if dot_mat[c, sel] > 2:
                    valid = False
                    break
            
            if valid:
                # Replace s with c
                selected_list[s_idx] = c
                selected_set.discard(s)
                selected_set.add(c)
    
    # Phase 3: Try to add 2-norm vectors
    best_points = [cand_arr[i] for i in best_selected_indices]
    
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
        
        for v in cand2_arr:
            if time.time() > deadline - 10:
                break
            # Check: for each selected u (norm_sq=4), need u.v <= 1
            # For each selected w (norm_sq=2), need w.v <= 0
            dots = sel_arr @ v
            # All current selected have norm_sq=4, so check dot <= 1
            if np.max(dots) <= 1:
                best_points.append(v)
    
    # Phase 4: Try to add more 4-norm vectors that were rejected
    if time.time() < deadline - 10:
        sel_arr = np.array(best_points, dtype=np.int64)
        for i in range(n_cand):
            if time.time() > deadline - 5:
                break
            if i in best_selected_indices:
                continue
            v = cand_arr[i]
            dots = sel_arr @ v
            # Mixed norms: for 4-norm selected, need dot <= 2
            # for 2-norm selected, need dot <= 1
            # Check each
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

# EVOLVE-BLOCK-END
