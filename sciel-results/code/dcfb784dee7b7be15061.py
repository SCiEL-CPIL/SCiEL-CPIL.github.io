import numpy as np
import time
import random

# EVOLVE-BLOCK-START
P = {'n': 12, 'm': 40}

def construct():
    n = P['n']
    m = P['m']
    N = 1 << n
    
    # Precompute subsets and supersets for all masks
    # subs[mask] contains masks m' such that m' < mask (strict subset)
    # sups[mask] contains masks m' such that mask < m' (strict superset)
    subs = [None] * N
    sups = [None] * N
    
    for mask in range(N):
        # Find subsets
        s = []
        # Iterate over submasks of mask
        # Standard trick: sub = (sub - 1) & mask
        sub = (mask - 1) & mask
        while sub >= 0:
            if sub < mask: # Strict subset
                s.append(sub)
            sub = (sub - 1) & mask
        # Ensure we don't include mask itself in the loop logic above?
        # (mask-1)&mask is always < mask.
        subs[mask] = np.array(s, dtype=np.int32)
        
        # Find supersets
        # Supersets of mask are masks m' such that mask | m' == m'
        # Equivalent to: m' = mask | x where x is subset of ~mask
        # Or iterate all and check? Too slow for precomp if done naively.
        # Better: iterate over all masks?
        # Actually, we can just compute on the fly or use a different precomp.
        # For N=4096, we can just build a list of lists by checking all pairs?
        # 4096^2 = 16M checks. Fast enough in Python? Maybe 1-2 seconds.
        
    # Let's do the O(N^2) precomputation for comparability matrix or lists
    # It's simpler and safer.
    subs = [[] for _ in range(N)]
    sups = [[] for _ in range(N)]
    
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            # Check if i is strict subset of j
            if (i & ~j) == 0: # i | j == j
                sups[i].append(j)
                subs[j].append(i)
                
    # Convert to numpy arrays for fast summing
    subs_np = [np.array(x, dtype=np.int32) if x else np.array([], dtype=np.int32) for x in subs]
    sups_np = [np.array(x, dtype=np.int32) if x else np.array([], dtype=np.int32) for x in sups]
    
    # Precompute comparability bitset? 
    # comp[i][j] = 1 if i and j are comparable.
    # We can just use the bitwise check in the inner loop.
    
    def get_score(selected_masks, is_sel_array):
        # selected_masks: list of int
        # is_sel_array: numpy array of 0/1
        score = 0
        for i in selected_masks:
            # Count comparable sets in S
            # sum(is_sel[subs[i]] + is_sel[sups[i]])
            # Note: i is not in subs[i] or sups[i]
            s = subs_np[i]
            p = sups_np[i]
            count = 0
            if len(s) > 0:
                count += np.sum(is_sel_array[s])
            if len(p) > 0:
                count += np.sum(is_sel_array[p])
            score += count
        return score // 2 # Each pair counted twice? 
        # Wait. 
        # For pair (A, B) with A < B:
        # When counting A, B is in sups[A]. Counted 1.
        # When counting B, A is in subs[B]. Counted 1.
        # Total sum is 2 * (# pairs).
        # So score = sum // 2.
        
    def evaluate_swap_delta(i_out, j_in, selected_masks, is_sel_array):
        # Calculate change in score when removing i_out and adding j_in
        # Current score contribution of i_out: L_i
        # New score contribution of j_in (in S \ {i_out}): G_j
        # Delta = G_j - L_i
        
        # L_i: comparable count of i_out in S
        s = subs_np[i_out]
        p = sups_np[i_out]
        L_i = 0
        if len(s) > 0:
            L_i += np.sum(is_sel_array[s])
        if len(p) > 0:
            L_i += np.sum(is_sel_array[p])
            
        # G_j: comparable count of j_in in S \ {i_out}
        # is_sel_array currently has i_out as 1.
        # We need to count comparable sets in S, excluding i_out.
        # Count comparable in S:
        s_j = subs_np[j_in]
        p_j = sups_np[j_in]
        G_j_full = 0
        if len(s_j) > 0:
            G_j_full += np.sum(is_sel_array[s_j])
        if len(p_j) > 0:
            G_j_full += np.sum(is_sel_array[p_j])
        
        # If i_out and j_in are comparable, G_j_full includes i_out.
        # We remove i_out, so we subtract 1 if comparable.
        comparable = 1 if ((i_out & ~j_in) == 0 or (j_in & ~i_out) == 0) else 0
        G_j = G_j_full - comparable
        
        return G_j - L_i

    # Main Loop
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin
    
    # Initial random solution
    all_masks = list(range(N))
    random.shuffle(all_masks)
    selected = all_masks[:m]
    is_sel = np.zeros(N, dtype=np.int8)
    for mask in selected:
        is_sel[mask] = 1
        
    best_score = get_score(selected, is_sel)
    best_selected = selected.copy()
    best_is_sel = is_sel.copy()
    
    current_score = best_score
    current_selected = selected.copy()
    current_is_sel = is_sel.copy()
    
    # Local Search
    while time.time() < deadline:
        improved = False
        
        # Calculate L_i for all i in current_selected
        L_vals = {}
        for i in current_selected:
            s = subs_np[i]
            p = sups_np[i]
            L = 0
            if len(s) > 0:
                L += np.sum(current_is_sel[s])
            if len(p) > 0:
                L += np.sum(current_is_sel[p])
            L_vals[i] = L
            
        # Calculate G_j for all j not in current_selected
        # To save time, we can just calculate on the fly for candidates
        # But let's precompute G_j_full for all j
        G_vals = {}
        unselected = [x for x in all_masks if current_is_sel[x] == 0]
        
        for j in unselected:
            s_j = subs_np[j]
            p_j = sups_np[j]
            G = 0
            if len(s_j) > 0:
                G += np.sum(current_is_sel[s_j])
            if len(p_j) > 0:
                G += np.sum(current_is_sel[p_j])
            G_vals[j] = G
            
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # Find best swap
        for i in current_selected:
            L_i = L_vals[i]
            for j in unselected:
                G_j_full = G_vals[j]
                comparable = 1 if ((i & ~j) == 0 or (j & ~i) == 0) else 0
                G_j = G_j_full - comparable
                delta = G_j - L_i
                if delta > best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
                    
        if best_delta > 0:
            # Apply swap
            current_is_sel[best_i] = 0
            current_is_sel[best_j] = 1
            current_selected.remove(best_i)
            current_selected.append(best_j)
            current_score += best_delta
            improved = True
            
            if current_score > best_score:
                best_score = current_score
                best_selected = current_selected.copy()
                best_is_sel = current_is_sel.copy()
        else:
            # No improvement, try random restart or perturbation
            # Simple perturbation: swap a random pair
            if random.random() < 0.5:
                i_rand = random.choice(current_selected)
                j_rand = random.choice(unselected)
                current_is_sel[i_rand] = 0
                current_is_sel[j_rand] = 1
                current_selected.remove(i_rand)
                current_selected.append(j_rand)
                # Recalculate score? 
                # It's safer to just recompute or update carefully.
                # Given the cost, let's just recompute score for best tracking?
                # No, we can just continue search. The local search will fix it.
            else:
                # Random restart
                random.shuffle(all_masks)
                current_selected = all_masks[:m]
                current_is_sel = np.zeros(N, dtype=np.int8)
                for mask in current_selected:
                    current_is_sel[mask] = 1
                current_score = get_score(current_selected, current_is_sel)
                if current_score > best_score:
                    best_score = current_score
                    best_selected = current_selected.copy()
                    best_is_sel = current_is_sel.copy()
                    
    # Convert best_selected to list of lists
    result = []
    for mask in best_selected:
        result.append([b for b in range(n) if (mask >> b) & 1])
        
    return result

# EVOLVE-BLOCK-END
