"""Program to find a large pairwise non-coprime subset of [1,255] containing 255."""
import math
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Return a nonempty subset of [1,255] containing 255,
    with gcd(a,b)>1 for all distinct pairs. Maximize cardinality.
    
    Strategy: Build compatibility graph using bitmasks, then use 
    greedy + local search to find maximum clique containing 255.
    """
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    n = 255
    # Candidates must share a factor with 255 (3, 5, or 17)
    candidates = [x for x in range(1, n + 1) if x % 3 == 0 or x % 5 == 0 or x % 17 == 0]
    num_cands = len(candidates)
    
    # Map candidate value to index
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    idx_255 = val_to_idx[255]
    
    # Build compatibility bitmask
    compat = [0] * num_cands
    for i in range(num_cands):
        for j in range(i + 1, num_cands):
            if math.gcd(candidates[i], candidates[j]) > 1:
                compat[i] |= (1 << j)
                compat[j] |= (1 << i)
    
    # Helper: check if vertex v is compatible with all vertices in mask
    def is_compatible(v, mask):
        return (compat[v] & mask) == mask
    
    # Helper: count bits
    def popcount(x):
        return bin(x).count('1')
    
    # Greedy construction: start with 255, iteratively add best compatible vertex
    def greedy_construct():
        selected_mask = (1 << idx_255)
        selected = {255}
        remaining_mask = 0
        for i in range(num_cands):
            if i != idx_255:
                remaining_mask |= (1 << i)
        
        while remaining_mask:
            # Find compatible vertices
            compat_mask = 0
            temp = remaining_mask
            while temp:
                lsb = temp & (-temp)
                i = lsb.bit_length() - 1
                if is_compatible(i, selected_mask):
                    compat_mask |= lsb
                temp ^= lsb
            
            if not compat_mask:
                break
            
            # Pick vertex with highest degree among compatible ones
            best_i = -1
            best_deg = -1
            temp = compat_mask
            while temp:
                lsb = temp & (-temp)
                i = lsb.bit_length() - 1
                deg = popcount(compat[i] & remaining_mask)
                if deg > best_deg:
                    best_deg = deg
                    best_i = i
                temp ^= lsb
            
            if best_i < 0:
                break
            
            selected_mask |= (1 << best_i)
            selected.add(candidates[best_i])
            remaining_mask &= ~(1 << best_i)
        
        return selected
    
    # Local search: 1-swap improvement
    def local_search(selected, max_iter=200):
        selected = set(selected)
        selected_mask = 0
        for v in selected:
            selected_mask |= (1 << val_to_idx[v])
        
        improved = True
        iteration = 0
        while improved and iteration < max_iter and time.time() < deadline:
            improved = False
            iteration += 1
            sel_list = list(selected)
            for u in sel_list:
                if time.time() > deadline:
                    return selected
                u_idx = val_to_idx[u]
                # Try removing u and adding some w
                for w in candidates:
                    if w in selected or w == 255:
                        continue
                    w_idx = val_to_idx[w]
                    # Check if w is compatible with all in selected \ {u}
                    if not is_compatible(w_idx, selected_mask & ~(1 << u_idx)):
                        continue
                    # Check if w is compatible with 255 (already ensured by candidate filter)
                    # Perform swap
                    new_selected = set(selected)
                    new_selected.remove(u)
                    new_selected.add(w)
                    # Verify
                    valid = True
                    for a in new_selected:
                        for b in new_selected:
                            if a != b and math.gcd(a, b) == 1:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        selected = new_selected
                        improved = True
                        break
                if improved:
                    break
        
        return selected
    
    # 2-swap local search
    def two_swap_search(selected, max_iter=50):
        selected = set(selected)
        improved = True
        iteration = 0
        while improved and iteration < max_iter and time.time() < deadline:
            improved = False
            iteration += 1
            sel_list = list(selected)
            for i, u in enumerate(sel_list):
                if time.time() > deadline:
                    return selected
                for j, v in enumerate(sel_list):
                    if i >= j:
                        continue
                    u_idx = val_to_idx[u]
                    v_idx = val_to_idx[v]
                    # Try removing u and v, adding w1 and w2
                    for w1 in candidates:
                        if w1 in selected or w1 == 255:
                            continue
                        w1_idx = val_to_idx[w1]
                        for w2 in candidates:
                            if w2 in selected or w2 == 255 or w2 == w1:
                                continue
                            w2_idx = val_to_idx[w2]
                            # Check compatibility
                            remaining_mask = 0
                            for x in selected:
                                if x != u and x != v:
                                    remaining_mask |= (1 << val_to_idx[x])
                            if not is_compatible(w1_idx, remaining_mask):
                                continue
                            if not is_compatible(w2_idx, remaining_mask):
                                continue
                            if math.gcd(w1, w2) == 1:
                                continue
                            # Perform swap
                            new_selected = set(selected)
                            new_selected.remove(u)
                            new_selected.remove(v)
                            new_selected.add(w1)
                            new_selected.add(w2)
                            # Verify
                            valid = True
                            for a in new_selected:
                                for b in new_selected:
                                    if a != b and math.gcd(a, b) == 1:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if valid:
                                selected = new_selected
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return selected
    
    # Try multiple greedy starts
    best_set = greedy_construct()
    best_size = len(best_set)
    
    # Local search on best
    best_set = local_search(best_set)
    best_size = len(best_set)
    
    # 2-swap search
    if time.time() < deadline:
        best_set = two_swap_search(best_set)
        best_size = len(best_set)
    
    # Also try: all multiples of 3 as baseline
    multiples_of_3 = set(range(3, 256, 3))
    if len(multiples_of_3) > best_size:
        best_set = multiples_of_3
        best_size = len(multiples_of_3)
    
    # Try local search on multiples of 3
    if time.time() < deadline:
        improved_set = local_search(multiples_of_3)
        if len(improved_set) > best_size:
            best_set = improved_set
            best_size = len(improved_set)
    
    # Final verification
    final_set = set(best_set)
    for a in final_set:
        for b in final_set:
            if a != b and math.gcd(a, b) == 1:
                # Invalid, fall back to multiples of 3
                return list(range(3, 256, 3))
    
    if 255 not in final_set:
        final_set.add(255)
    
    return list(final_set)
# EVOLVE-BLOCK-END
