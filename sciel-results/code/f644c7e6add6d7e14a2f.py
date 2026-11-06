import time
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    # Generate smooth numbers (2^a * 3^b * 5^c) <= 128
    smooth = set()
    for a in range(8):
        for b in range(5):
            for c in range(4):
                val = (2 ** a) * (3 ** b) * (5 ** c)
                if val <= 128:
                    smooth.add(val)
    
    smooth_list = sorted(smooth)
    n_smooth = len(smooth_list)
    smooth_set = set(smooth_list)
    
    # Precompute divisibility: div[i][j] = True if smooth_list[i] divides smooth_list[j]
    # Antichain constraint: no two chosen terms where one divides the other
    # Build compatibility: compat[i] = bitmask of indices j where i and j can coexist
    # i and j can coexist if smooth_list[i] does NOT divide smooth_list[j] AND vice versa
    compat = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i == j:
                continue
            vi = smooth_list[i]
            vj = smooth_list[j]
            # They can coexist if neither divides the other
            if vi % vj != 0 and vj % vi != 0:
                compat[i] |= (1 << j)
    
    # For a set represented by bitmask, check if it's an antichain
    # We can precompute for small masks, but let's use a different approach
    # Actually, let's just check incrementally during DFS
    
    # Precompute for each smooth number, which other smooth numbers it's compatible with
    # compat_mask[i] = bitmask of j (j != i) such that i and j form an antichain pair
    
    # For DFS, we maintain a bitmask of chosen indices
    # When adding index i, we need: (chosen_mask & compat[i]) == chosen_mask? No.
    # Actually: for all j in chosen_mask, i must be compatible with j
    # i.e., for all j in chosen_mask: (compat[i] >> j) & 1 == 1
    # Equivalently: chosen_mask & ~compat[i] == 0
    
    # Let's precompute: for each i, incompat[i] = bitmask of j that are incompatible with i
    # Then when adding i, we need: chosen_mask & incompat[i] == 0
    incompat = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i == j:
                continue
            vi = smooth_list[i]
            vj = smooth_list[j]
            if vi % vj == 0 or vj % vi == 0:
                incompat[i] |= (1 << j)
    
    reps = [None] * 129
    
    def dfs(rem, start_idx, current_mask, k_remaining):
        if k_remaining == 0:
            return current_mask if rem == 0 else None
        if rem <= 0:
            return None
        
        # Pruning: need at least k_remaining more terms
        # The smallest possible sum of k_remaining distinct smooth numbers >= start_idx
        # that are mutually compatible... this is complex. Use simpler bound.
        
        # Upper bound: sum of largest k_remaining smooth numbers from start_idx onward
        # that could fit in rem. But we also need them to be compatible.
        # Simpler: if rem > sum of all smooth numbers from start_idx that are <= rem, fail
        # Actually, just iterate and let the recursion handle it.
        
        # Pruning: if the sum of the largest k_remaining available smooth numbers (from start_idx)
        # that are <= rem is less than rem, fail
        # But we also have compatibility constraints. Let's use a weaker bound first.
        
        # Weak bound: sum of k_remaining smallest smooth numbers >= start_idx
        # If that sum > rem, we can't fit k_remaining numbers
        # Actually we need sum >= rem, so if even the largest k_remaining can't sum to rem, fail
        
        # Let's just iterate
        for idx in range(start_idx, n_smooth):
            val = smooth_list[idx]
            if val > rem:
                break
            
            # Check compatibility with all currently chosen
            if current_mask & incompat[idx]:
                continue
            
            new_mask = current_mask | (1 << idx)
            res = dfs(rem - val, idx + 1, new_mask, k_remaining - 1)
            if res is not None:
                return res
        return None
    
    def mask_to_list(mask):
        result = []
        for i in range(n_smooth):
            if mask & (1 << i):
                result.append(smooth_list[i])
        return result
    
    for t in range(1, 129):
        if time.time() > deadline:
            break
        
        found = False
        
        # Try k=1
        if t in smooth_set:
            # Find index
            for i in range(n_smooth):
                if smooth_list[i] == t:
                    reps[t] = tuple([t])
                    break
            found = True
        else:
            # Try k=2
            for i in range(n_smooth):
                a = smooth_list[i]
                if a >= t:
                    break
                b = t - a
                if b < a:
                    continue
                if b in smooth_set:
                    # Find index of b
                    j = smooth_list.index(b) if b in smooth_list else -1
                    # Actually, let's find index properly
                    # Since smooth_list is sorted, we can binary search or just check
                    # But we need to ensure a and b are compatible
                    if i < n_smooth and (1 << i) & incompat[j] == 0 if j >= 0 else False:
                        pass
                    # Let me redo this more carefully
                    # Find index of b
                    j_idx = None
                    for j in range(n_smooth):
                        if smooth_list[j] == b:
                            j_idx = j
                            break
                    if j_idx is not None and j_idx != i and (incompat[i] & (1 << j_idx)) == 0:
                        reps[t] = tuple(sorted([a, b]))
                        found = True
                        break
                if time.time() > deadline:
                    break
            if found:
                continue
            
            # Try k=3
            for i in range(n_smooth):
                a = smooth_list[i]
                if a * 3 >= t:  # a + b + c >= 3a, so if 3a >= t, a is too big
                    break
                for j in range(i + 1, n_smooth):
                    b = smooth_list[j]
                    if a + b >= t:
                        break
                    c = t - a - b
                    if c <= b:
                        continue
                    if c in smooth_set:
                        # Find index of c
                        k_idx = None
                        for k in range(n_smooth):
                            if smooth_list[k] == c:
                                k_idx = k
                                break
                        if k_idx is not None and k_idx != i and k_idx != j:
                            # Check all pairs compatible
                            if (incompat[i] & (1 << j)) == 0 and \
                               (incompat[i] & (1 << k_idx)) == 0 and \
                               (incompat[j] & (1 << k_idx)) == 0:
                                reps[t] = tuple(sorted([a, b, c]))
                                found = True
                                break
                    if time.time() > deadline:
                        break
                if found or time.time() > deadline:
                    break
            if found:
                continue
            
            # Try k>=4 using DFS
            for k in range(4, 11):
                if time.time() > deadline:
                    break
                res_mask = dfs(t, 0, 0, k)
                if res_mask is not None:
                    reps[t] = tuple(mask_to_list(res_mask))
                    found = True
                    break
            
            if not found:
                # Should not happen for t <= 128
                pass
    
    # Build result
    result = []
    for t in range(1, 129):
        if reps[t] is None:
            # Fallback: this should not happen
            # Try a greedy approach as last resort
            result.append([])
        else:
            result.append(list(reps[t]))
    
    return result
