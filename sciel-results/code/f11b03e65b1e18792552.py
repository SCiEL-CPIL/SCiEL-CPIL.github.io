import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    N = 256
    
    # Precompute masks for 4-AP conflicts and 3-AP gains
    # conflict_masks[x] contains bitmasks of the other 3 elements for each 4-AP x is in
    # gain_masks[x] contains bitmasks of the other 2 elements for each 3-AP x is in
    
    conflict_masks = [[] for _ in range(N + 1)]
    gain_masks = [[] for _ in range(N + 1)]
    
    # Generate all 4-APs
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e > N:
                break
            
            # 4-AP: a, b, c, e
            # If we add x, and the other 3 are present, it's invalid.
            
            # x = a
            mask = (1 << b) | (1 << c) | (1 << e)
            conflict_masks[a].append(mask)
            
            # x = b
            mask = (1 << a) | (1 << c) | (1 << e)
            conflict_masks[b].append(mask)
            
            # x = c
            mask = (1 << a) | (1 << b) | (1 << e)
            conflict_masks[c].append(mask)
            
            # x = e
            mask = (1 << a) | (1 << b) | (1 << c)
            conflict_masks[e].append(mask)
            
    # Generate all 3-APs
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 2 + 1):
            b = a + d
            c = a + 2 * d
            if c > N:
                break
            
            # 3-AP: a, b, c
            
            # x = a
            mask = (1 << b) | (1 << c)
            gain_masks[a].append(mask)
            
            # x = b
            mask = (1 << a) | (1 << c)
            gain_masks[b].append(mask)
            
            # x = c
            mask = (1 << a) | (1 << b)
            gain_masks[c].append(mask)

    def is_valid_add(x, M):
        # M is bitmask of current set A
        for mask in conflict_masks[x]:
            if (M & mask) == mask:
                return False
        return True

    def count_3ap_gain(x, M):
        gain = 0
        for mask in gain_masks[x]:
            if (M & mask) == mask:
                gain += 1
        return gain

    def count_total_3ap(M):
        # Count all 3-APs in M
        # We can iterate through all precomputed 3-APs? 
        # Or just sum gain for each x in M? 
        # Sum of gain for x in M counts each 3-AP exactly 3 times? 
        # No, gain(x, M) counts 3-APs where x is the "new" element and the other 2 are in M.
        # If we sum gain(x, M) for all x in M, we count each 3-AP 3 times (once for each member).
        # So total = sum(gain(x, M) for x in M) // 3.
        
        # However, we must be careful: gain(x, M) requires the other 2 to be in M.
        # If x is in M, and we calculate gain(x, M), it counts 3-APs {x, y, z} where y,z in M.
        # So yes, summing over x in M and dividing by 3 works.
        
        total = 0
        # Iterate through bits set in M
        m = M
        while m:
            # Get lowest set bit
            lsb = m & -m
            x = lsb.bit_length() - 1
            total += count_3ap_gain(x, M)
            m ^= lsb
        return total // 3

    best_M = 0
    best_score = 0
    
    # Greedy Construction
    for _ in range(50): # Increase restarts due to speedup
        if time.time() > deadline - 10:
            break
            
        M = 0
        remaining = list(range(1, N + 1))
        random.shuffle(remaining)
        
        # We can optimize the search for best_x by iterating all remaining
        # But we can also just pick the best gain among valid ones
        
        while remaining and time.time() < deadline - 5:
            best_x = -1
            best_gain = -1
            
            # Optimization: If we have time, check all. 
            # For N=256, remaining is small.
            for x in remaining:
                if is_valid_add(x, M):
                    g = count_3ap_gain(x, M)
                    if g > best_gain:
                        best_gain = g
                        best_x = x
            
            if best_x == -1:
                break
            M |= (1 << best_x)
            remaining.remove(best_x)
            
        current_score = count_total_3ap(M)
        if current_score > best_score:
            best_score = current_score
            best_M = M

    # Local Search on best_M
    if time.time() < deadline - 2:
        M = best_M
        current_score = best_score
        improved = True
        
        while improved and time.time() < deadline - 1:
            improved = False
            
            # 1. Try adding any valid element
            for x in range(1, N + 1):
                if not (M & (1 << x)):
                    if is_valid_add(x, M):
                        new_M = M | (1 << x)
                        new_score = count_total_3ap(new_M)
                        if new_score > current_score:
                            M = new_M
                            current_score = new_score
                            improved = True
                            break # Restart loop with new M
            
            # 2. Try swapping: remove a, add b
            # This is O(N^2 * cost). N=256, so 65k swaps.
            # cost is ~ O(100) bitwise ops. 6.5M ops. Should be fast enough in Python? 
            # Maybe 0.1s per full pass.
            if time.time() < deadline - 1:
                A = []
                m = M
                while m:
                    lsb = m & -m
                    A.append(lsb.bit_length() - 1)
                    m ^= lsb
                
                B = [x for x in range(1, N + 1) if not (M & (1 << x))]
                
                for a in A:
                    if time.time() > deadline - 1: break
                    M_temp = M & ~(1 << a)
                    # Check if removing a allows adding b
                    for b in B:
                        if is_valid_add(b, M_temp):
                            new_M = M_temp | (1 << b)
                            # We can estimate score change:
                            # Score change = gain(b, M_temp) - loss(a, M)
                            # loss(a, M) is number of 3-APs containing a in M.
                            # gain(b, M_temp) is number of 3-APs containing b in M_temp.
                            
                            # Let's just calculate full score for simplicity if swap looks promising?
                            # Or approximate.
                            # Let's just do full calc for now, it's fast.
                            new_score = count_total_3ap(new_M)
                            if new_score > current_score:
                                M = new_M
                                current_score = new_score
                                improved = True
                                break
                    if improved:
                        break
            
            if current_score > best_score:
                best_score = current_score
                best_M = M

    # Convert bitmask to sorted list
    result = []
    m = best_M
    while m:
        lsb = m & -m
        result.append(lsb.bit_length() - 1)
        m ^= lsb
        
    return sorted(result)
