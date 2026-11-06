import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = 24
    # LCM of 1..24
    # 1..24 LCM is 5354228880
    L = 5354228880
    
    # Weights
    w = [L // i for i in range(1, n + 1)]
    
    # Precompute sums for all masks
    # This might be slow in pure python if not optimized
    # 2^24 = 16777216
    # We can use a list of size 2^24
    # But memory? 16M integers. ~128MB. OK.
    
    # To speed up sum calculation, we can use DP
    # sum_w[m] = sum_w[m & (m-1)] + w[bit]
    
    # However, iterating 16M times in Python takes time.
    # Let's try to do it efficiently.
    
    # Alternative:
    # We only need to find masks that sum to w[x] for x in 1..24.
    # We can use a dictionary: sum -> list of masks.
    # But we still need to generate all masks to populate it.
    
    # Let's try the DP approach for sums.
    # It's a single loop of 16M.
    
    # Actually, we can just iterate masks and calculate sum on the fly?
    # No, we need to group by sum.
    
    # Let's use the dictionary approach.
    # sum_to_masks = {}
    # for m in range(1, 1 << n):
    #    # Calculate sum
    #    # This is O(n) per mask if done naively.
    #    # O(n) * 2^n = 24 * 16M = 400M ops. Too slow.
    
    # We need the O(1) update.
    sum_w = [0] * (1 << n)
    # sum_w[0] = 0
    for m in range(1, 1 << n):
        lsb = m & -m
        idx = lsb.bit_length() - 1
        sum_w[m] = sum_w[m ^ lsb] + w[idx]
        
    # Check time
    if time.time() > deadline - 10:
        return list(range(13, 25))
        
    # Group masks by sum
    sum_to_masks = {}
    for m in range(1, 1 << n):
        s = sum_w[m]
        if s in sum_to_masks:
            sum_to_masks[s].append(m)
        else:
            sum_to_masks[s] = [m]
            
    # Find forbidden masks
    forbidden = set()
    for x in range(1, n + 1):
        target = w[x - 1]
        if target in sum_to_masks:
            for m in sum_to_masks[target]:
                # Check if x is not in m
                if (m & (1 << (x - 1))) == 0:
                    # Forbidden set is m union {x}
                    fm = m | (1 << (x - 1))
                    forbidden.add(fm)
                    
    # Remove supersets
    # Sort by popcount
    forbidden_list = sorted(forbidden, key=lambda x: bin(x).count('1'))
    minimal_forbidden = []
    for fm in forbidden_list:
        is_superset = False
        for mf in minimal_forbidden:
            if (fm & mf) == mf:
                is_superset = True
                break
        if not is_superset:
            minimal_forbidden.append(fm)
            
    # Now search for max valid set
    # Iterate sizes from n down to 1
    best_set = None
    
    # Optimization: Precompute bit counts?
    # We are using combinations, so we know the size.
    
    for k in range(n, 0, -1):
        if time.time() > deadline - 5:
            break
            
        # Generate combinations of size k
        # Using itertools.combinations on indices 0..23
        for comb in itertools.combinations(range(n), k):
            # Create mask
            mask = 0
            for i in comb:
                mask |= (1 << i)
                
            # Check validity
            valid = True
            for fm in minimal_forbidden:
                if (mask & fm) == fm:
                    valid = False
                    break
                    
            if valid:
                best_set = comb
                break
                
        if best_set:
            break
            
    if best_set:
        # Convert to 1-based indices
        return [i + 1 for i in best_set]
    else:
        # Fallback
        return [1]
