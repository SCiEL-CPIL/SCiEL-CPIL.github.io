import random

def construct():
    n = 512
    
    # Precompute all 4-term arithmetic progressions
    # An AP is (a, a+d, a+2d, a+3d) with 1 <= a, a+3d <= n
    # We store conflicts for each element x.
    # conflicts[x] is a list of bitmasks representing the other 3 elements in an AP containing x.
    # If (current_mask & conflict_mask) == conflict_mask, then adding x would complete an AP.
    
    conflicts = [[] for _ in range(n)]
    
    # Generate all APs
    for d in range(1, (n - 3) // 3 + 1):
        for start in range(1, n - 3 * d + 1):
            # Elements are 1-indexed in the problem, 0-indexed in code
            a = start - 1
            b = start + d - 1
            c = start + 2 * d - 1
            e = start + 3 * d - 1
            
            # The AP is {a, b, c, e}
            # For each element, the "other three" form a conflict mask.
            # For a: others are b, c, e
            mask_a = (1 << b) | (1 << c) | (1 << e)
            # For b: others are a, c, e
            mask_b = (1 << a) | (1 << c) | (1 << e)
            # For c: others are a, b, e
            mask_c = (1 << a) | (1 << b) | (1 << e)
            # For e: others are a, b, c
            mask_e = (1 << a) | (1 << b) | (1 << c)
            
            conflicts[a].append(mask_a)
            conflicts[b].append(mask_b)
            conflicts[c].append(mask_c)
            conflicts[e].append(mask_e)
            
    best_size = 0
    best_mask = 0
    
    # Run multiple restarts with random orderings
    # 200 restarts should be fast enough.
    # Each restart: 512 steps * ~336 checks * 1 bit op.
    # ~170k ops per restart. 200 restarts = 34M ops.
    # Python might take ~1-2 seconds for 34M simple ops.
    # We can increase restarts to 1000 if needed, but 200 is a safe start.
    # Let's try 500 restarts to be safe on quality.
    
    num_restarts = 500
    
    for _ in range(num_restarts):
        current_mask = 0
        # Random permutation of indices 0 to n-1
        perm = list(range(n))
        random.shuffle(perm)
        
        for i in perm:
            # Check if adding i creates a 4-AP
            can_add = True
            for cm in conflicts[i]:
                if (current_mask & cm) == cm:
                    can_add = False
                    break
            
            if can_add:
                current_mask |= (1 << i)
                
        size = bin(current_mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = current_mask
            
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    return result
