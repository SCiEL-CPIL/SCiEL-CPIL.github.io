import random

def construct():
    n = 512
    
    def can_add(S, x):
        # S is an integer bitmask. Bit i is 1 if (i+1) is in S.
        # x is 1-indexed.
        
        # Case 1: x = a
        # Need x+d, x+2d, x+3d in S
        max_d = (n - x) // 3
        for d in range(1, max_d + 1):
            if ((S >> (x - 1 + d)) & 1) and ((S >> (x - 1 + 2*d)) & 1) and ((S >> (x - 1 + 3*d)) & 1):
                return False
        
        # Case 2: x = a+d
        # Need x-d, x+d, x+2d in S
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            if ((S >> (x - 1 - d)) & 1) and ((S >> (x - 1 + d)) & 1) and ((S >> (x - 1 + 2*d)) & 1):
                return False
        
        # Case 3: x = a+2d
        # Need x-2d, x-d, x+d in S
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            if ((S >> (x - 1 - 2*d)) & 1) and ((S >> (x - 1 - d)) & 1) and ((S >> (x - 1 + d)) & 1):
                return False
        
        # Case 4: x = a+3d
        # Need x-3d, x-2d, x-d in S
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if ((S >> (x - 1 - 3*d)) & 1) and ((S >> (x - 1 - 2*d)) & 1) and ((S >> (x - 1 - d)) & 1):
                return False
        
        return True

    best_size = 0
    best_set_bits = 0
    
    # Run many randomized greedy attempts
    # Bitset operations are fast, allowing more iterations.
    for _ in range(3000):
        S = 0
        # Random permutation
        perm = list(range(1, n + 1))
        random.shuffle(perm)
        
        for x in perm:
            if can_add(S, x):
                S |= (1 << (x - 1))
        
        size = bin(S).count('1')
        if size > best_size:
            best_size = size
            best_set_bits = S
    
    # Convert bitmask to sorted list
    result = []
    for i in range(n):
        if (best_set_bits >> i) & 1:
            result.append(i + 1)
            
    return result
