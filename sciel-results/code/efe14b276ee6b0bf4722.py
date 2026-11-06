def _has_independent_set_of_size_10(n, S):
    # Returns True if alpha >= 10
    adj_mask = [0] * n
    for u in range(n):
        for v in S:
            adj_mask[u] |= (1 << ((u + v) % n))
            adj_mask[u] |= (1 << ((u - v) % n))
    
    found = [False]
    
    def popcount(x):
        return bin(x).count('1')
        
    def backtrack(candidates, size):
        if found[0]:
            return
        if size == 10:
            found[0] = True
            return
        if candidates == 0:
            return
        if size + popcount(candidates) < 10:
            return
            
        v = (candidates & -candidates).bit_length() - 1
        
        # Try to include v
        new_candidates = candidates & ~(1 << v) & ~adj_mask[v]
        backtrack(new_candidates, size + 1)
        
        if found[0]:
            return
            
        # Try to exclude v
        backtrack(candidates & ~(1 << v), size)
        
    all_vertices = (1 << n) - 1
    backtrack(all_vertices, 0)
    return found[0]
