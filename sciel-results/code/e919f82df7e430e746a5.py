import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    def bit_count(x):
        # Fallback for older Python
        return bin(x).count('1')
        
    # Try to find a valid graph for n = 22, 21, 20
    for n in [22, 21, 20]:
        if time.time() > deadline - 5:
            break
            
        # Random search
        found = False
        best_graph = None
        
        # Number of attempts
        # We need to be careful with time
        # Let's do a few hundred attempts
        
        for attempt in range(500):
            if time.time() > deadline - 5:
                break
                
            # Generate random graph
            edges = set()
            adj_mask = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        edges.add((i, j))
                        adj_mask[i] |= (1 << j)
                        adj_mask[j] |= (1 << i)
            
            # Verify
            # Check all subsets of size >= 6
            # Optimization: Only check up to a certain size if we find violations?
            # No, we need to ensure NO subset is regular.
            
            valid = True
            # Iterate masks
            # To speed up, we can skip masks with popcount < 6
            # And we can break early if we find a regular one
            
            # Precompute vertices for masks? No.
            # Let's iterate combinations of vertices instead of masks?
            # For n=22, C(22, 6) is 74k. C(22, 7) is 170k.
            # Total subsets ~ 4M.
            # Iterating 4M masks and doing bit operations is fast in C, but in Python loop it's slow.
            
            # Alternative: Iterate combinations of vertices directly.
            # For k in 6..n:
            #   for comb in combinations(range(n), k):
            #      mask = ...
            #      check regularity
            
            # This avoids iterating over all 2^n masks, only the relevant ones.
            # Sum C(n,k) for k>=6 is ~ 4M for n=22.
            # 4M iterations in Python might take ~ 2-5 seconds.
            # This is acceptable if we do it a few times.
            
            for k in range(6, n + 1):
                if time.time() > deadline - 5:
                    valid = False
                    break
                for comb in itertools.combinations(range(n), k):
                    # Build mask
                    mask = 0
                    for v in comb:
                        mask |= (1 << v)
                    
                    # Check regularity
                    # Get degrees
                    # We can compute degrees for all v in comb
                    degs = []
                    for v in comb:
                        degs.append((adj_mask[v] & mask).bit_count())
                    
                    # Check if all equal
                    if degs:
                        first_deg = degs[0]
                        is_reg = True
                        for d in degs[1:]:
                            if d != first_deg:
                                is_reg = False
                                break
                        if is_reg:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                found = True
                best_graph = {'n': n, 'edges': [list(e) for e in edges]}
                break
        
        if found:
            return best_graph
        
    # Fallback: return a known valid graph or the seed
    # The seed is n=5.
    # We should try to return a valid graph for n=20 if we didn't find 21/22.
    # But if the loop above ran out of time or failed to find, we might return n=5.
    # To be safe, if we didn't find a better one, we can try a few more times for n=20 with different seeds?
    # Or just return n=5.
    
    # Actually, if we didn't find anything in 110s, we should return the best we have.
    # Since we start with n=22 and go down, if we fail 22 and 21, we try 20.
    # If we fail 20, we return n=5.
    
    return {'n': 5, 'edges': []}
