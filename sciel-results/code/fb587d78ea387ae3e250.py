import time
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    # Use 85 seconds to be safe against overhead and GC
    deadline = start_time + 85.0
    
    best_seq = []
    best_len = 0
    
    def dfs(seq, prods, next_min):
        nonlocal best_seq, best_len
        
        # Time check
        if time.time() > deadline:
            return
        
        # Update best
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)
            
        # Pruning
        # Max possible length if we add all remaining numbers
        remaining = 128 - next_min + 1
        if len(seq) + remaining <= best_len:
            return
            
        for x in range(next_min, 129):
            # Check if x can be added
            # New subarrays end at x.
            # Products: x, seq[-1]*x, seq[-2]*seq[-1]*x, ...
            
            valid = True
            new_prods_to_add = []
            
            # Calculate products
            p = x
            new_prods_to_add.append(p)
            if p in prods:
                valid = False
                break
                
            # Multiply by previous elements
            # We can precompute suffix products of seq?
            # No, just loop.
            for i in range(len(seq) - 1, -1, -1):
                p *= seq[i]
                new_prods_to_add.append(p)
                if p in prods:
                    valid = False
                    break
            
            if valid:
                # Optimization: Check time before expensive set update/recursion
                if time.time() > deadline:
                    return
                
                next_prods = set(prods)
                next_prods.update(new_prods_to_add)
                
                dfs(seq + [x], next_prods, x + 1)
                
                if time.time() > deadline:
                    return

    dfs([], set(), 2)
    
    return best_seq
