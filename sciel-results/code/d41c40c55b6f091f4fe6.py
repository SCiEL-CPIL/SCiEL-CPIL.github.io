"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys

def construct():
    # Increase recursion limit just in case, though depth is small
    sys.setrecursionlimit(10000)
    
    candidates = list(range(2, 129))
    n = len(candidates)
    best_len = 0
    best_seq = []
    
    def dfs(start_idx, seq, prod_set):
        nonlocal best_len, best_seq
        
        # Update best if current sequence is longer
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
            
        # Pruning: 
        # The maximum number of elements we can still add is (n - start_idx).
        # If even adding all of them doesn't exceed best_len, we prune.
        if len(seq) + (n - start_idx) <= best_len:
            return
            
        for i in range(start_idx, n):
            x = candidates[i]
            
            # Check if x can be appended to seq
            # We need to ensure that no new product ending at x is already in prod_set
            
            # 1. Check if x itself is in prod_set
            if x in prod_set:
                continue
                
            # 2. Check suffix products: x * s_k, x * s_{k-1} * s_k, ...
            # We compute these on the fly.
            # We also collect them in a list to update the set if valid.
            new_prods = [x]
            prod = x
            valid = True
            
            # Iterate backwards through the current sequence
            for a in reversed(seq):
                prod *= a
                if prod in prod_set:
                    valid = False
                    break
                new_prods.append(prod)
                
            if not valid:
                continue
                
            # State update
            prod_set.update(new_prods)
            seq.append(x)
            
            # Recurse
            dfs(i + 1, seq, prod_set)
            
            # Backtrack
            seq.pop()
            prod_set.difference_update(new_prods)

    dfs(0, [], set())
    
    return best_seq

# EVOLVE-BLOCK-END
