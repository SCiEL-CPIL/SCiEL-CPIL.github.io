import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    best_seq = []
    
    # suffix_sums: list of sums of suffixes of the current sequence
    # e.g., if seq = [a, b, c], suffix_sums = [a+b+c, b+c, c]
    # sums_set: set of all consecutive block sums
    
    def dfs(seq, sums_set, suffix_sums, last_val):
        nonlocal best_seq
        
        # Update best if current is longer
        if len(seq) > len(best_seq):
            best_seq = seq.copy()
            
        # Pruning: Check if we can possibly beat the best length
        # Max elements we can add is 256 - last_val
        if len(seq) + (256 - last_val) <= len(best_seq):
            return
            
        # Time check
        if time.time() - start_time > deadline:
            return
            
        # Try adding next element
        # We iterate x from last_val + 1 to 256
        # Trying smaller x first is a good heuristic for maximizing length (packing)
        
        for x in range(last_val + 1, 257):
            # Check time periodically inside the loop if branching is wide
            if time.time() - start_time > deadline:
                return
            
            # Calculate new suffix sums
            # New suffixes are x + old_suffixes, and x itself
            # old_suffixes are sums of suffixes of current seq
            # New sums to check:
            new_sums = [x + s for s in suffix_sums]
            new_sums.append(x)
            
            # Check for collisions with existing sums
            # Since new_sums elements are distinct from each other,
            # we just need to check intersection with sums_set
            collision = False
            for s in new_sums:
                if s in sums_set:
                    collision = True
                    break
            
            if collision:
                continue
            
            # No collision, proceed
            new_sums_set = sums_set.union(new_sums)
            new_suffix_sums = new_sums  # This is the list of suffix sums for the new sequence
            
            # Recurse
            dfs(seq + [x], new_sums_set, new_suffix_sums, x)
            
            # Backtracking is handled by creating new sets/lists
            
    dfs([], set(), [], 0)
    
    return best_seq
# EVOLVE-BLOCK-END
