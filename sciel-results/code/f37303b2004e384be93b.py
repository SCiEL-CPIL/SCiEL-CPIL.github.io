"""Construct 12 distinct positive integers with distinct subset sums, minimizing the largest element."""
# EVOLVE-BLOCK-START
import time

def _all_subsets_distinct(arr):
    """Check if all 2^n subset sums are distinct."""
    n = len(arr)
    sums = set()
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += arr[i]
        if s in sums:
            return False
        sums.add(s)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit
    
    n = 12
    
    # Greedy construction using difference sets
    # S is the set of all subset sums
    # D is the set of all positive differences between elements in S
    # We pick the smallest positive integer x not in D
    
    S = {0}
    D = set()
    seq = []
    
    for i in range(n):
        if time.time() > deadline:
            # Fallback to powers of 2
            return [1 << j for j in range(n)]
            
        if i == 0:
            x = 1
        else:
            # Find smallest x > 0 such that x not in D
            x = 1
            while x in D:
                x += 1
        
        seq.append(x)
        
        # Update S: new_S = S union (S + x)
        new_S = set(S)
        for s in S:
            new_S.add(s + x)
        S = new_S
        
        # Recompute D efficiently using sorted list
        ss = sorted(S)
        m = len(ss)
        D = set()
        for i_s in range(m):
            for j_s in range(i_s + 1, m):
                d = ss[j_s] - ss[i_s]
                D.add(d)
        
        # Early exit if we've hit deadline
        if time.time() > deadline:
            break
    
    # Verify the result
    if len(seq) == n and len(set(seq)) == n and max(seq) <= 1048576:
        if _all_subsets_distinct(seq):
            return seq
    
    # Fallback: return the seed solution
    return [1 << i for i in range(n)]

# EVOLVE-BLOCK-END
