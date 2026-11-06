"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import lcm
    from itertools import combinations
    
    n = 128
    # Precompute LCMs
    lcm_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(1, n+1):
            lcm_table[i][j] = lcm(i, j)
    
    # Find all bad triples: (a,b,c) distinct with lcm(a,b)=lcm(a,c)=lcm(b,c)
    # Represent as: for each pair (i,j), the set of k such that (i,j,k) is a bad triple
    # Use bitsets for efficiency
    
    # bad[i][j] = bitmask of k > j such that (i,j,k) is bad (with i < j < k)
    # Actually, let's use: for each pair (i,j) with i<j, bad_pair[i][j] = set of k where triple is bad
    
    # Better approach: for branch and bound, we need for a candidate element x,
    # given current selected set S, check if adding x creates a bad triple.
    # This happens if there exist a,b in S such that (a,b,x) is a bad triple.
    
    # Precompute: for each x, and for each pair (a,b) with a<b, whether (a,b,x) is bad.
    # Store as: bad_with[x] = set of pairs (a,b) such that (a,b,x) is a bad triple.
    # Or better: conflict[x][y] = set of z such that (x,y,z) is bad.
    
    # Let's use: for each element x, bad_pairs[x] = list of (a,b) with a<b, a!=x, b!=x, such that (min(a,b,x), mid, max) is bad.
    # Actually simpler: precompute all bad triples as frozensets, then for each x, 
    # bad_neighbors[x] = set of pairs (a,b) such that {a,b,x} is a bad triple.
    
    # For efficiency in branch-and-bound, let's use:
    # For each x, and for each y < x, precompute the set of z > y such that (y, x, z) or (y, z, x) etc. is bad.
    
    # Let me use a different representation:
    # conflict[i][j] = bitmask over k where k != i, k != j, and (i,j,k) is a bad triple.
    # Since n=128, we can use Python integers as bitsets.
    
    conflict = [[0]*n for _ in range(n)]  # 0-indexed: elements 0..127 represent 1..128
    
    for i in range(n):
        for j in range(i+1, n):
            li = i+1
            lj = j+1
            l = lcm_table[li][lj]
            mask = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                lk = k+1
                if lcm_table[li][lk] == l and lcm_table[lj][lk] == l:
                    mask |= (1 << k)
            conflict[i][j] = mask
            conflict[j][i] = mask
    
    # Now branch and bound for maximum independent set in 3-uniform hypergraph.
    # State: current set of selected elements (as bitmask), and we process elements in order.
    # 
    # Key idea: process elements 0,1,2,...,n-1 in order.
    # For each element, decide include or exclude.
    # When considering including element x, check if there exist a,b in selected such that (a,b,x) is bad.
    # This is equivalent to: for each a in selected, check if (a, x) conflict contains any b in selected.
    
    # Better: maintain for the current selected set S, and for each candidate x not in S,
    # the number of pairs (a,b) in S such that (a,b,x) is bad. If this count > 0, x cannot be added.
    
    # For branch and bound with good pruning, let's use the following:
    # Process elements in a fixed order (say 0 to n-1).
    # State: (pos, selected_mask, count)
    # At position pos, we decide whether to include element pos.
    # To check if we can include element pos: for each pair (a,b) where a,b < pos and a,b in selected,
    # check if conflict[a][b] has bit pos set.
    
    # This check is O(|selected|^2) which could be slow. Let's optimize.
    
    # Alternative: for each element x, precompute the set of pairs (a,b) with a<b<x that conflict with x.
    # Then when building S incrementally, for each new element x added, we can check against previous elements.
    
    # Let me use a recursive approach with memoization-like pruning via upper bounds.
    
    # Upper bound: remaining elements + current count. Simple but effective with good ordering.
    
    import sys
    sys.setrecursionlimit(10000)
    
    best = [0]
    best_set = [0]
    
    # Precompute for each x, the list of pairs (a,b) with a<b<x such that (a,b,x) is bad.
    # bad_pairs_before[x] = list of (a,b) tuples
    bad_pairs_before = [[] for _ in range(n)]
    for x in range(n):
        for a in range(x):
            for b in range(a+1, x):
                if conflict[a][b] & (1 << x):
                    bad_pairs_before[x].append((a, b))
    
    # For branch and bound:
    # We process elements in order 0,1,...,n-1.
    # selected is a bitmask.
    # At step i, we consider element i.
    # can_include(i): for each (a,b) in bad_pairs_before[i], check if both a and b are in selected.
    
    # To speed up: for each i, precompute bad_pairs_before[i] as a list of (a,b).
    # Check: all((selected >> a) & 1 and (selected >> b) & 1 for (a,b) in bad_pairs_before[i]) is False
    
    # Upper bound pruning: if current_count + (n - i) <= best[0], prune.
    
    def can_include(i, selected):
        for a, b in bad_pairs_before[i]:
            if (selected >> a) & 1 and (selected >> b) & 1:
                return False
        return True
    
    def dfs(i, selected, count):
        if i == n:
            if count > best[0]:
                best[0] = count
                best_set[0] = selected
            return
        
        # Upper bound: current count + remaining elements
        if count + (n - i) <= best[0]:
            return
        
        # Option 1: exclude i
        dfs(i+1, selected, count)
        
        # Option 2: include i (if possible)
        if can_include(i, selected):
            new_selected = selected | (1 << i)
            dfs(i+1, new_selected, count + 1)
    
    dfs(0, 0, 0)
    
    # Convert best_set bitmask to list of elements (1-indexed)
    result = []
    for i in range(n):
        if best_set[0] & (1 << i):
            result.append(i+1)
    
    return result
# EVOLVE-BLOCK-END
