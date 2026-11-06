def construct():
    import time
    from itertools import combinations
    from math import gcd
    
    deadline = time.time() + 55  # 55s safety margin
    
    n = 128
    # Elements are 1..128, index 0..127
    
    # Precompute lcm
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # Find all bad triples
    bad_triples = []
    # For each pair (i,j), find k such that lcm(i,j)==lcm(i,k)==lcm(j,k)
    # Precompute: for each pair (i,j), the set of k that form a bad triple
    
    # First, compute all LCMs
    # For efficiency, let's find bad triples by iterating
    
    # For each unordered pair (a,b) with a<b, compute L = lcm(a,b)
    # Then find all c ≠ a,b such that lcm(a,c) == L and lcm(b,c) == L
    # If there are at least 2 such c's (or 1 c and we need 3 total), we have bad triples
    
    # Actually: {a,b,c} is bad iff lcm(a,b)==lcm(a,c)==lcm(b,c)
    # So for fixed (a,b), find all c such that lcm(a,c)==lcm(a,b) and lcm(b,c)==lcm(a,b)
    
    # Precompute conflict structure using bitsets
    # Number of pairs: C(128,2) = 8128
    # Map pair (i,j) with i<j to index
    
    pair_to_idx = {}
    idx_to_pair = []
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            pair_to_idx[(i,j)] = idx
            idx_to_pair.append((i,j))
            idx += 1
    num_pairs = idx
    
    # For each element x (0-indexed, representing x+1), 
    # forbidden_pairs[x] is a bitset over pairs: bit p set if pair p with x forms a bad triple
    
    # First find all bad triples
    bad_triple_set = set()
    third_elem = [set() for _ in range(n)]  # third_elem[i][j] = set of k such that {i,j,k} bad, i<j
    
    for i in range(n):
        for j in range(i+1, n):
            L = lcm(i+1, j+1)
            # Find all k != i,j such that lcm(i+1,k+1)==L and lcm(j+1,k+1)==L
            for k in range(n):
                if k == i or k == j:
                    continue
                if lcm(i+1, k+1) == L and lcm(j+1, k+1) == L:
                    # {i,j,k} is a bad triple
                    a, b, c = sorted([i, j, k])
                    bad_triple_set.add((a, b, c))
    
    # Now build forbidden_pairs for each element
    # forbidden_pairs[x]: bitset where bit p is set if pair idx_to_pair[p] with x forms a bad triple
    forbidden_pairs = [0] * n
    for (a, b, c) in bad_triple_set:
        # Pairs: (a,b), (a,c), (b,c)
        for (p, q) in [(a,b), (a,c), (b,c)]:
            if p > q:
                p, q = q, p
            pair_idx = pair_to_idx[(p,q)]
            # This pair conflicts with the third element
            third = a+b+c-p-q
            forbidden_pairs[third] |= (1 << pair_idx)
    
    # Branch and bound
    # State: current set as bitset, active pairs as bitset, remaining elements
    best = 0
    best_set = set()
    
    # Order elements by number of conflicts (degree) for better pruning
    degrees = [bin(forbidden_pairs[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda x: -degrees[x])
    
    def popcount(x):
        return bin(x).count('1')
    
    def dfs(current_mask, active_pairs, remaining, current_size):
        nonlocal best, best_set, deadline
        
        if time.time() > deadline:
            return
        
        # Upper bound
        if current_size + popcount(remaining) <= best:
            return
        
        if remaining == 0:
            if current_size > best:
                best = current_size
                best_set = set(i for i in range(n) if (current_mask >> i) & 1)
            return
        
        # Try to find a valid element to add
        # For each element in remaining, check if it's valid
        valid_elements = []
        rem = remaining
        while rem:
            # Get lowest set bit
            lsb = rem & (-rem)
            x = lsb.bit_length() - 1
            rem ^= lsb
            
            # Check if x is valid: no active pair conflicts with x
            if (active_pairs & forbidden_pairs[x]) == 0:
                valid_elements.append(x)
        
        if not valid_elements:
            if current_size > best:
                best = current_size
                best_set = set(i for i in range(n) if (current_mask >> i) & 1)
            return
        
        # Try adding elements in order of degree (most constrained first)
        valid_elements.sort(key=lambda x: -degrees[x])
        
        for x in valid_elements:
            if time.time() > deadline:
                return
            
            new_mask = current_mask | (1 << x)
            new_remaining = remaining & ~(1 << x)
            
            # Update active pairs: add pairs (x, y) for all y in current_mask
            new_active = active_pairs
            cm = current_mask
            while cm:
                lsb = cm & (-cm)
                y = lsb.bit_length() - 1
                cm ^= lsb
                if x < y:
                    p = pair_to_idx[(x, y)]
                else:
                    p = pair_to_idx[(y, x)]
                new_active |= (1 << p)
            
            dfs(new_mask, new_active, new_remaining, current_size + 1)
    
    full_mask = (1 << n) - 1
    dfs(0, 0, full_mask, 0)
    
    # Convert to 1-indexed
    result = sorted(i+1 for i in best_set)
    if not result:
        result = [1]
    return result
