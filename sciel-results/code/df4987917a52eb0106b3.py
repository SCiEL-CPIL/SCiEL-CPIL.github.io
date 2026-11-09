import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 90s kill

    P = 27720  # LCM(1..12)
    n = 12
    max_L = 144  # upper bound from contract

    # Precompute masks: for each i in 1..12, for each residue r in 0..P-1,
    # which positions j (0-indexed, representing r+1+j) satisfy (r+1+j) % i == 0
    # But we need this for variable L, so let's think differently.
    
    # For a given r and L, the window is {r+1, ..., r+L}.
    # For each i, the valid positions are j where (r+1+j) % i == 0, i.e., j ≡ (i-1-r) mod i, but only for j in 0..L-1.
    
    # Actually, let me precompute for each r (mod P) and each i, the set of j mod P where (r+1+j) % i == 0.
    # Since (r+1+j) mod i depends only on (r+1+j) mod i, and i divides P, we can precompute.
    
    # For fixed i and r, j is valid iff (r + 1 + j) % i == 0, i.e., j % i == (i - 1 - r) % i
    # So for each i and r, the valid j's are those with j ≡ c (mod i) for some c.
    # In a window of length L, the number of valid j's is either floor(L/i) or ceil(L/i) depending on alignment.
    
    # Let me use a bitmask approach for matching.
    # For each i (1..12) and each residue r (0..P-1), precompute which positions in a "period" are valid.
    # But L varies, so let's just compute on the fly but optimize the matching.
    
    # Alternative: Use Hall's theorem / max flow with bitmasks.
    # For each r, we want to find the minimum L such that a perfect matching exists.
    # We can use the fact that the structure is periodic with period P.
    
    # Let me precompute for each i and each residue class mod i, and then for a given r and L,
    # compute the valid positions quickly.
    
    # Actually, let's just use a fast bipartite matching with the structure exploited.
    
    # For a given r and L:
    # For each i in 1..12, valid[i] = sorted list of j in 0..L-1 where (r+1+j) % i == 0
    
    # Let me precompute for each i, for each r mod i, the pattern.
    # For i, r mod i = s, then j is valid iff (r+1+j) % i == 0 iff (s+1+j) % i == 0 iff j % i == (i-1-s) % i
    # So for each i and s (0..i-1), the valid j's are j = c, c+i, c+2i, ... where c = (i-1-s) % i
    
    # Let me precompute: for each i in 1..12, for each s in 0..i-1, the offset c = (i-1-s) % i
    # Then for a given r, L, and i: s = r % i, c = offset[i][s], valid j's are c, c+i, c+2i, ... < L
    
    offset = [[0]*13 for _ in range(13)]  # offset[i][s] = (i-1-s) % i
    for i in range(1, 13):
        for s in range(i):
            offset[i][s] = (i - 1 - s) % i
    
    def has_matching(r, L):
        """Check if there's a matching for residue r with window size L."""
        # Build valid lists
        valid = [None] * (n + 1)
        for i in range(1, n + 1):
            s = r % i
            c = offset[i][s]
            if c >= L:
                # No valid position
                return False
            valid[i] = list(range(c, L, i))
        
        # Quick check: each i must have at least one valid position
        # Also, for the most constrained i's, check first
        
        # Use Hopcroft-Karp or simple DFS matching
        # Process from most constrained (largest i) to least
        
        match_to = [-1] * L  # match_to[j] = i or -1
        
        def dfs(i, visited):
            for j in valid[i]:
                if visited[j]:
                    continue
                visited[j] = True
                if match_to[j] == -1 or dfs(match_to[j], visited):
                    match_to[j] = i
                    return True
            return False
        
        # Try to match all i
        # Order: 12, 11, 10, ..., 1
        for i in range(n, 0, -1):
            visited = [False] * L
            if not dfs(i, visited):
                return False
        
        return True
    
    # For each residue r, find f(12, r) = smallest L >= 12 with matching
    # We want to maximize this value
    best_L = 0
    best_r = 0
    
    for r in range(P):
        # Check if we're still within deadline
        if time.time() > deadline:
            break
        
        # Find smallest L >= 12 such that has_matching(r, L) is True
        found = False
        for L in range(12, max_L + 1):
            if has_matching(r, L):
                if L > best_L:
                    best_L = L
                    best_r = r
                found = True
                break
    
    # Return best_r as m
    return int(best_r)
