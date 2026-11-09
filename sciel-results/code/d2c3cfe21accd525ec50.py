"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def _compute_f(m, n=12):
    """Compute f(n, m): least positive L such that (m, m+L] has distinct b_1..b_n with i|b_i."""
    for L in range(1, 200):
        # For each i, the candidates in (m, m+L] divisible by i
        # We need to check if we can assign distinct candidates
        # Use greedy: for each i, the earliest available candidate
        # But greedy might not work due to conflicts, so use matching
        
        # Build adjacency: for each i, which numbers in (m, m+L] are divisible by i
        # Numbers are m+1, m+2, ..., m+L
        # For matching, map each number to its index 0..L-1
        
        # adj[i] = set of indices j (0-based) such that (m+1+j) % i == 0
        adj = [set() for _ in range(n + 1)]
        for i in range(1, n + 1):
            # Find all j in 0..L-1 such that (m+1+j) % i == 0
            # (m+1+j) % i == 0 => j ≡ (-m-1) mod i
            rem = (-m - 1) % i
            # j = rem, rem+i, rem+2i, ... < L
            j = rem
            while j < L:
                adj[i].add(j)
                j += i
        
        # Bipartite matching: left side = indices 1..n, right side = positions 0..L-1
        match_right = [-1] * L  # match_right[j] = i that is matched to position j, or -1
        match_left = [-1] * (n + 1)  # match_left[i] = position j matched to i, or -1
        
        def dfs(i, visited):
            for j in adj[i]:
                if visited[j]:
                    continue
                visited[j] = True
                if match_right[j] == -1 or dfs(match_right[j], visited):
                    match_right[j] = i
                    match_left[i] = j
                    return True
            return False
        
        ok = True
        for i in range(1, n + 1):
            visited = [False] * L
            if not dfs(i, visited):
                ok = False
                break
        
        if ok:
            return L
    return 200

def construct():
    """Find m in [0, 1000000] that maximizes f(12, m)."""
    best_m = 12
    best_f = 0
    
    candidates = set()
    
    # Key insight: f(12,m) is large when m is such that for some i (especially large i),
    # the first multiple of i after m is far away. j_i = (-m-1) mod i.
    # For i=12, j_12 is maximized (11) when m ≡ 0 mod 12.
    # For i=11, j_11 is maximized (10) when m ≡ 0 mod 11.
    # So m divisible by 132 = lcm(11,12) is promising.
    # Also check m divisible by lcm of subsets of {7,8,9,10,11,12}.
    
    # Check m values that are multiples of various LCMs
    lcms_to_check = [132, 252, 360, 840, 1260, 2520, 27720]
    for lcm_val in lcms_to_check:
        for k in range(0, 1000000 // lcm_val + 2):
            base = k * lcm_val
            # Check base and nearby values
            for offset in range(-20, 21):
                m_val = base + offset
                if 0 <= m_val <= 1000000:
                    candidates.add(m_val)
    
    # Also specifically check m values where m mod 12 == 0 and m mod 11 == 0 (i.e., m mod 132 == 0)
    # and nearby values, plus m where m mod 12 is in {0,1,2,...} to explore
    for m_val in range(0, 1000001):
        if m_val % 132 == 0:
            for offset in range(-5, 6):
                mv = m_val + offset
                if 0 <= mv <= 1000000:
                    candidates.add(mv)
    
    # Add some random samples for robustness
    import random
    random.seed(42)
    for _ in range(50000):
        candidates.add(random.randint(0, 1000000))
    
    # Also check small values
    for m_val in range(0, 2000):
        candidates.add(m_val)
    
    # Evaluate candidates
    for m_val in candidates:
        f_val = _compute_f(m_val, 12)
        if f_val > best_f:
            best_f = f_val
            best_m = m_val
    
    return best_m

# EVOLVE-BLOCK-END
