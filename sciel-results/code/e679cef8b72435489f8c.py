import time

def construct():
    start_time = time.time()
    deadline = start_time + 85

    P = 27720
    n = 12
    max_L = 144

    # Precompute bitmask of valid positions for each (i, r, L)
    # For each i (1..12) and r (0..P-1), the valid positions in (m, m+L]
    # where m % P = r are positions j (1..L) such that (r + j) % i == 0
    # i.e., j % i == (-r) % i = (i - r % i) % i
    # So valid positions are j where j % i == (i - r % i) % i
    # For a given L, the valid positions are: first_valid, first_valid+i, first_valid+2i, ... < L
    # where first_valid = (i - r % i) % i, but if first_valid == 0, it's i (since j>=1)
    
    # Precompute for each (i, r) the list of valid j values for each L
    # Actually, for a fixed (i, r), the valid j's are periodic with period i
    # Let's precompute the "offset" for each (i, r): the smallest j >= 1 with j % i == (i - r%i) % i
    # Then valid positions are offset, offset+i, offset+2i, ...
    
    offsets = [[0]*P for _ in range(n+1)]
    for i in range(1, n+1):
        for r in range(P):
            target = (i - r % i) % i
            if target == 0:
                offsets[i][r] = i
            else:
                offsets[i][r] = target

    # For each (i, r, L), build a bitmask of valid positions 0..L-1 (position j-1 for j in 1..L)
    # valid[j-1] = 1 if j % i == offsets[i][r] % i (but j >= offsets[i][r])
    # Actually j is valid if j >= offsets[i][r] and (j - offsets[i][r]) % i == 0
    # i.e., j = offsets[i][r] + k*i for k >= 0, j <= L
    
    # Build bitmasks: for each i (1..12), r (0..P-1), L (1..max_L)
    # mask[i][r][L] has bit (j-1) set if j is valid
    # But that's 12 * 27720 * 144 bitmasks... too much memory potentially
    # Instead, for each (i, r), precompute the list of valid j's up to max_L
    # Then for a given L, the valid j's are those <= L
    
    valid_js = [[[0]*max_L for _ in range(P)] for _ in range(n+1)]
    # valid_js[i][r][L] = bitmask of positions 0..L-1 that are valid for divisor i with residue r
    for i in range(1, n+1):
        for r in range(P):
            off = offsets[i][r]
            # Build incrementally
            mask = 0
            for L in range(1, max_L+1):
                if L >= off and (L - off) % i == 0:
                    mask |= (1 << (L-1))
                valid_js[i][r][L] = mask

    # Now for matching: given r and L, we have for each i (1..12) a bitmask of valid positions
    # We need to find if there's a perfect matching in the bipartite graph
    # Left: divisors 1..12, Right: positions 1..L
    # Edge (i, j) exists if j is in valid_js[i][r][L]
    
    # Use Hopcroft-Karp or simple DFS with bit operations
    # Since n=12 is small, we can use a recursive matching with bitsets
    
    def has_matching(r, L):
        # For each divisor i, get the set of valid positions as a bitmask
        # We need to assign each divisor to a distinct position
        # Use DFS with backtracking, trying divisors in order of fewest choices first
        
        # Count choices for each divisor
        counts = []
        for i in range(1, n+1):
            mask = valid_js[i][r][L]
            cnt = bin(mask).count('1')
            if cnt == 0:
                return False
            counts.append((cnt, i))
        
        counts.sort()
        
        # If any divisor has more choices than L, that's fine, but if total choices < n, fail
        # Actually, just do the matching
        
        # match_pos[j] = which divisor is matched to position j, or -1
        match_pos = [-1] * L  # positions 0..L-1
        
        def dfs(i, visited):
            mask = valid_js[i][r][L]
            # Try each valid position
            m = mask
            while m:
                # Get lowest set bit
                bit = m & (-m)
                j = bit.bit_length() - 1  # position index (0-based)
                m ^= bit
                if visited[j]:
                    continue
                visited[j] = True
                if match_pos[j] == -1 or dfs(match_pos[j], visited):
                    match_pos[j] = i
                    return True
            return False
        
        for cnt, i in counts:
            visited = [False] * L
            if not dfs(i, visited):
                return False
        return True

    best_L = 0
    best_r = 0

    for r in range(P):
        if time.time() > deadline:
            break

        # Binary search for the minimum L such that has_matching(r, L) is True
        # f(12, r) = smallest L >= 12 with matching
        # But we want to maximize f, so we want the largest r where the min L is large
        
        # First check if L=144 works (max possible)
        if not has_matching(r, max_L):
            # Even max L doesn't work, so f is undefined or > 144
            # But the problem says f is at most 144, so this shouldn't happen
            # Actually, if no matching exists even at L=144, then f(12,r) is not defined
            # But the task says "the least positive L such that...", implying it exists
            # Let's just skip this r or treat it as having very high f
            # Actually, re-reading: "the LEAST positive L such that (m,m+L] contains distinct b_1,...,b_12"
            # If no such L exists, f is undefined. But the task says it's at most 144.
            # Let's check: for L=144, each divisor i has about 144/i positions
            # For i=12, that's 12 positions. Should be enough.
            # If it fails, something is wrong. Let's continue.
            continue

        # Binary search for smallest L with matching
        lo, hi = 12, max_L
        while lo < hi:
            mid = (lo + hi) // 2
            if has_matching(r, mid):
                hi = mid
            else:
                lo = mid + 1

        if lo > best_L:
            best_L = lo
            best_r = r

    return int(best_r)
