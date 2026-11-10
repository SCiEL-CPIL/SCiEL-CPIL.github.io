import time

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 12
    # LCM(1..12) = 27720
    period = 27720
    
    best_m = 0
    best_f = 0
    
    # Precompute indices 1..12
    indices = list(range(1, n + 1))
    
    def check_matching(m, L):
        """
        Checks if there exist distinct b_1..b_n in (m, m+L] such that i | b_i.
        """
        # Generate candidates for each i
        cands = {}
        for i in indices:
            # First multiple of i strictly greater than m
            k_start = (m // i) + 1
            first_val = k_start * i
            # Last multiple of i <= m + L
            k_end = (m + L) // i
            last_val = k_end * i
            
            if first_val > last_val:
                return False
            
            # Store candidates as list of values
            # To optimize, we can just store the range or generate on fly in DFS
            # But for N=12, list is fine.
            cands[i] = list(range(first_val, last_val + 1, i))
            
        # Sort indices by number of candidates (most constrained first)
        sorted_indices = sorted(indices, key=lambda x: len(cands[x]))
        
        # Bitmask or boolean array for used values
        # Values are in (m, m+L], so we can map v to v - m - 1 (0 to L-1)
        used = [False] * (L + 1)
        
        def dfs(pos):
            if pos == n:
                return True
            i = sorted_indices[pos]
            for v in cands[i]:
                idx = v - m - 1
                if not used[idx]:
                    used[idx] = True
                    if dfs(pos + 1):
                        return True
                    used[idx] = False
            return False
            
        return dfs(0)

    def compute_f(m):
        """
        Computes the least positive L such that matching exists.
        """
        # Binary search for L
        # Max L is small. 144 is a safe upper bound from prompt.
        lo = 1
        hi = 144
        
        # If even L=144 doesn't work, something is wrong, but it should.
        # Actually, for any m, there is always a solution for large enough L.
        # We assume 144 is sufficient.
        
        while lo < hi:
            mid = (lo + hi) // 2
            if check_matching(m, mid):
                hi = mid
            else:
                lo = mid + 1
        return lo

    for r in range(period):
        if time.time() > deadline:
            break
        
        f_val = compute_f(r)
        if f_val > best_f:
            best_f = f_val
            best_m = r

    return int(best_m)

# EVOLVE-BLOCK-END
