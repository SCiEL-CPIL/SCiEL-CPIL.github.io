"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin out of 90s
    
    N = 12
    LCM_PERIOD = 27720  # lcm(1..12)
    MAX_L = 144
    
    def can_match(m, L):
        """Check if there exist distinct b_1..b_12 in (m, m+L] with i|b_i."""
        # For each i, compute candidates: multiples of i in (m, m+L]
        candidates = []
        for i in range(1, N+1):
            # Multiples of i in (m, m+L]: first multiple > m, last <= m+L
            first = (m // i + 1) * i
            last = (m + L) // i * i
            if first > m + L:
                return False
            cands = list(range(first, last + 1, i))
            candidates.append(cands)
        
        # Order by number of candidates (ascending) for better pruning
        order = sorted(range(N), key=lambda i: len(candidates[i]))
        
        # Backtracking
        used = set()
        
        def backtrack(idx):
            if idx == N:
                return True
            i = order[idx]
            for c in candidates[i]:
                if c not in used:
                    used.add(c)
                    if backtrack(idx + 1):
                        return True
                    used.remove(c)
            return False
        
        return backtrack(0)
    
    def compute_f(m):
        """Compute f(N, m): least positive L such that matching exists."""
        # Binary search on L
        lo, hi = 1, MAX_L
        if not can_match(m, hi):
            # Shouldn't happen if MAX_L is correct, but be safe
            # Try larger
            while hi <= 200 and not can_match(m, hi):
                hi += 1
            if not can_match(m, hi):
                return 0
        while lo < hi:
            mid = (lo + hi) // 2
            if can_match(m, mid):
                hi = mid
            else:
                lo = mid + 1
        return lo
    
    best_m = 0
    best_f = 0
    
    # Search over one period
    for m in range(LCM_PERIOD):
        if time.time() > deadline:
            break
        f_val = compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
    
    return int(best_m)

# EVOLVE-BLOCK-END
