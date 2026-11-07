"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

N = 12
MAX_START = 1000000
LCM_12 = 27720  # lcm(1,2,...,12)

def _can_match_fast(m, L):
    """Fast check if (m, m+L] contains distinct b_1..b_N with i | b_i using bitmask matching."""
    n = N
    # For each i (1-indexed), candidates are multiples of i in (m, m+L]
    # Use a list of lists of candidate values
    adj = []
    for i in range(1, n + 1):
        first = ((m // i) + 1) * i
        end = m + L
        if first > end:
            return False
        # Collect candidates
        cands = []
        v = first
        while v <= end:
            cands.append(v)
            v += i
        adj.append(cands)
    
    # Kuhn's algorithm with early termination
    match_to = [-1] * n  # match_to[i] = which left node is matched to right candidate index... 
    # Actually, let's use a different approach: match right side to left side
    # right_match[j] = left index matched to right candidate j
    # But right side is dynamic per L...
    
    # Simpler: use dict-based matching
    right_match = {}  # value -> left index
    
    def dfs(left, seen):
        for val in adj[left]:
            if val in seen:
                continue
            seen.add(val)
            if val not in right_match or dfs(right_match[val], seen):
                right_match[val] = left
                return True
        return False
    
    for i in range(n):
        seen = set()
        if not dfs(i, seen):
            return False
    return True

def _f_val(m):
    """Compute f(N, m): least positive L such that (m, m+L] has distinct b_1..b_N with i|b_i."""
    # For N=12, the answer is at most 144 per task description
    # But let's be safe and check up to 200
    lo = 1
    hi = 150  # task says at most 144
    
    # Check if hi is feasible
    if not _can_match_fast(m, hi):
        hi = 200
        if not _can_match_fast(m, hi):
            hi = 300
            if not _can_match_fast(m, hi):
                return 0
    
    # Binary search for minimum L
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match_fast(m, mid):
            hi = mid
        else:
            lo = mid + 1
    
    return lo

def construct():
    """Find m in [0, MAX_START] that maximizes f(12, m)."""
    start_time = time.time()
    deadline = start_time + 85  # 85s out of 90s budget
    
    # Key insight: f(12, m) depends only on m mod LCM_12 because
    # shifting m by LCM_12 shifts all candidate multiples by LCM_12,
    # and since LCM_12 is divisible by all i in 1..12, divisibility is preserved.
    # So we only need to search m in [0, LCM_12 - 1] = [0, 27719].
    
    best_m = 0
    best_f = 0
    
    # Exhaustive search over the fundamental domain [0, 27719]
    for m in range(LCM_12):
        if time.time() > deadline:
            break
        f = _f_val(m)
        if f > best_f:
            best_f = f
            best_m = m
    
    # If we finished the exhaustive search, verify with a few nearby values
    # and also check m values that map to the same residue but are in range
    if time.time() < deadline:
        # Check the best_m and nearby values
        for dm in range(-5, 6):
            m_test = best_m + dm
            if 0 <= m_test <= MAX_START:
                f = _f_val(m_test)
                if f > best_f:
                    best_f = f
                    best_m = m_test
    
    # Also check m = 0 and m = MAX_START specifically
    for m_check in [0, MAX_START]:
        if time.time() > deadline:
            break
        if m_check != best_m:
            f = _f_val(m_check)
            if f > best_f:
                best_f = f
                best_m = m_check
    
    return int(best_m)

# EVOLVE-BLOCK-END
