import time
import math

# EVOLVE-BLOCK-START
# Strategy:
# The condition for f(12, m) depends on the residues of m+1, ..., m+L modulo 1, ..., 12.
# Since LCM(1, ..., 12) = 27720, the pattern of divisibility repeats every 27720.
# Specifically, the set of valid matchings in (m, m+L] depends only on m mod 27720.
# Therefore, f(12, m) is periodic with period 27720.
# We can compute f(12, r) for all r in [0, 27720) and find the r that maximizes f(12, r).
# Then we choose the largest m in [0, 1000000] such that m % 27720 == r_best.
# This guarantees the maximum possible value of f(12, m) for any m in range.

PERIOD = 27720
N = 12
MAX_M = 1000000

def has_valid_set_fast(m, L):
    """
    Check if there is a perfect matching for indices 1..12 to distinct numbers
    in (m, m+L] such that i divides b_i.
    Uses a bitmask DP over the indices, processing numbers in the interval.
    However, since L is small (<=144), a simpler backtracking with pruning
    or Hopcroft-Karp is fine. 
    For speed in the precomputation loop (27720 * ~10 checks), we need this to be very fast.
    
    Let's use a recursive matching with memoization on (idx, used_mask) is not possible 
    because used_mask has 144 bits.
    
    Alternative: 
    Since we just need to know if a matching exists, we can use the standard 
    augmenting path algorithm (Kuhn's algorithm) which is O(V*E).
    V = 12 + L <= 156. E is at most 12*L.
    This is very fast.
    """
    # Left nodes: 0..11 (representing indices 1..12)
    # Right nodes: 0..L-1 (representing m+1..m+L)
    
    # match_right[j] = i means right node j is matched to left node i
    match_right = [-1] * L
    
    def dfs(u, visited):
        i = u + 1
        # Candidates for i are numbers in (m, m+L] divisible by i
        # These are m + k where k in 1..L and (m+k) % i == 0
        # Instead of iterating all j, we can iterate multiples of i in the range.
        # But iterating j is simpler and L is small.
        for j in range(L):
            if visited[j]:
                continue
            num = m + 1 + j
            if num % i == 0:
                visited[j] = True
                if match_right[j] == -1 or dfs(match_right[j], visited):
                    match_right[j] = u
                    return True
        return False
    
    for u in range(N):
        visited = [False] * L
        if not dfs(u, visited):
            return False
    return True

def f_12(m):
    """
    Compute the least positive L such that (m, m+L] has a valid set.
    """
    # Lower bound is 12 (need 12 distinct numbers)
    # Upper bound is 144 (given in problem description, though we can verify)
    # Actually, the max L is bounded by the worst case gap.
    # Let's binary search L in [12, 144].
    lo, hi = 12, 144
    while lo < hi:
        mid = (lo + hi) // 2
        if has_valid_set_fast(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start_time = time.time()
    
    # 1. Find the best residue r in [0, PERIOD)
    best_r = 0
    best_val = 0
    
    # We can speed this up by parallelization if available, but standard lib only.
    # 27720 iterations * (binary search ~7 steps * matching check ~100 ops) is ~20M ops.
    # In Python, this might take a few seconds.
    
    # Optimization: 
    # f(12, m) is determined by the local structure.
    # Let's just iterate.
    
    # To save time, we can check if we have exceeded a time limit and return a heuristic.
    # But 27720 is small enough.
    
    for r in range(PERIOD):
        if time.time() - start_time > 70: # Safety margin
            # If we time out, return a known good value or the best found so far
            break
        
        val = f_12(r)
        if val > best_val:
            best_val = val
            best_r = r
    
    # 2. Map best_r to the largest m in [0, MAX_M] with m % PERIOD == best_r
    # m = k * PERIOD + best_r
    # We want max m <= MAX_M.
    # k_max = (MAX_M - best_r) // PERIOD
    # m = k_max * PERIOD + best_r
    
    if best_r > MAX_M:
        m = best_r
    else:
        k_max = (MAX_M - best_r) // PERIOD
        m = k_max * PERIOD + best_r
        
    # Ensure m is within bounds
    if m < 0:
        m = 0
    if m > MAX_M:
        m = MAX_M
        
    # Verify the result is a valid integer
    if not isinstance(m, int):
        m = int(m)
        
    return m
# EVOLVE-BLOCK-END
