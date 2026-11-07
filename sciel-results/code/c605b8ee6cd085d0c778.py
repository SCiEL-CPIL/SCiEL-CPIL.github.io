import math
import time

# EVOLVE-BLOCK-START
DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
N = len(DIVISORS)

def _lcm(a, b):
    return a * b // math.gcd(a, b)

# LCM of all divisors - useful for periodicity
LCM_ALL = 1
for d in DIVISORS:
    LCM_ALL = _lcm(LCM_ALL, d)

def _find_assignment_matching(L, W):
    """
    Try to find distinct multiples of each divisor in [L, L+W].
    Uses bipartite matching (DFS-based) for efficiency.
    Returns list of chosen values (one per divisor, in order) or None.
    """
    R = L + W
    # For each divisor, compute candidate multiples in [L, R]
    candidates = []
    for d in DIVISORS:
        k_start = (L + d - 1) // d
        k_end = R // d
        if k_start > k_end:
            return None
        mults = list(range(k_start * d, k_end * d + 1, d))
        candidates.append(mults)
    
    # Build adjacency: divisor index -> list of value indices
    # Values are the actual integers
    # For matching, we match divisors to values
    # Use DFS-based bipartite matching
    
    # Create a map from value to list of divisors that can use it
    val_to_divs = {}
    for i, mults in enumerate(candidates):
        for v in mults:
            if v not in val_to_divs:
                val_to_divs[v] = []
            val_to_divs[v].append(i)
    
    # For each divisor, sort candidates by value (for determinism)
    # matching[i] = value assigned to divisor i, or -1
    matching = [-1] * N
    
    def dfs(i, visited):
        for v in candidates[i]:
            if visited.get(v, False):
                continue
            visited[v] = True
            # Find which divisor currently has v, if any
            # We need to track which divisor is matched to which value
            # Let's use a different approach: match_val[v] = divisor index
            pass
        return False
    
    # Better approach: use standard bipartite matching
    # match_val: value -> divisor index
    match_val = {}
    
    def try_kuhn(i, visited):
        for v in candidates[i]:
            if v in visited:
                continue
            visited.add(v)
            if v not in match_val or try_kuhn(match_val[v], visited):
                match_val[v] = i
                return True
        return False
    
    # Order divisors by number of candidates (fewest first) for better performance
    order = sorted(range(N), key=lambda i: len(candidates[i]))
    
    for i in order:
        visited = set()
        if not try_kuhn(i, visited):
            return None
    
    # Reconstruct result
    result = [None] * N
    for v, i in match_val.items():
        result[i] = v
    
    # Verify all divisors got assigned
    if any(v is None for v in result):
        return None
    
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    best_result = None
    best_width = float('inf')
    
    # Exploit LCM periodicity: the pattern of multiples repeats every LCM_ALL
    # So we only need to check L in [1, LCM_ALL] for each width
    # But LCM_ALL = 32730, which is manageable
    
    # For each width W, we need to find if there exists L such that
    # all divisors have at least one multiple in [L, L+W]
    # and a valid matching exists
    
    # Minimum width: 9 (10 distinct integers need at least span 9)
    # But realistically larger
    
    # Strategy: iterate W from 9 upward, for each W check L in [1, LCM_ALL]
    # Use the fact that if a solution exists for some L, it also exists for
    # L + k*LCM_ALL for any k >= 0 (within bounds)
    
    max_L_check = min(LCM_ALL, MAX_VAL - 9)  # Check at most one full period
    
    for W in range(9, 300):
        if time.time() > deadline:
            break
        
        found = False
        # Check L values in one period
        # For efficiency, we can step by 1 but limit the range
        # Actually, for small W, most L values won't work, so step by 1 is fine
        # For larger W, more L values work, but we'll find one quickly
        
        for L in range(1, max_L_check + 1):
            if time.time() > deadline:
                break
            
            # Quick feasibility check: each divisor must have at least one multiple in [L, L+W]
            feasible = True
            for d in DIVISORS:
                k_start = (L + d - 1) // d
                k_end = (L + W) // d
                if k_start > k_end:
                    feasible = False
                    break
            if not feasible:
                continue
            
            result = _find_assignment_matching(L, W)
            if result is not None:
                # Verify
                if len(set(result)) == N:
                    if all(isinstance(v, int) and v > 0 and v <= MAX_VAL for v in result):
                        if all(result[i] % DIVISORS[i] == 0 for i in range(N)):
                            width = max(result) - min(result)
                            if width < best_width:
                                best_width = width
                                best_result = result[:]
                            found = True
                            break
        
        if found:
            break
    
    if best_result is None:
        # Fallback
        best_result = DIVISORS[:]
    
    return [int(v) for v in best_result]
# EVOLVE-BLOCK-END
