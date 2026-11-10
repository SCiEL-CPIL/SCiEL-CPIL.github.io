"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = 27720
N = 12
MAX_L = 144

def _has_matching(m, L):
    """
    Checks if there is a perfect matching between indices 1..N and positions 1..L
    in the interval (m, m+L] such that i divides m+j.
    Uses bitmask DP over subsets of indices.
    """
    # For each index i (1..12), compute bitmask of positions j in 1..L where (m+j) % i == 0
    # Position j corresponds to value m+j
    masks = [0] * (N + 1)  # masks[i] is bitmask of valid positions for index i
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, L + 1):
            if (m + j) % i == 0:
                mask |= (1 << (j - 1))  # position j-1 in 0-indexed bits
        masks[i] = mask
        if mask == 0:
            return False
    
    # Bitmask DP: dp[mask] = True if we can assign the first popcount(mask) indices
    # to distinct positions indicated by mask
    # Actually, let's do: dp[s] = True if indices 1..k (where k = popcount(s)) can be matched
    # to positions in set s
    # Better: iterate over subsets of positions, track which indices are assigned
    
    # Alternative DP: dp[i][pos_mask] is too large. Let's use:
    # dp[s] where s is a subset of positions (up to L bits), and we've assigned
    # the first popcount(s) indices. But L can be up to 144, so 2^144 is too large.
    
    # Since N=12 is small, use DP over subsets of indices:
    # dp[S] = True if the set of indices S can be matched to distinct positions
    # But we need to track which positions are used... 
    
    # Better approach: since N=12, use DP where state is (i, pos_mask) 
    # but pos_mask has L bits which is too many.
    
    # Let's use a different formulation: 
    # dp[i][mask] where i is the next index to assign (0..N), mask is which positions used
    # mask has L bits - too large for L=144.
    
    # Alternative: Hall's theorem check. For every subset S of {1,...,12},
    # the number of positions divisible by at least one i in S must be >= |S|.
    # 2^12 = 4096 subsets, each check is O(L) = O(144), so ~589K ops per (m,L) check.
    
    # Let's use Hall's theorem via inclusion-exclusion or direct counting.
    # For each subset S of indices, count positions j in 1..L where (m+j) is divisible by
    # at least one i in S.
    
    # Precompute for each subset S (as bitmask of indices 1..12), the LCM of indices in S
    # Then count = number of j in 1..L such that (m+j) % lcm(S) == 0... no, that's for a single i.
    # For multiple i, need inclusion-exclusion.
    
    # Actually, let's just use the bitmask DP over indices with a different state:
    # Since N=12, we can do: for each permutation-like assignment... no.
    
    # Let's go back to backtracking but with bitmask for positions, and order by most constrained.
    # With N=12 and L<=144, and good ordering, this should be fast.
    
    # Build adjacency: for each i, list of positions j
    adj = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(1, L + 1):
            if (m + j) % i == 0:
                adj[i].append(j - 1)  # 0-indexed
    
    # Order indices by fewest candidates first
    indices = list(range(1, N + 1))
    indices.sort(key=lambda x: len(adj[x]))
    
    # Use bitmask for positions
    def backtrack(idx_pos, pos_mask):
        if idx_pos == N:
            return True
        i = indices[idx_pos]
        for j in adj[i]:
            bit = 1 << j
            if pos_mask & bit == 0:
                if backtrack(idx_pos + 1, pos_mask | bit):
                    return True
        return False
    
    return backtrack(0, 0)

def _compute_f(m):
    """
    Computes f(N, m) using binary search on L.
    """
    lo = 1
    hi = MAX_L
    ans = -1
    
    while lo <= hi:
        mid = (lo + hi) // 2
        if _has_matching(m, mid):
            ans = mid
            hi = mid - 1
        else:
            lo = mid + 1
            
    return ans if ans != -1 else MAX_L

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    best_m = 0
    best_f = -1
    
    # Search over one period (LCM(1..12) = 27720)
    for m in range(P):
        if time.time() > deadline:
            break
            
        f_val = _compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
    
    return int(best_m)
# EVOLVE-BLOCK-END
