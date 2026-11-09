"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from functools import lru_cache

P = {'n': 12, 'maximum_start': 1000000}

# LCM of 1..12
def _lcm(a, b):
    return a // math.gcd(a, b) * b

_LCM = 1
for _i in range(2, 13):
    _LCM = _lcm(_LCM, _i)

# For each residue r in 0..LCM-1, compute f(12, r) = least L such that (r, r+L] has valid matching.
# We precompute this using a lookup table approach.

# Key insight: For a given residue r, the divisibility pattern of (r+1), (r+2), ..., (r+L) 
# depends on r mod lcm(1..12) = 27720. But since L <= 144, we can directly compute.

# Precompute f(12, r) for all r in 0..LCM-1
# Use bipartite matching with early termination

def _max_matching(adj, n_left=12):
    """adj[i] = list of right node indices adjacent to left node i.
    Returns size of maximum matching."""
    n_right = max((max(adj[i]) for i in range(n_left) if adj[i]), default=-1) + 1
    if n_right == 0:
        return 0
    
    match_r = [-1] * n_right
    
    def dfs(u, visited):
        for v in adj[u]:
            if visited[v]:
                continue
            visited[v] = True
            if match_r[v] == -1 or dfs(match_r[v], visited):
                match_r[v] = u
                return True
        return False
    
    result = 0
    for u in range(n_left):
        visited = [False] * n_right
        if dfs(u, visited):
            result += 1
            if result == n_left:
                return n_left
    return result

def _compute_f_for_residue(r, max_L=144):
    """Compute f(12, r) = least L such that (r, r+L] has valid matching."""
    for L in range(1, max_L + 1):
        # Build adjacency: left node i (1-indexed) connects to position j (1-indexed) if (r+j) % i == 0
        adj = [[] for _ in range(12)]
        for i in range(1, 13):
            # Find all j in [1, L] such that (r + j) % i == 0
            # j ≡ (-r) mod i
            rem = (-r) % i
            if rem == 0:
                first_j = i
            else:
                first_j = rem
            
            if first_j <= L:
                j = first_j
                while j <= L:
                    adj[i-1].append(j-1)  # 0-indexed position
                    j += i
        
        if _max_matching(adj) == 12:
            return L
    return max_L

# Precompute f for all residues
def _precompute():
    best_f = 0
    best_r = 0
    for r in range(_LCM):
        f_val = _compute_f_for_residue(r)
        if f_val > best_f:
            best_f = f_val
            best_r = r
    return best_f, best_r

_best_f, _best_r = _precompute()

# Find m in 0..1000000 with m % LCM == _best_r
_k_max = (1000000 - _best_r) // _LCM
if _k_max < 0:
    _best_m = _best_r
else:
    _best_m = _best_r

def construct():
    return int(_best_m)
# EVOLVE-BLOCK-END
