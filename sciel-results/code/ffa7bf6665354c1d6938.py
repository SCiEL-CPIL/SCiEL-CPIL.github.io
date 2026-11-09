"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import time
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_range(n):
    return reduce(_lcm, range(1, n + 1))

LCM_PERIOD = _lcm_range(12)  # 27720

def _has_perfect_matching(adj, n):
    """
    Check if there's a perfect matching in bipartite graph.
    adj[i] = bitmask of right-side nodes available for left node i.
    n = number of left nodes (12).
    Uses DFS-based augmenting path with bitmask optimization.
    """
    # match_r[v] = which left node is matched to right node v, or -1
    # We'll use an array of size 144 (max L)
    match_r = [-1] * 144
    
    def try_augment(u, visited_mask):
        avail = adj[u]
        # Iterate over set bits in avail
        while avail:
            # Get lowest set bit
            v = (avail & (-avail)).bit_length() - 1
            avail &= avail - 1  # clear lowest set bit
            if visited_mask >> v & 1:
                continue
            visited_mask |= (1 << v)
            if match_r[v] == -1 or try_augment(match_r[v], visited_mask):
                match_r[v] = u
                return True
        return False
    
    for u in range(n):
        if try_augment(u, 0):
            pass
        else:
            return False
    return True

def _compute_f(m, n=12, max_L=144):
    """Compute f(n, m): least positive L such that (m, m+L] has distinct b_1..b_n with i|b_i."""
    for L in range(1, max_L + 1):
        # Build adjacency as bitmasks
        adj = []
        for i in range(1, n + 1):
            mask = 0
            for j in range(L):
                if (m + 1 + j) % i == 0:
                    mask |= (1 << j)
            if mask == 0:
                break
            adj.append(mask)
        
        if len(adj) < n:
            continue
        
        # Quick Hall check: union of all adj must have at least n bits
        union = 0
        for mask in adj:
            union |= mask
        if union.bit_count() < n:
            continue
        
        if _has_perfect_matching(adj, n):
            return L
    return max_L

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    # Precompute f(12, r) for all residues r in [0, LCM_PERIOD)
    best_f = 0
    best_residue = 0
    
    for r in range(LCM_PERIOD):
        if time.time() > deadline:
            break
        f_val = _compute_f(r, n=12, max_L=144)
        if f_val > best_f:
            best_f = f_val
            best_residue = r
            if best_f == 144:
                # Maximum possible, can stop early
                break
    
    # Find m in [0, 1000000] with m % LCM_PERIOD == best_residue
    m = best_residue
    if m < 0 or m > 1000000:
        m = 0
    
    return int(m)

# EVOLVE-BLOCK-END
