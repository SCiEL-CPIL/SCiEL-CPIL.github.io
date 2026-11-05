"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

_START = time.time()
_DEADLINE = 85.0

def _elapsed():
    return time.time() - _START

def construct():
    MOD = 101
    N = 10
    
    # Precompute inverses
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, -1, MOD)
    
    # Enumerate all non-empty subsets: map residue -> list of (size, mask)
    res_to_subsets = defaultdict(list)
    for mask in range(1, 1 << N):
        if _elapsed() > _DEADLINE:
            break
        s = 0
        m = mask
        size = 0
        while m:
            lsb = m & (-m)
            x = lsb.bit_length()
            s += inv[x]
            size += 1
            m -= lsb
        r = s % MOD
        res_to_subsets[r].append((size, mask))
    
    # For each residue, sort subsets by size
    for r in res_to_subsets:
        res_to_subsets[r].sort()
    
    # Binary search on max_size K: can we assign all 101 residues using subsets of size <= K?
    # Use bipartite matching: left = residues (101), right = subsets (up to 1023)
    # Edge from residue r to subset s if s's residue == r and s's size <= K
    # We want a matching that covers all 101 residues.
    
    def can_cover(max_size):
        # Build adjacency: for each residue, which subsets (indices) are available
        # Subset index is just the mask (1..1023)
        # Use DFS-based bipartite matching (Kuhn's algorithm)
        # Left nodes: 0..100 (residues)
        # Right nodes: masks 1..1023
        
        # For efficiency, precompute for each residue the list of masks with size <= max_size
        adj = [None] * MOD
        for r in range(MOD):
            if r not in res_to_subsets:
                return False
            adj[r] = [mask for (sz, mask) in res_to_subsets[r] if sz <= max_size]
            if not adj[r]:
                return False
        
        # Kuhn's algorithm
        match_r = [-1] * (1 << N)  # match_r[mask] = residue assigned to this mask, or -1
        
        def dfs(r, visited):
            for mask in adj[r]:
                if visited[mask]:
                    continue
                visited[mask] = True
                if match_r[mask] == -1 or dfs(match_r[mask], visited):
                    match_r[mask] = r
                    return True
            return False
        
        count = 0
        for r in range(MOD):
            visited = [False] * (1 << N)
            if dfs(r, visited):
                count += 1
            if count < MOD:
                # Early exit not possible mid-way, but we can break if remaining can't be covered
                pass
        
        return count == MOD
    
    # Binary search on max_size
    lo, hi = 1, N
    best_K = N
    while lo <= hi:
        mid = (lo + hi) // 2
        if can_cover(mid):
            best_K = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    # Now construct the actual assignment using max_size = best_K
    # Re-run matching to get the actual assignment
    adj = [None] * MOD
    for r in range(MOD):
        if r in res_to_subsets:
            adj[r] = [mask for (sz, mask) in res_to_subsets[r] if sz <= best_K]
        else:
            adj[r] = []
    
    match_r = [-1] * (1 << N)
    
    def dfs2(r, visited):
        for mask in adj[r]:
            if visited[mask]:
                continue
            visited[mask] = True
            if match_r[mask] == -1 or dfs2(match_r[mask], visited):
                match_r[mask] = r
                return True
        return False
    
    for r in range(MOD):
        visited = [False] * (1 << N)
        dfs2(r, visited)
    
    # Build rows
    rows = [None] * MOD
    for r in range(MOD):
        # Find the mask assigned to residue r
        assigned_mask = None
        for mask in range(1, 1 << N):
            if match_r[mask] == r:
                assigned_mask = mask
                break
        if assigned_mask is None:
            # Fallback: use the smallest subset for this residue
            if res_to_subsets[r]:
                assigned_mask = res_to_subsets[r][0][1]
            else:
                assigned_mask = (1 << N) - 1  # full set
        terms = [x for x in range(1, N + 1) if (assigned_mask >> (x - 1)) & 1]
        rows[r] = terms
    
    return rows

# EVOLVE-BLOCK-END
