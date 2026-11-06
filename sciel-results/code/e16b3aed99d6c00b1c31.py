"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def construct():
    deadline = time.time() + 85  # Safety margin
    
    # Precompute modular inverses of 1..10 mod 101
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, 101)
        
    # Precompute all non-empty subsets: (size, residue, mask)
    # mask_to_terms[mask] = list of denominators
    # residue_to_masks[residue] = list of (size, mask)
    mask_to_terms = [None] * (1 << 10)
    residue_to_masks = [[] for _ in range(101)]
    
    for mask in range(1, 1 << 10):
        terms = []
        s = 0
        for x in range(1, 11):
            if mask & (1 << (x - 1)):
                terms.append(x)
                s += inv[x]
        size = len(terms)
        residue = s % 101
        mask_to_terms[mask] = terms
        residue_to_masks[residue].append((size, mask))
        
    # Sort by size for each residue to facilitate early termination in binary search
    for r in range(101):
        residue_to_masks[r].sort()
        
    def hopcroft_karp(adj, n_left=101):
        """
        adj: list of lists. adj[r] contains mask IDs.
        Returns matchL array where matchL[r] = matched mask, or -1.
        """
        matchL = [-1] * n_left
        matchR = [-1] * (1 << 10)  # Index by mask (1..1023)
        
        def bfs():
            dist = [-1] * n_left
            queue = deque()
            for r in range(n_left):
                if matchL[r] == -1:
                    dist[r] = 0
                    queue.append(r)
            found = False
            while queue:
                r = queue.popleft()
                for mask in adj[r]:
                    m = matchR[mask]
                    if m == -1:
                        found = True
                    elif dist[m] == -1:
                        dist[m] = dist[r] + 1
                        queue.append(m)
            return found, dist
            
        def dfs(r, dist, iter_ptr):
            for i in range(iter_ptr[r], len(adj[r])):
                iter_ptr[r] = i + 1
                mask = adj[r][i]
                m = matchR[mask]
                if m == -1 or (dist[m] == dist[r] + 1 and dfs(m, dist, iter_ptr)):
                    matchL[r] = mask
                    matchR[mask] = r
                    return True
            dist[r] = -1
            return False
            
        matching_size = 0
        while True:
            if time.time() > deadline - 2:
                break
            found, dist = bfs()
            if not found:
                break
            iter_ptr = [0] * n_left
            for r in range(n_left):
                if matchL[r] == -1:
                    if dfs(r, dist, iter_ptr):
                        matching_size += 1
        return matchL, matching_size
        
    def can_match(K):
        """Check if perfect matching exists with all subset sizes <= K."""
        adj = [[] for _ in range(101)]
        for r in range(101):
            for size, mask in residue_to_masks[r]:
                if size <= K:
                    adj[r].append(mask)
                else:
                    break
        _, matching_size = hopcroft_karp(adj)
        return matching_size == 101
        
    # Binary search for minimum K
    lo, hi = 1, 10
    best_K = 10
    while lo <= hi:
        if time.time() > deadline - 5:
            break
        mid = (lo + hi) // 2
        if can_match(mid):
            best_K = mid
            hi = mid - 1
        else:
            lo = mid + 1
            
    # Construct the actual assignment using best_K
    adj = [[] for _ in range(101)]
    for r in range(101):
        for size, mask in residue_to_masks[r]:
            if size <= best_K:
                adj[r].append(mask)
            else:
                break
                
    matchL, matching_size = hopcroft_karp(adj)
    
    # Build result
    result = [None] * 101
    for r in range(101):
        mask = matchL[r]
        if mask != -1:
            result[r] = mask_to_terms[mask]
        else:
            # Fallback (should not happen if best_K is correct)
            if residue_to_masks[r]:
                size, mask = residue_to_masks[r][0]
                result[r] = mask_to_terms[mask]
            else:
                result[r] = [1]
                
    return result
# EVOLVE-BLOCK-END
