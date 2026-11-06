"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def construct():
    deadline = time.time() + 85  # 85s safety margin within 90s limit
    
    # Precompute modular inverses of 1..10 mod 101
    inv = [pow(x, -1, 101) for x in range(1, 11)]
    
    # Precompute all non-empty subsets: (size, residue, mask)
    # Also precompute mask -> list of terms
    mask_to_terms = {}
    all_subsets = []  # (size, residue, mask)
    for mask in range(1, 1 << 10):
        terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
        size = len(terms)
        residue = sum(inv[x - 1] for x in terms) % 101
        all_subsets.append((size, residue, mask))
        mask_to_terms[mask] = terms
    
    # Group subsets by residue, sorted by size
    residue_to_masks = [[] for _ in range(101)]
    for size, residue, mask in all_subsets:
        residue_to_masks[residue].append((size, mask))
    
    for r in range(101):
        residue_to_masks[r].sort()
    
    # Precompute adjacency for each K: adj_K[r] = list of masks for residue r with size <= K
    # We'll build these incrementally
    # adj_by_K[K][r] = list of masks
    max_K = 10
    adj_by_K = [None] * (max_K + 1)
    for K in range(1, max_K + 1):
        adj_by_K[K] = [[] for _ in range(101)]
        for r in range(101):
            for size, mask in residue_to_masks[r]:
                if size <= K:
                    adj_by_K[K][r].append(mask)
                else:
                    break
    
    def hopcroft_karp(adj):
        """Run Hopcroft-Karp. Returns matchL (list of 101 masks) or None if no perfect matching."""
        # Left: residues 0..100, Right: masks
        matchL = [-1] * 101
        matchR = {}  # mask -> residue
        
        # For efficiency, use arrays instead of dict for matchR
        # Masks range from 1 to 1023, so use an array of size 1024
        matchR_arr = [-1] * 1024  # matchR_arr[mask] = residue or -1
        
        def bfs():
            dist = [-1] * 101
            queue = deque()
            for r in range(101):
                if matchL[r] == -1:
                    dist[r] = 0
                    queue.append(r)
            found = False
            while queue:
                r = queue.popleft()
                for mask in adj[r]:
                    m = matchR_arr[mask]
                    if m == -1:
                        found = True
                    elif dist[m] == -1:
                        dist[m] = dist[r] + 1
                        queue.append(m)
            return found, dist
        
        def dfs(r, dist):
            for mask in adj[r]:
                m = matchR_arr[mask]
                if m == -1 or (dist[m] == dist[r] + 1 and dfs(m, dist)):
                    matchL[r] = mask
                    matchR_arr[mask] = r
                    return True
            dist[r] = -1
            return False
        
        matching_size = 0
        while True:
            if time.time() > deadline - 3:
                return None
            found, dist = bfs()
            if not found:
                break
            for r in range(101):
                if matchL[r] == -1:
                    if dfs(r, dist):
                        matching_size += 1
                    if matching_size == 101:
                        return matchL
        if matching_size == 101:
            return matchL
        return None
    
    # Try K in increasing order (likely small)
    best_K = None
    best_matchL = None
    for K in range(1, max_K + 1):
        if time.time() > deadline - 5:
            break
        # Quick feasibility check: each residue must have at least one mask of size <= K
        feasible = True
        for r in range(101):
            if not adj_by_K[K][r]:
                feasible = False
                break
        if not feasible:
            continue
        matchL = hopcroft_karp(adj_by_K[K])
        if matchL is not None:
            best_K = K
            best_matchL = matchL
            break
    
    # If no K worked (shouldn't happen), fall back to K=10
    if best_K is None:
        best_K = 10
        best_matchL = hopcroft_karp(adj_by_K[10])
        if best_matchL is None:
            # Last resort: greedy assignment
            best_matchL = [-1] * 101
            used_masks = set()
            for r in range(101):
                for size, mask in residue_to_masks[r]:
                    if mask not in used_masks:
                        best_matchL[r] = mask
                        used_masks.add(mask)
                        break
                if best_matchL[r] == -1:
                    # Find any unused mask for this residue
                    for size, mask in residue_to_masks[r]:
                        if mask not in used_masks:
                            best_matchL[r] = mask
                            used_masks.add(mask)
                            break
    
    # Build result
    result = [None] * 101
    for r in range(101):
        mask = best_matchL[r]
        if mask is not None and mask != -1 and mask in mask_to_terms:
            result[r] = list(mask_to_terms[mask])
        else:
            # Fallback
            if residue_to_masks[r]:
                size, mask = residue_to_masks[r][0]
                result[r] = list(mask_to_terms[mask])
            else:
                result[r] = [1]
    
    # Verify all residues are covered
    for r in range(101):
        if result[r] is None or len(result[r]) == 0:
            result[r] = [1]
    
    return result
# EVOLVE-BLOCK-END
