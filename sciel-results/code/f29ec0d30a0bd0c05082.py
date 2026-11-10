import time
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin
    
    # Precompute subsets
    # mask: 1..1023
    # size: number of bits
    # residue: sum of inverses mod 101
    
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, 101)
        
    mask_size = [0] * 1024
    mask_residue = [0] * 1024
    
    for mask in range(1, 1024):
        size = 0
        res = 0
        m = mask
        x = 1
        while m:
            if m & 1:
                size += 1
                res = (res + inv[x]) % 101
            m >>= 1
            x += 1
        mask_size[mask] = size
        mask_residue[mask] = res

    # We want to find the minimum k such that there is a matching
    # covering all 101 residues using only subsets of size <= k.
    
    # Bipartite matching:
    # Left: Residues 0..100
    # Right: Masks 1..1023
    
    # For a fixed k, we build adjacency list for residues.
    # adj[r] = list of masks m such that mask_residue[m] == r and mask_size[m] <= k
    
    def try_match(k):
        # Build adjacency list
        adj = [[] for _ in range(101)]
        for m in range(1, 1024):
            if mask_size[m] <= k:
                r = mask_residue[m]
                adj[r].append(m)
                
        # Kuhn's algorithm
        # match_r[r] = mask matched to residue r
        # match_m[m] = residue matched to mask m
        match_r = [-1] * 101
        match_m = [-1] * 1024
        
        def dfs(r, visited):
            for m in adj[r]:
                if visited[m]:
                    continue
                visited[m] = True
                if match_m[m] == -1 or dfs(match_m[m], visited):
                    match_m[m] = r
                    match_r[r] = m
                    return True
            return False
            
        count = 0
        for r in range(101):
            visited = [False] * 1024
            if dfs(r, visited):
                count += 1
            if time.time() > deadline:
                return None
            if count < 101:
                # Optimization: if we fail early, we might break? 
                # No, Kuhn's doesn't guarantee order independence for early exit on failure 
                # unless we know it's impossible. But 101 is small.
                pass
                
        if count == 101:
            # Construct result
            result = [None] * 101
            for r in range(101):
                m = match_r[r]
                # Convert mask to list of denominators
                terms = []
                mm = m
                x = 1
                while mm:
                    if mm & 1:
                        terms.append(x)
                    mm >>= 1
                    x += 1
                result[r] = terms
            return result
        return None

    # Try k from 1 to 10
    for k in range(1, 11):
        res = try_match(k)
        if res is not None:
            return res
        if time.time() > deadline:
            # Fallback: return the greedy solution from the seed program
            # to ensure we return *something* valid
            representations = {}
            for mask in range(1, 1 << 10):
                terms = [x for x in range(1, 11) if mask >> (x - 1) & 1]
                residue = sum(pow(x, -1, 101) for x in terms) % 101
                if residue not in representations:
                    representations[residue] = terms
            # If some residues are missing, fill with any valid subset (shouldn't happen)
            # But greedy might miss some? No, greedy covers all because 
            # sum of all inverses is not necessarily 0, but the set of all subsets 
            # covers all residues? 
            # Actually, the seed program returned a valid solution, so it must cover all.
            # However, the seed program didn't minimize size.
            # If we timeout, we return the seed program's logic.
            return [representations[r] for r in range(101)]
            
    # If no k worked (should not happen), return seed logic
    representations = {}
    for mask in range(1, 1 << 10):
        terms = [x for x in range(1, 11) if mask >> (x - 1) & 1]
        residue = sum(pow(x, -1, 101) for x in terms) % 101
        if residue not in representations:
            representations[residue] = terms
    return [representations[r] for r in range(101)]
# EVOLVE-BLOCK-END
