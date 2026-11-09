import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency matrix for O(1) edge checks
    adj = [[False]*n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        
    # Precompute all valid subsets (cliques or independent sets)
    # Represent subsets as bitmasks
    valid_sets = []
    for mask in range(1, 1 << n):
        # Extract vertices
        verts = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
            
        is_clique = True
        is_independent = True
        
        for i in range(len(verts)):
            for j in range(i+1, len(verts)):
                if adj[verts[i]][verts[j]]:
                    is_independent = False
                else:
                    is_clique = False
                if not is_clique and not is_independent:
                    break
            if not is_clique and not is_independent:
                break
                
        if is_clique or is_independent:
            valid_sets.append(mask)
            
    # Sort valid sets by size descending to try larger sets first?
    # Actually, the "smallest unassigned vertex" strategy makes order less critical for correctness, 
    # but trying larger sets might cover more vertices, reducing depth.
    valid_sets.sort(key=lambda x: bin(x).count('1'), reverse=True)
    
    # Helper to get smallest unassigned vertex
    def get_smallest_unassigned(assigned_mask):
        unassigned = ((1 << n) - 1) ^ assigned_mask
        if unassigned == 0:
            return -1
        lsb = unassigned & -unassigned
        return lsb.bit_length() - 1

    # Backtracking search
    def search(k, assigned_mask, current_parts):
        if assigned_mask == (1 << n) - 1:
            return current_parts[:]
        if k == 0:
            return None
            
        v = get_smallest_unassigned(assigned_mask)
        if v == -1:
            return None
            
        # Try all valid sets containing v
        for s in valid_sets:
            if (s & (1 << v)) and (s & assigned_mask) == 0:
                # Check if this set is valid to add
                # It is valid by construction.
                # Recurse
                res = search(k-1, assigned_mask | s, current_parts + [s])
                if res is not None:
                    return res
        return None

    # Try k from 1 to n
    for k in range(1, n+1):
        result_masks = search(k, 0, [])
        if result_masks is not None:
            # Convert masks to lists of vertices
            parts = []
            for mask in result_masks:
                verts = []
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    verts.append(idx)
                    m ^= lsb
                parts.append(verts)
            return parts
            
    # Fallback (should not happen)
    return [[i] for i in range(n)]
