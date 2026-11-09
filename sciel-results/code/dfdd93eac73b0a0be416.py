import time
import sys

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency matrix/bitmask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Precompute valid masks
    # valid_mask[m] is True if mask m induces a clique or independent set
    valid_mask = [False] * (1 << n)
    valid_mask[0] = True
    valid_mask[1 << 0] = True # Single vertex is both clique and ind set
    
    for mask in range(1, 1 << n):
        # Get vertices in mask
        # Efficient way: check if adding a vertex to a valid submask keeps it valid?
        # No, clique/ind property is not monotonic in a simple way for mixed types.
        # But we can check pairs.
        # Optimization: 
        # A set is a clique if for every v in S, v is adjacent to all other u in S.
        # A set is an ind set if for every v in S, v is not adjacent to any other u in S.
        
        # Let's just iterate vertices in the mask.
        is_clique = True
        is_ind = True
        
        # Extract vertices
        verts = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m -= lsb
            
        # Check pairs
        for i in range(len(verts)):
            u = verts[i]
            for j in range(i+1, len(verts)):
                w = verts[j]
                if (adj[u] >> w) & 1:
                    is_ind = False
                else:
                    is_clique = False
                if not is_clique and not is_ind:
                    break
            if not is_clique and not is_ind:
                break
                
        valid_mask[mask] = is_clique or is_ind

    # Backtracking Search
    full_mask = (1 << n) - 1
    best_parts = []
    best_count = n + 1
    start_time = time.time()
    deadline = start_time + 50.0 # 50s limit to be safe
    
    # State: assigned_mask, list of part_masks
    # To reduce symmetry, we can sort part_masks? 
    # Or just rely on the "first unassigned vertex" heuristic which fixes the order of vertex assignment.
    # The only symmetry is the order of parts in the list.
    # We can enforce that part_masks are sorted by their integer value.
    
    def dfs(assigned_mask, part_masks):
        nonlocal best_parts, best_count
        
        if time.time() > deadline:
            return False # Indicate timeout
            
        if assigned_mask == full_mask:
            if len(part_masks) < best_count:
                best_count = len(part_masks)
                best_parts = part_masks[:]
            return True
            
        if len(part_masks) >= best_count:
            return True
            
        # Find first unassigned vertex
        # ~assigned_mask & full_mask
        unassigned = full_mask ^ assigned_mask
        if unassigned == 0:
            return True
            
        lsb = unassigned & -unassigned
        v = lsb.bit_length() - 1
        bit_v = 1 << v
        new_assigned = assigned_mask | bit_v
        
        # Option 1: Add v to an existing part
        # To avoid symmetry, we can try adding to each unique "type" of part?
        # Simpler: try all, but if we have two parts that are identical in terms of 
        # their interaction with the remaining graph, we'd skip one. 
        # Given N=14, just trying all is fine.
        
        # To slightly reduce branching, we can skip adding to a part if we've already 
        # tried adding to a part with the same "signature" regarding the remaining vertices?
        # Let's just do it straightforwardly first.
        
        for i, pm in enumerate(part_masks):
            new_pm = pm | bit_v
            if valid_mask[new_pm]:
                # Create new list of part masks
                # We can modify in place for speed, but need to backtrack
                new_parts = part_masks[:]
                new_parts[i] = new_pm
                # Optional: sort new_parts to maintain canonical form?
                # Sorting might help with symmetry breaking if we detect duplicates.
                # But since vertices are unique, parts are unique sets.
                # However, if part A and part B are "symmetric" in the graph, 
                # adding v to A is equivalent to adding v to B.
                # Hard to detect without graph automorphism.
                
                dfs(new_assigned, new_parts)
                if time.time() > deadline:
                    return False
                if best_count <= len(part_masks): # Pruning: if we found a solution with same or fewer parts, we don't need to explore deeper branches that start with same part count?
                    # Actually, if best_count is updated, we can prune.
                    pass

        # Option 2: Create a new part
        # Symmetry breaking: Only create a new part if it's the "last" one?
        # No, we must try it.
        # However, if we have already created a part with the same "structure" relative to the rest, we might skip.
        # For now, just add.
        
        # Optimization: If we have a part that is a singleton, and we create another singleton, 
        # they are symmetric? Not necessarily if the vertices are different.
        
        new_parts = part_masks + [bit_v]
        dfs(new_assigned, new_parts)
        
        return True

    dfs(0, [])
    
    # Construct the result from best_parts
    # best_parts is a list of masks
    result = []
    for pm in best_parts:
        part_verts = []
        m = pm
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            part_verts.append(v)
            m -= lsb
        result.append(part_verts)
        
    # Ensure types are correct
    # "sequences must be list or tuple"
    # "integers must be Python int"
    
    return result
