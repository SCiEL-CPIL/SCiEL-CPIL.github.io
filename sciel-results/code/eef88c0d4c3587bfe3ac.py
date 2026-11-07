import time
import sys

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Precompute non-adjacency bitmasks (excluding self)
    # coadj[u] has bits set for v != u such that (u,v) is NOT an edge
    all_mask = (1 << n) - 1
    coadj = [0] * n
    for u in range(n):
        # neighbors are in adj[u]
        # non-neighbors are in all_mask ^ adj[u] ^ (1 << u)
        coadj[u] = all_mask ^ adj[u] ^ (1 << u)

    # Types: 0 = CLIQUE, 1 = INDEPENDENT, 2 = BOTH (size 1)
    CL = 0
    IND = 1
    BOTH = 2
    
    def can_add(v, part_mask, part_type):
        # part_mask does not contain v
        if part_type == CL:
            # v must be adjacent to all in part_mask
            return (part_mask & ~adj[v]) == 0
        elif part_type == IND:
            # v must be non-adjacent to all in part_mask
            return (part_mask & adj[v]) == 0
        elif part_type == BOTH:
            # v can be added, type will be determined
            return True
        return False

    def get_new_type(v, part_mask, part_type):
        if part_type == BOTH:
            # Check if v is adjacent to any vertex in part_mask
            # Since part_mask has 1 vertex u, check adj[u]
            # Actually, just check if (part_mask & adj[v]) != 0
            if (part_mask & adj[v]) != 0:
                return CL
            else:
                return IND
        return part_type

    # State for DFS
    # parts: list of (mask, type)
    # unassigned: bitmask
    
    # Heuristic: order vertices by degree descending?
    # Let's just iterate 0..n-1 but pick the one with fewest options?
    # For simplicity, we pick the lowest set bit in unassigned.
    
    def dfs(unassigned, parts, k_rem):
        if unassigned == 0:
            return True
        
        # Lower bound: if number of unassigned vertices > k_rem * max_possible_part_size
        # Max part size is n. This is weak.
        # Stronger: if unassigned vertices cannot be covered by k_rem parts.
        # For now, just proceed.
        
        # Pick a vertex
        # To improve pruning, pick a vertex that has the fewest options to add to existing parts?
        # Or just the first one.
        v = (unassigned & -unassigned).bit_length() - 1
        new_unassigned = unassigned ^ (1 << v)
        
        # Try adding v to existing parts
        for i in range(len(parts)):
            p_mask, p_type = parts[i]
            if can_add(v, p_mask, p_type):
                new_type = get_new_type(v, p_mask, p_type)
                new_mask = p_mask | (1 << v)
                
                # Create new parts list
                new_parts = parts[:]
                new_parts[i] = (new_mask, new_type)
                
                if dfs(new_unassigned, new_parts, k_rem):
                    return True
        
        # Try creating a new part
        if k_rem > 0:
            new_parts = parts[:] + [(1 << v, BOTH)]
            if dfs(new_unassigned, new_parts, k_rem - 1):
                return True
        
        return False

    # Search for minimum k
    for k in range(1, n + 1):
        # Initial state: no parts, all vertices unassigned
        if dfs((1 << n) - 1, [], k):
            # We found a solution with k parts.
            # Now we need to reconstruct the partition.
            # The dfs above only returns True/False. We need to store the solution.
            # Let's modify dfs to return the parts list or None.
            pass

    # Re-implement dfs to return solution
    def dfs_sol(unassigned, parts, k_rem):
        if unassigned == 0:
            return parts[:]
        
        v = (unassigned & -unassigned).bit_length() - 1
        new_unassigned = unassigned ^ (1 << v)
        
        # Try adding v to existing parts
        for i in range(len(parts)):
            p_mask, p_type = parts[i]
            if can_add(v, p_mask, p_type):
                new_type = get_new_type(v, p_mask, p_type)
                new_mask = p_mask | (1 << v)
                
                new_parts = parts[:]
                new_parts[i] = (new_mask, new_type)
                
                res = dfs_sol(new_unassigned, new_parts, k_rem)
                if res is not None:
                    return res
        
        # Try creating a new part
        if k_rem > 0:
            new_parts = parts[:] + [(1 << v, BOTH)]
            res = dfs_sol(new_unassigned, new_parts, k_rem - 1)
            if res is not None:
                return res
        
        return None

    for k in range(1, n + 1):
        res = dfs_sol((1 << n) - 1, [], k)
        if res is not None:
            # Convert parts to list of lists of vertices
            final_parts = []
            for mask, _ in res:
                verts = []
                for v in range(n):
                    if mask & (1 << v):
                        verts.append(v)
                final_parts.append(verts)
            
            # Ensure types are Python int and lists are lists
            return final_parts
            
    # Fallback (should not happen)
    return [[v] for v in range(n)]
