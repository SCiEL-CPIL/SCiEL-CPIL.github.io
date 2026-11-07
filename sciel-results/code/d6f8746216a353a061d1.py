import time
import sys

sys.setrecursionlimit(10000)

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency masks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # Precompute non-adjacency for faster checks
    full_mask = (1 << n) - 1
    non_adj_mask = [0] * n
    for i in range(n):
        non_adj_mask[i] = full_mask ^ (1 << i) ^ adj_mask[i]
    
    # Order vertices: use a heuristic that considers both clique and independent set constraints
    # A vertex with extreme degree (very high or very low) is more constrained
    degrees = [bin(adj_mask[i]).count('1') for i in range(n)]
    # Sort by how "constrained" the vertex is: high degree makes it hard for independent set,
    # low degree makes it hard for clique. Use max(deg, n-1-deg) as constraint measure.
    constraint_score = [max(degrees[i], n - 1 - degrees[i]) for i in range(n)]
    vertices = list(range(n))
    vertices.sort(key=lambda x: constraint_score[x], reverse=True)
    
    start_time = time.time()
    deadline = start_time + 85.0
    
    for k in range(1, n + 1):
        if time.time() > deadline:
            break
            
        masks = [0] * k
        is_c = [True] * k
        is_i = [True] * k
        
        def can_add_to_part(v, i):
            """Check if vertex v can be added to part i, return (new_is_c, new_is_i) or (False, False)"""
            new_is_c = is_c[i]
            new_is_i = is_i[i]
            
            if new_is_c:
                if (adj_mask[v] & masks[i]) != masks[i]:
                    new_is_c = False
            
            if new_is_i:
                if (adj_mask[v] & masks[i]) != 0:
                    new_is_i = False
            
            if new_is_c or new_is_i:
                return (new_is_c, new_is_i)
            return (False, False)
        
        def forward_check(idx):
            """Check if all remaining vertices can be added to some part"""
            for j in range(idx, n):
                v = vertices[j]
                v_bit = 1 << v
                found = False
                for i in range(k):
                    if masks[i] == 0:
                        if i > 0 and masks[i-1] == 0:
                            continue
                        # Empty part: always valid (single vertex is both clique and independent)
                        found = True
                        break
                    result = can_add_to_part(v, i)
                    if result[0] or result[1]:
                        found = True
                        break
                if not found:
                    return False
            return True
        
        def dfs(idx):
            if time.time() > deadline:
                return None
            
            if idx == n:
                res = []
                for i in range(k):
                    if masks[i] != 0:
                        part_verts = []
                        for v in range(n):
                            if masks[i] & (1 << v):
                                part_verts.append(v)
                        res.append(part_verts)
                return res
            
            # Forward check
            if not forward_check(idx):
                return None
            
            v = vertices[idx]
            v_bit = 1 << v
            
            # Try adding v to existing parts
            # Order parts: try more constrained parts first
            part_order = list(range(k))
            
            def part_sort_key(i):
                if masks[i] == 0:
                    return 0
                # More constrained parts (only one type possible) first
                if is_c[i] and is_i[i]:
                    return 2
                elif is_c[i] or is_i[i]:
                    return 1
                else:
                    return 0
            
            part_order.sort(key=part_sort_key)
            
            for i in part_order:
                if masks[i] == 0:
                    if i > 0 and masks[i-1] == 0:
                        continue
                
                result = can_add_to_part(v, i)
                if result[0] or result[1]:
                    new_is_c, new_is_i = result
                    
                    old_mask = masks[i]
                    old_c = is_c[i]
                    old_i = is_i[i]
                    
                    masks[i] |= v_bit
                    is_c[i] = new_is_c
                    is_i[i] = new_is_i
                    
                    res = dfs(idx + 1)
                    if res is not None:
                        return res
                    
                    masks[i] = old_mask
                    is_c[i] = old_c
                    is_i[i] = old_i
            
            return None
        
        res = dfs(0)
        if res is not None:
            final_parts = []
            for part in res:
                final_parts.append([int(v) for v in part])
            return final_parts

    return [[i] for i in range(n)]
