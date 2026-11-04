def _cayley_girth_ok(group_elements, generators, g):
    # group_elements: list of elements
    # generators: list of k elements
    # Build adjacency? No, just BFS.
    # Identity is group_elements[0] (assume index 0)
    
    # Map element to index
    elem_to_idx = {e: i for i, e in enumerate(group_elements)}
    
    dist = [-1] * len(group_elements)
    dist[0] = 0
    queue = [0]
    head = 0
    
    while head < len(queue):
        u_idx = queue[head]
        head += 1
        u = group_elements[u_idx]
        
        if dist[u_idx] >= g:
            continue
            
        for s in generators:
            v = _mul_psl2(u, s, q) # Need q here
            v_idx = elem_to_idx[v]
            
            if dist[v_idx] == -1:
                dist[v_idx] = dist[u_idx] + 1
                queue.append(v_idx)
            else:
                # Check cycle
                cycle_len = dist[u_idx] + dist[v_idx] + 1
                if cycle_len < g:
                    return False
    return True
