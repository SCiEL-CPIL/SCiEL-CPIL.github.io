def construct():
    from collections import deque
    
    # Generators for the Mathieu group M12 acting on 12 points (0-indexed)
    # g1 is a 12-cycle
    g1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]
    # g2 corresponds to the cycle (0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11)
    g2 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 0]
    
    identity = list(range(12))
    visited = {tuple(identity)}
    queue = deque([identity])
    group_elements = []
    
    def compose(p, q):
        # Returns permutation p composed with q: p(q(x))
        return [p[q[i]] for i in range(12)]
        
    def apply_perm(p, block):
        return frozenset(p[x] for x in block)
        
    while queue:
        p = queue.popleft()
        group_elements.append(p)
        
        p_g1 = compose(p, g1)
        if tuple(p_g1) not in visited:
            visited.add(tuple(p_g1))
            queue.append(p_g1)
            
        p_g2 = compose(p, g2)
        if tuple(p_g2) not in visited:
            visited.add(tuple(p_g2))
            queue.append(p_g2)
            
    base_block = frozenset([0, 1, 2, 3, 4, 5])
    
    blocks = set()
    for p in group_elements:
        b = apply_perm(p, base_block)
        blocks.add(b)
        
    return [list(b) for b in blocks]
