def construct():
    import itertools
    
    # Generators for the Mathieu group M12 acting on 12 points {0, ..., 11}
    # These are standard generators (1-indexed in GAP: (1..11) and (1,2,5,6,10,11,3,4,7,8,9,12))
    g1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 11]
    g2 = [1, 4, 5, 9, 10, 2, 3, 6, 7, 8, 11, 0]
    
    def compose(p, q):
        return [p[q[i]] for i in range(12)]
    
    # Generate the group M12
    group = {tuple(g1)}
    queue = [tuple(g1)]
    while queue:
        current = queue.pop(0)
        for gen in (g1, g2):
            prod = tuple(compose(current, gen))
            if prod not in group:
                group.add(prod)
                queue.append(prod)
            prod2 = tuple(compose(gen, current))
            if prod2 not in group:
                group.add(prod2)
                queue.append(prod2)
                
    # Base block
    base_block = frozenset([0, 1, 2, 3, 4, 5])
    
    # Generate orbit
    blocks = set()
    for perm in group:
        b = frozenset(perm[i] for i in base_block)
        blocks.add(b)
        
    result = [list(b) for b in blocks]
    return result
