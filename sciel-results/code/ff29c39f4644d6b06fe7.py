def construct():
    # Construct projective plane of order 4
    # GF(4) elements: 0, 1, 2, 3
    # 2^2 = 2+1=3, 2*3=6=2, 3^2=9=1? No.
    # Let alpha = 2. alpha^2 = alpha + 1 = 3.
    # alpha^3 = alpha*3 = 6 = 2. Wait.
    # In GF(4), x^2+x+1=0. So 2^2+2+1=0 => 4+2+1=7=0 mod ? No.
    # Let's use 0, 1, 2, 3.
    # 2^2 = 3. 2*3 = 6 = 2. 3^2 = 9 = 1.
    # So 2 is a primitive element.
    
    def add(a, b):
        return a ^ b
        
    def mul(a, b):
        if a == 0 or b == 0:
            return 0
        if a == 1:
            return b
        if b == 1:
            return a
        if a == 2 and b == 2:
            return 3
        if a == 3 and b == 3:
            return 1
        if (a == 2 and b == 3) or (a == 3 and b == 2):
            return 2
        return 0 # Should not happen

    # Points
    # 0-15: (x, y)
    # 16-19: (1, m) for m in 0..3
    # 20: (1, 0, 1)
    
    # Lines
    # 21-36: y = mx + b
    # 37-40: x = c
    # 41: line at infinity
    
    edges = []
    
    # Incidence for y = mx + b
    for m in range(4):
        for b in range(4):
            line_id = 21 + m*4 + b
            for x in range(4):
                y = add(mul(m, x), b)
                point_id = x*4 + y
                edges.append([point_id, line_id])
                
    # Incidence for x = c
    for c in range(4):
        line_id = 37 + c
        for y in range(4):
            point_id = c*4 + y
            edges.append([point_id, line_id])
            
    # Incidence for line at infinity (41)
    # Connects to points 16-20
    for m in range(4):
        point_id = 16 + m
        edges.append([point_id, 41])
    edges.append([20, 41])
    
    # Now add 5 new vertices 42-46
    # Choose distinct pairs (p, l)
    # To be safe, pick (0, 21), (1, 22), (2, 23), (3, 24), (4, 25)
    # Just ensure pairs are distinct.
    new_vertices = [42, 43, 44, 45, 46]
    pairs = [(0, 21), (1, 22), (2, 23), (3, 24), (4, 25)]
    
    for i, v in enumerate(new_vertices):
        p, l = pairs[i]
        edges.append([v, p])
        edges.append([v, l])
        
    n = 47
    return {'n': n, 'edges': edges}
