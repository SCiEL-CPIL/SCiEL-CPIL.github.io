def construct_luw(q, d):
    # q is prime power, d is dimension
    # Returns n, edges
    n_side = q ** d
    n = 2 * n_side
    
    # We need addition and multiplication in GF(q)
    # For prime q: mod q
    # For q=4: XOR for add, specific table for mul
    # For general prime power, we'd need a library or specific implementation.
    # Assuming k is prime or 4 for now.
    
    def add(a, b):
        if q == 4:
            return a ^ b
        else:
            return (a + b) % q
            
    def mul(a, b):
        if q == 4:
            # GF(4) multiplication
            # 0*0=0, 0*1=0, 0*2=0, 0*3=0
            # 1*0=0, 1*1=1, 1*2=2, 1*3=3
            # 2*0=0, 2*1=2, 2*2=3, 2*3=1
            # 3*0=0, 3*1=3, 3*2=1, 3*3=2
            table = [
                [0,0,0,0],
                [0,1,2,3],
                [0,2,3,1],
                [0,3,1,2]
            ]
            return table[a][b]
        else:
            return (a * b) % q

    edges = []
    edge_set = set()
    
    # Iterate over all x in side 0
    for x_idx in range(n_side):
        # Decode x_idx to vector x
        x = [0] * d
        temp = x_idx
        for i in range(d):
            x[i] = temp % q
            temp //= q
            
        # For each a in GF(q), compute neighbor y
        for a in range(q):
            y = [0] * d
            y[0] = a
            for i in range(1, d):
                y[i] = add(x[i], mul(x[i-1], y[i-1]))
                
            # Encode y to index
            y_idx = 0
            mult = 1
            for i in range(d):
                y_idx += y[i] * mult
                mult *= q
            
            u = x_idx
            v = n_side + y_idx
            e = (min(u,v), max(u,v))
            if e not in edge_set:
                edge_set.add(e)
                edges.append(e)
                
    return n, edges
