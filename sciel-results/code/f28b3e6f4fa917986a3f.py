def construct():
    n = 10
    
    # Mapping: int <-> (a, b) where a in 0..4, b in 0..1
    def to_group(x):
        return (x % 5, x // 5)
        
    def to_int(g):
        return g[0] + 5 * g[1]
        
    def add(g1, g2):
        return ((g1[0] + g2[0]) % 5, (g1[1] + g2[1]) % 2)
        
    def mul_scalar(g, m):
        # m is (u, v)
        return ((g[0] * m[0]) % 5, (g[1] * m[1]) % 2)
        
    def sub(g1, g2):
        return ((g1[0] - g2[0]) % 5, (g1[1] - g2[1]) % 2)
        
    # Multipliers m1, m2 such that m1 - m2 is an automorphism
    # m1 = (1, 1)
    # m2 = (2, 0)
    # m1 - m2 = (-1, 1) = (4, 1)
    # 4 is invertible in Z5, 1 is invertible in Z2. So it's an automorphism.
    
    m1 = (1, 1)
    m2 = (2, 0)
    
    squares = []
    
    for m in [m1, m2]:
        sq = []
        for i in range(n):
            row = []
            gi = to_group(i)
            for j in range(n):
                gj = to_group(j)
                # L(i, j) = i + m * j
                term = mul_scalar(gj, m)
                val_g = add(gi, term)
                val = to_int(val_g)
                row.append(val)
            sq.append(row)
        squares.append(sq)
        
    return squares
