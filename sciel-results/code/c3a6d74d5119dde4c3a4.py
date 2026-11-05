def construct():
    # GF(4) operations: represent elements as 0,1,2,3
    # GF(4) addition is XOR (since it's characteristic 2)
    # GF(4) multiplication table (using alpha^0=0, alpha^1=1, alpha^2=2, alpha^3=3)
    # Actually let's use standard GF(4): {0, 1, alpha, alpha+1}
    # Let's label: 0=0, 1=1, 2=alpha, 3=alpha+1
    # alpha^2 = alpha + 1 = 3
    # Multiplication table:
    #   * | 0 1 2 3
    #   0 | 0 0 0 0
    #   1 | 0 1 2 3
    #   2 | 0 2 3 1
    #   3 | 0 3 1 2
    
    gf4_mult = [
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2]
    ]
    
    # Vertex (r, c) maps to index r*4 + c
    # Edge (u, v) where u = r1*4+c1, v = r2*4+c2
    # Color assignment:
    #   Horizontal (r1==r2): color = c1 ^ c2, range [0,3]
    #   Vertical (c1==c2): color = 4 + r1 ^ r2, range [4,7]
    #   Diagonal: color = 8 + (c1^c2) ^ (r1^r2), range [8,14]
    
    colors = []
    for i in range(16):
        for j in range(i+1, 16):
            r1, c1 = divmod(i, 4)
            r2, c2 = divmod(j, 4)
            
            if r1 == r2:
                # Horizontal edge
                col = c1 ^ c2
            elif c1 == c2:
                # Vertical edge
                col = 4 + (r1 ^ r2)
            else:
                # Diagonal edge
                col = 8 + ((c1 ^ c2) ^ (r1 ^ r2))
            
            colors.append(col)
    
    return colors
