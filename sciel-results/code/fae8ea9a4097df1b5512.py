"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    n = 10
    squares = []
    
    # The 4 MOLS of order 5 are defined by slopes k in {1, 2, 3, 4}
    # A_k(r, c) = (r + k * c) % 5
    
    for k in [1, 2, 3, 4]:
        sq = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                # Decompose i and j into (mod 2, mod 5) parts
                # i = 2*r + u is not the right decomposition for independence.
                # We need i such that i%2=u and i%5=r.
                # We can just use i%2 and i%5 directly.
                
                u = i % 2
                r = i % 5
                v = j % 2
                c = j % 5
                
                # Calculate the order 5 symbol
                a_val = (r + k * c) % 5
                
                # Calculate the high bit (0 or 5)
                # H = 5 * (u + v) % 2
                h_val = 5 * ((u + v) % 2)
                
                sq[i][j] = h_val + a_val
        
        squares.append(sq)
    
    return squares

# EVOLVE-BLOCK-END
