"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Paley colouring of K29.
    # p=29 is a prime such that p = 1 (mod 4).
    # The Paley graph on Z_p is defined by edges (i, j) if (i-j) is a quadratic residue.
    # It is known that the Paley graph of order 29 is K_{2,2,2}-free in both
    # the graph and its complement, thus R(K_{2,2,2}) > 29.
    
    p = 29
    squares = {x * x % p for x in range(1, p)}
    
    # Generate colors in lexicographic order (i, j) with i < j
    colors = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            # Color 0 if difference is a square, 1 otherwise
            if diff in squares:
                colors.append(0)
            else:
                colors.append(1)
                
    return {'n': p, 'colors': colors}
# EVOLVE-BLOCK-END
