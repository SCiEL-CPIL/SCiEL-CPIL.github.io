"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Returns a 2-coloring of K_29 where neither color class contains K_{2,2,2}.
    This uses the Paley graph of order 29, which is a known valid Ramsey coloring
    for R(K_{2,2,2}) = 30.
    """
    q = 29
    # Compute quadratic residues modulo q
    squares = {x * x % q for x in range(1, q)}
    
    n = q
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % q
            # Color 0 if difference is a quadratic residue, else 1
            if diff in squares:
                colors.append(0)
            else:
                colors.append(1)
                
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
