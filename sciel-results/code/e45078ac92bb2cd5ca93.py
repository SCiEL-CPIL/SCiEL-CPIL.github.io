"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Returns the Paley tournament on 67 vertices.
    This is the smallest known tournament where every 4 vertices have a common dominator.
    """
    p = 67
    # Compute quadratic residues modulo p
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    
    # Build the orientation list
    # One bit per pair (i, j) with i < j in lexicographic order
    # 0 means i -> j, 1 means j -> i
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            # Check if j - i is a square mod p
            # If (j - i) % p is in squares, then i -> j (0)
            # Else j -> i (1)
            if (j - i) % p in squares:
                orientation.append(0)
            else:
                orientation.append(1)
                
    return {'n': p, 'orientation': orientation}
# EVOLVE-BLOCK-END
