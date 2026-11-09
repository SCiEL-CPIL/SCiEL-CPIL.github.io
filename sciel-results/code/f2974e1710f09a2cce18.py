"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs a 2-coloring of K_n such that neither color class contains K_{2,2,2}.
    Uses the Paley graph of order 29, which is known to be a valid solution.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Paley graph of order q = 29
    # q must be a prime power congruent to 1 mod 4
    q = 29
    
    # Calculate quadratic residues mod q
    squares = set()
    for x in range(1, q):
        squares.add((x * x) % q)
    
    n = q
    colors = []
    
    # Generate colors in lexicographic order for pairs (i, j) with i < j
    for i in range(n):
        for j in range(i + 1, n):
            # Check if difference is a quadratic residue
            diff = (j - i) % q
            if diff in squares:
                colors.append(0)
            else:
                colors.append(1)
                
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
