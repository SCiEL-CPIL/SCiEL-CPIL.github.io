import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a tournament on n vertices where every set of 4 vertices
    has a common dominator.
    Uses the Paley tournament construction for p=67, which is the
    smallest known solution to Schütte's problem for k=4.
    """
    # p must be a prime of the form 4k+3 for the Paley tournament
    # to be regular and have good structural properties.
    p = 67
    
    # Compute quadratic residues mod p
    squares = np.zeros(p, dtype=bool)
    for x in range(1, p):
        squares[(x * x) % p] = True
    
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            # If diff is a quadratic residue, i -> j (orientation 0)
            # Otherwise j -> i (orientation 1)
            if squares[diff]:
                orientation.append(0)
            else:
                orientation.append(1)
    
    return {'n': p, 'orientation': orientation}
# EVOLVE-BLOCK-END
