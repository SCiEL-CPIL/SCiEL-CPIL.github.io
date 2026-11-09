import time
import random

def construct():
    """
    Returns the Paley tournament on p=67 vertices, which satisfies the
    4-dominator property. This is the smallest known valid tournament.
    The orientation list has one bit per pair (i,j) with i<j in lexicographic
    order: 0 means i -> j, 1 means j -> i.
    """
    p = 67
    
    # Precompute quadratic residues modulo p
    # For prime p ≡ 3 (mod 4), x^2 ≡ -1 has no solution, so the Paley tournament
    # is a valid tournament (skew-symmetric adjacency).
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    
    # Build orientation list in lexicographic order of pairs (i, j), i < j
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                # i -> j
                orientation.append(0)
            else:
                # j -> i
                orientation.append(1)
    
    return {'n': p, 'orientation': orientation}
