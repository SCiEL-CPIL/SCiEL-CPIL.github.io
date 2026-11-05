import numpy as np
import time

def _paley_orientation_67():
    p = 67
    # Compute quadratic residues mod p
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return orientation

def construct():
    # The smallest known tournament satisfying the condition for k=4 has 67 vertices.
    # It is a Paley tournament on the prime p=67.
    # We return this directly as it is the optimal solution.
    n = 67
    orientation = _paley_orientation_67()
    return {'n': n, 'orientation': orientation}
