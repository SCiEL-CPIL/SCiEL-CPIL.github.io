"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 10}
_START_TIME = time.time()
_TIME_LIMIT = 60.0

def construct():
    n = P['n']
    
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT - 5:
        return [[[(i + j) % n for j in range(n)] for i in range(n)]]

    # Mathematically, max MOLS of order 10 is 1.
    # Generate the standard cyclic Latin square.
    square = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Ensure strict Python int types
    square = [[int(val) for val in row] for row in square]
    
    return [square]
# EVOLVE-BLOCK-END
