import time
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a tournament on n vertices where every set of 4 vertices
    has a common dominator.
    
    Strategy:
    1. Return the known valid Paley tournament on n=67.
    This is the smallest known solution to Schütte's problem for k=4.
    """
    
    # Helper: Generate Paley tournament orientation list
    def get_paley_orientation(p):
        squares = np.zeros(p, dtype=bool)
        for x in range(1, p):
            squares[(x * x) % p] = True
        
        orientation = []
        for i in range(p):
            for j in range(i + 1, p):
                diff = (j - i) % p
                if squares[diff]:
                    orientation.append(0)
                else:
                    orientation.append(1)
        return orientation

    # Return Paley tournament for n=67
    # This is the known smallest valid solution.
    p = 67
    orientation = get_paley_orientation(p)
    return {'n': p, 'orientation': orientation}
# EVOLVE-BLOCK-END
