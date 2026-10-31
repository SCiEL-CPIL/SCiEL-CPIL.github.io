import time

# EVOLVE-BLOCK-START
def priority(vector, n):
    """
    Prioritize vectors with many 1s (center of hypercube) 
    and minimize the interaction between 0s and 2s.
    """
    c1 = 0
    c0 = 0
    c2 = 0
    for x in vector:
        if x == 1:
            c1 += 1
        elif x == 0:
            c0 += 1
        else:
            c2 += 1
            
    # Maximize c1
    # Minimize c0 * c2 (penalize mixing 0s and 2s)
    return c1 * 1000.0 - (c0 * c2)
# EVOLVE-BLOCK-END
