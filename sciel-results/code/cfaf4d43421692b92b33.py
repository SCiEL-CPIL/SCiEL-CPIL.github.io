import numpy as np

N = 50

# EVOLVE-BLOCK-START
def construct():
    # The optimal ratio for a 2-element support is 1.25, achieved by weights [2/3, 1/3].
    # We embed this solution into the 50-element vector.
    h = [0.0] * N
    h[0] = 2.0 / 3.0
    h[1] = 1.0 / 3.0
    return h
# EVOLVE-BLOCK-END
