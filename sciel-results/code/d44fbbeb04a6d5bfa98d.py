"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from collections import deque

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Pinwheel-style construction for N=12
    # a: cycle 0->1->...->11->0
    # b: 0->0, 1->0, i->i-1 for i>=2
    a = [(i + 1) % N for i in range(N)]
    b = [0] * N
    for i in range(2, N):
        b[i] = i - 1
    # b[0]=0, b[1]=0, b[2]=1, b[3]=2, ..., b[11]=10
    return [a, b]
# EVOLVE-BLOCK-END
