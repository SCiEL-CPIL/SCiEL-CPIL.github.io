"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12
import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the Cerny automaton for N states.
    This automaton is known to have the maximum possible reset word length of (N-1)^2.
    For N=12, the length is 121.
    
    Letter 'a' is a cyclic shift: a(i) = (i + 1) % N
    Letter 'b' is a reset-like map: b(0) = 0, b(i) = i - 1 for i > 0
    """
    a_map = [(i + 1) % N for i in range(N)]
    b_map = [0 if i == 0 else i - 1 for i in range(N)]
    
    return [a_map, b_map]
# EVOLVE-BLOCK-END
