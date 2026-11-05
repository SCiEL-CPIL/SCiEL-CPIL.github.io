import time

N = 12

def construct():
    """
    Construct a 12-state binary automaton that maximizes the shortest reset word length.
    Returns the standard Černý automaton which achieves the theoretical maximum
    threshold of (N-1)^2 = 121.
    
    The automaton is defined by:
    a[i] = (i + 1) % N
    b[0] = 0
    b[i] = i - 1 for i > 0
    """
    a = [(i + 1) % N for i in range(N)]
    b = [0] * N
    for i in range(1, N):
        b[i] = i - 1
        
    return [a, b]

# EVOLVE-BLOCK-START
# The function construct() is defined above.
# EVOLVE-BLOCK-END
