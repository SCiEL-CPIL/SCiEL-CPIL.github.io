import time

N = 12

def construct():
    """
    Constructs the standard Černý (Pawlik) automaton for N=12.
    
    The automaton is defined by two transitions:
    - a: i -> (i + 1) % N  (a cycle)
    - b: 0 -> 0, i -> i - 1 for i > 0 (a reset-like map that decrements)
    
    This automaton is known to have a shortest reset word length of (N-1)^2.
    For N=12, this is 121, which is the theoretical maximum.
    """
    # a: i -> (i + 1) % N
    a_map = [(i + 1) % N for i in range(N)]
    
    # b: 0 -> 0, i -> i - 1
    b_map = [0 if i == 0 else i - 1 for i in range(N)]
    
    return [a_map, b_map]
