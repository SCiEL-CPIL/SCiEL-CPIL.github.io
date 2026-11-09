def construct():
    """
    Constructs the 12-state Černý automaton.
    This automaton achieves the theoretical maximum reset word length
    predicted by Černý's conjecture for N=12, which is (N-1)^2 = 121.
    
    Structure:
    Letter 'a' (index 0): Cyclic shift i -> (i+1) % N
    Letter 'b' (index 1):
        - Maps 0 to 0
        - Maps 1 to 0
        - Maps i to (i+1) % N for i >= 2
    """
    trans_a = [(i + 1) % N for i in range(N)]
    
    trans_b = [0] * N
    trans_b[0] = 0
    trans_b[1] = 0
    for i in range(2, N):
        trans_b[i] = (i + 1) % N
        
    return [trans_a, trans_b]
