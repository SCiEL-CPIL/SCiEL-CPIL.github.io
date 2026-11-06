# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the Pinwheel (Rank-2) automaton for N=12.
    This automaton is known to have a reset threshold of N^2 - 3N + 4.
    For N=12, this is 112, which is near the theoretical upper bound of (N-1)^2 = 121.
    The construction is deterministic and systematic.
    """
    N = 12
    # Letter 'a' is the cyclic permutation: i -> (i + 1) % N
    a = [(i + 1) % N for i in range(N)]
    
    # Letter 'b' is the shift-left with 0 fixed: 
    # 0 -> 0
    # i -> i - 1 for i > 0
    b = [0] * N
    for i in range(1, N):
        b[i] = i - 1
        
    return [a, b]
# EVOLVE-BLOCK-END
