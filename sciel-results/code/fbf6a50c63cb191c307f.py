"""
Constructs a 12-state binary deterministic automaton that achieves the Černý bound
(N-1)^2 for the shortest reset word length.
"""

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Letter 'a': A cycle 0 -> 1 -> ... -> N-1 -> 0
    a = [(i + 1) % N for i in range(N)]
    
    # Letter 'b': Fixes states 0 to N-2, maps N-1 to 0
    # This specific "Černý automaton" structure is known to maximize the reset word length
    # to exactly (N-1)^2.
    b = list(range(N))
    b[N-1] = 0
    
    return [a, b]
# EVOLVE-BLOCK-END
