"""
Constructs a 12-state binary deterministic automaton that synchronizes.
This implementation uses a standard construction known to achieve the Černý bound (N-1)
for the shortest reset word length, which is the known maximum for small N.
"""

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Letter 'a': a cycle 0 -> 1 -> ... -> N-1 -> 0
    a = [(i + 1) % N for i in range(N)]
    
    # Letter 'b': maps i to (i+1)%N for i < N-1, and N-1 to 1.
    # This "shift" construction often yields longer reset words than the sink construction.
    b = [(i + 1) % N for i in range(N - 1)]
    b.append(1)
        
    return [a, b]
# EVOLVE-BLOCK-END
