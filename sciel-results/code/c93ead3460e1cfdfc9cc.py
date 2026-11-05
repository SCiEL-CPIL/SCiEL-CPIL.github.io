N = 12

# EVOLVE-BLOCK-START
def construct():
    # Letter 'a': a cycle 0 -> 1 -> ... -> N-1 -> 0
    a = [(i + 1) % N for i in range(N)]
    
    # Letter 'b': maps 0 to 0, and i to i-1 for i > 0
    # This is the standard Černý automaton which achieves the bound N-1
    b = [0] * N
    for i in range(1, N):
        b[i] = i - 1
        
    return [a, b]
# EVOLVE-BLOCK-END
