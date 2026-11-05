"""
Optimized program for the Černý problem with N=12.
Uses the standard Černý automaton construction which achieves the theoretical
upper bound of (N-1)^2 for the synchronization threshold.
"""
N = 12

# EVOLVE-BLOCK-START
def construct():
    # Construct the Černý automaton.
    # Letter 0: Cyclic shift right (i -> i+1 mod N)
    # Letter 1: 0 stays 0, others shift left (i -> i-1 mod N for i>0)
    
    # Letter 0 mapping
    letter_a = [(i + 1) % N for i in range(N)]
    
    # Letter 1 mapping
    letter_b = [0] * N
    for i in range(1, N):
        letter_b[i] = i - 1
        
    return [letter_a, letter_b]
# EVOLVE-BLOCK-END
