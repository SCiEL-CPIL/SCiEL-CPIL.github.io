"""
Constructs the Pawlik (Cerny) automaton for N=12.
This automaton is known to achieve the theoretical maximum reset threshold of (N-1)^2.
For N=12, the threshold is 121.
"""

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Letter 0: Cyclic permutation i -> (i + 1) % N
    sigma0 = [(i + 1) % N for i in range(N)]
    
    # Letter 1: 
    # 0 -> 0
    # i -> i - 1 for i > 0
    sigma1 = [0] * N
    for i in range(1, N):
        sigma1[i] = i - 1
        
    return [sigma0, sigma1]
# EVOLVE-BLOCK-END
