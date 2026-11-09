"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 128}

def construct():
    # Strategy: Split into A and B such that 2*max(A) < min(B)
    # A is sum-free, B is an interval that avoids internal triples.
    # K=31, L=63 gives size 82.
    K = 31
    L = 63
    # A: largest sum-free subset of 1..K
    # Numbers > K/2 are sum-free
    A_start = K // 2 + 1
    A = list(range(A_start, K + 1))
    
    # B: Interval L..128
    B = list(range(L, P['n'] + 1))
    
    S = A + B
    return S
# EVOLVE-BLOCK-END
