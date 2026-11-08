"""Program for rank-minimal tensor decomposition of 2x3x10 matrix multiplication."""
import numpy as np

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices
    for the tensor decomposition of 2x3x10 matrix multiplication.
    
    Uses a block decomposition with 5 blocks of 2x3x2, each of rank 11.
    Total rank: 55.
    """
    n, m, p = 2, 3, 10
    rank = 55
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, col):
        for i, val in u_idx:
            U[i, col] += val
        for i, val in v_idx:
            V[i, col] += val
        for i, val in w_idx:
            W[i, col] += val

    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        base = 11 * t
        
        # Term 0
        add_term([(0,1), (4,1)], [(0*10+k0,1), (1*10+k1,1)], [(k0*2+0,1), (k1*2+1,1)], base)
        # Term 1
        add_term([(3,1), (4,1)], [(0*10+k0,1)], [(k0*2+1,1), (k1*2+1,-1)], base+1)
        # Term 2
        add_term([(0,1)], [(0*10+k1,1), (1*10+k1,-1)], [(k1*2+0,1), (k1*2+1,1)], base+2)
        # Term 3
        add_term([(4,1)], [(1*10+k0,1), (0*10+k0,-1)], [(k0*2+0,1), (k0*2+1,1)], base+3)
        # Term 4
        add_term([(0,1), (1,1)], [(1*10+k1,1)], [(k0*2+0,-1), (k1*2+0,1)], base+4)
        # Term 5
        add_term([(3,1), (0,-1)], [(0*10+k0,1), (0*10+k1,1)], [(k1*2+1,1)], base+5)
        # Term 6
        add_term([(1,1), (4,-1)], [(1*10+k0,1), (1*10+k1,1)], [(k0*2+0,1)], base+6)
        # Term 7
        add_term([(2,1)], [(2*10+k0,1)], [(k0*2+0,1)], base+7)
        # Term 8
        add_term([(2,1)], [(2*10+k1,1)], [(k1*2+0,1)], base+8)
        # Term 9
        add_term([(5,1)], [(2*10+k0,1)], [(k0*2+1,1)], base+9)
        # Term 10
        add_term([(5,1)], [(2*10+k1,1)], [(k1*2+1,1)], base+10)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
