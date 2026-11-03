"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    # Use Strassen on 2x2 blocks of 3x3 submatrices
    # 7 multiplications, each of 3x3 matrices (schoolbook rank 27)
    # Total rank = 7 * 27 = 188
    
    block_size = 3  # each block is 3x3
    rank = 7 * block_size * block_size  # 188
    
    U = [[0] * rank for _ in range(n * m)]  # (36, 188)
    V = [[0] * rank for _ in range(m * p)]  # (36, 188)
    W = [[0] * rank for _ in range(p * n)]  # (36, 188)
    
    def u_idx(row, col):
        return row * m + col
    
    def v_idx(row, col):
        return row * p + col
    
    def w_idx(row, col):
        # W[k*n+i, r] where C[i,k] is the output
        return col * n + row
    
    strassen_mults = [
        {'A': [(0,0,1), (1,1,1)], 'B': [(0,0,1), (1,1,1)]},       # M1
        {'A': [(1,0,1), (1,1,1)], 'B': [(0,0,1)]},                 # M2
        {'A': [(0,0,1)], 'B': [(0,1,1), (1,1,-1)]},                # M3
        {'A': [(1,1,1)], 'B': [(1,0,1), (0,0,-1)]},                # M4
        {'A': [(0,0,1), (0,1,1)], 'B': [(1,1,1)]},                 # M5
        {'A': [(1,0,1), (0,0,-1)], 'B': [(0,0,1), (0,1,1)]},       # M6
        {'A': [(0,1,1), (1,1,-1)], 'B': [(1,0,1), (1,1,1)]},       # M7
    ]
    
    c_combos = {
        (0,0): [(0,1), (3,1), (4,-1), (6,1)],
        (0,1): [(2,1), (4,1)],
        (1,0): [(1,1), (3,1)],
        (1,1): [(0,1), (1,-1), (2,1), (5,1)],
    }
    
    col = 0
    for mult_idx, mult in enumerate(strassen_mults):
        A_coefs = mult['A']
        B_coefs = mult['B']
        
        for a_local in range(block_size):
            for b_local in range(block_size):
                for c_local in range(block_size):
                    for (abr, abc, acoeff) in A_coefs:
                        row = abr * block_size + a_local
                        ccol = abc * block_size + b_local
                        U[u_idx(row, ccol)][col] += acoeff
                    
                    for (bbr, bbc, bcoeff) in B_coefs:
                        row = bbr * block_size + b_local
                        ccol = bbc * block_size + c_local
                        V[v_idx(row, ccol)][col] += bcoeff
                    
                    for (cbi, cbj), combos in c_combos.items():
                        for (midx, mcoeff) in combos:
                            if midx == mult_idx:
                                row = cbi * block_size + a_local
                                ccol = cbj * block_size + c_local
                                W[w_idx(row, ccol)][col] += mcoeff
                    
                    col += 1
    
    assert col == rank, f"col={col}, rank={rank}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
