import numpy as np

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # p=10, so 5 blocks of 2 columns
    num_blocks = p // 2
    rank_per_block = 7
    total_rank = num_blocks * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # Each entry is a list of (u_idx, v_idx, w_idx, coeff_u, coeff_v, coeff_w)
    # But easier to just define the non-zero entries for U, V, W columns.
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    # M2: (A10+A11)(B00)     -> C10
    # M3: (A00)(B01-B11)      -> C01
    # M4: (A11)(B10-B00)      -> C10
    # M5: (A00+A01)(B11)      -> -C00, C01
    # M6: (A10-A00)(B00+B01)  -> C11
    # M7: (A01-A11)(B10+B11)  -> C00
    
    # Indices:
    # U: i*m + j
    # V: j*p + k (k is global col)
    # W: k*n + i
    
    for b in range(num_blocks):
        base_rank = b * rank_per_block
        base_k = b * 2 # global column offset
        
        # M1
        r = base_rank
        U[0*2+0][r] = 1; U[1*2+1][r] = 1
        V[0*2+base_k][r] = 1; V[1*2+base_k+1][r] = 1
        W[0*2+0][r] = 1; W[1*2+1][r] = 1
        
        # M2
        r = base_rank + 1
        U[1*2+0][r] = 1; U[1*2+1][r] = 1
        V[0*2+base_k][r] = 1
        W[0*2+1][r] = 1
        
        # M3
        r = base_rank + 2
        U[0*2+0][r] = 1
        V[0*2+base_k+1][r] = 1; V[1*2+base_k+1][r] = -1
        W[1*2+0][r] = 1
        
        # M4
        r = base_rank + 3
        U[1*2+1][r] = 1
        V[1*2+base_k][r] = 1; V[0*2+base_k][r] = -1
        W[0*2+1][r] = 1
        
        # M5
        r = base_rank + 4
        U[0*2+0][r] = 1; U[0*2+1][r] = 1
        V[1*2+base_k+1][r] = 1
        W[0*2+0][r] = -1; W[1*2+0][r] = 1
        
        # M6
        r = base_rank + 5
        U[1*2+0][r] = 1; U[0*2+0][r] = -1
        V[0*2+base_k][r] = 1; V[0*2+base_k+1][r] = 1
        W[1*2+1][r] = 1
        
        # M7
        r = base_rank + 6
        U[0*2+1][r] = 1; U[1*2+1][r] = -1
        V[1*2+base_k][r] = 1; V[1*2+base_k+1][r] = 1
        W[0*2+0][r] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
