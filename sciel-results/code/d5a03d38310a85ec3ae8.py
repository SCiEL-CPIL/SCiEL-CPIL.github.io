import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Split m=4 into 2 blocks of size 2
    # Split p=8 into 4 blocks of size 2
    # n=2 is not split
    m_blocks = 2
    p_blocks = 4
    block_size_m = 2
    block_size_p = 2
    block_size_n = 2
    
    total_blocks = m_blocks * p_blocks
    strassen_rank = 7
    rank = total_blocks * strassen_rank
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen definitions for 2x2x2
    # Indices: i, j, k in {0, 1}
    # M1: (a00+a11)(b00+b11) -> c00, c11
    # M2: (a10+a11)b00 -> c10, c11
    # M3: a00(b01-b11) -> c01, c11
    # M4: a11(b10-b00) -> c00, c10
    # M5: (a00+a01)b11 -> c00, c01
    # M6: (a10-a00)(b00+b01) -> c11
    # M7: (a01-a11)(b10+b11) -> c00, c11
    
    # Define the 7 terms as (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict {(i,j): coeff}
    # V_coeffs: dict {(j,k): coeff}
    # W_coeffs: dict {(k,i): coeff}
    
    strassen_terms = [
        # M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}), # c10=1, c11=-1? 
        # Check Strassen:
        # c00 = M1 + M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M6
        
        # Let's re-verify coefficients for W in each term.
        # Term c contributes W_ki = coeff.
        
        # M1: c00 (+1), c11 (+1)
        # M2: c10 (+1), c11 (-1)
        # M3: c01 (+1), c11 (+1)
        # M4: c00 (+1), c10 (+1)
        # M5: c00 (-1), c01 (+1)
        # M6: c11 (+1)
        # M7: c00 (+1), c11 (+1)
    ]
    
    # Corrected Strassen W coefficients:
    # M1: W[(0,0)]=1, W[(1,1)]=1
    # M2: W[(1,0)]=1, W[(1,1)]=-1
    # M3: W[(0,1)]=1, W[(1,1)]=1
    # M4: W[(0,0)]=1, W[(1,0)]=1
    # M5: W[(0,0)]=-1, W[(0,1)]=1
    # M6: W[(1,1)]=1
    # M7: W[(0,0)]=1, W[(1,1)]=1
    
    strassen_terms = [
        # M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
    ]
    
    col_idx = 0
    for bj in range(m_blocks):
        for bk in range(p_blocks):
            for u_local, v_local, w_local in strassen_terms:
                # Map local indices to global
                for (li, lj), coeff_u in u_local.items():
                    gi = li
                    gj = bj * block_size_m + lj
                    U[gi * m + gj][col_idx] = coeff_u
                
                for (lj, lk), coeff_v in v_local.items():
                    gj = bj * block_size_m + lj
                    gk = bk * block_size_p + lk
                    V[gj * p + gk][col_idx] = coeff_v
                
                for (lk, li), coeff_w in w_local.items():
                    gk = bk * block_size_p + lk
                    gi = li
                    W[gk * n + gi][col_idx] = coeff_w
                
                col_idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
