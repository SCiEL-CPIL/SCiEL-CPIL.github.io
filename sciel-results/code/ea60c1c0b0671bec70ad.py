"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    block_idx = 0
    while col_offset < p:
        # m1
        U[0][block_idx*7 + 0] = 1
        U[3][block_idx*7 + 0] = 1
        V[0*12+col_offset][block_idx*7 + 0] = 1
        V[1*12+col_offset+1][block_idx*7 + 0] = 1
        W[col_offset*2+0][block_idx*7 + 0] += 1
        W[(col_offset+1)*2+1][block_idx*7 + 0] += 1
        
        # m2
        U[2][block_idx*7 + 1] = 1
        U[3][block_idx*7 + 1] = 1
        V[0*12+col_offset][block_idx*7 + 1] = 1
        W[col_offset*2+1][block_idx*7 + 1] += 1
        W[(col_offset+1)*2+1][block_idx*7 + 1] -= 1
        
        # m3
        U[0][block_idx*7 + 2] = 1
        V[0*12+col_offset+1][block_idx*7 + 2] = 1
        V[1*12+col_offset+1][block_idx*7 + 2] = -1
        W[(col_offset+1)*2+0][block_idx*7 + 2] += 1
        W[(col_offset+1)*2+1][block_idx*7 + 2] += 1
        
        # m4
        U[3][block_idx*7 + 3] = 1
        V[1*12+col_offset][block_idx*7 + 3] = 1
        V[0*12+col_offset][block_idx*7 + 3] = -1
        W[col_offset*2+0][block_idx*7 + 3] += 1
        W[col_offset*2+1][block_idx*7 + 3] += 1
        
        # m5
        U[0][block_idx*7 + 4] = 1
        U[1][block_idx*7 + 4] = 1
        V[1*12+col_offset+1][block_idx*7 + 4] = 1
        W[col_offset*2+0][block_idx*7 + 4] -= 1
        W[(col_offset+1)*2+0][block_idx*7 + 4] += 1
        
        # m6
        U[2][block_idx*7 + 5] = 1
        U[0][block_idx*7 + 5] = -1
        V[0*12+col_offset][block_idx*7 + 5] = 1
        V[0*12+col_offset+1][block_idx*7 + 5] = 1
        W[(col_offset+1)*2+1][block_idx*7 + 5] += 1
        
        # m7
        U[1][block_idx*7 + 6] = 1
        U[3][block_idx*7 + 6] = -1
        V[1*12+col_offset][block_idx*7 + 6] = 1
        V[1*12+col_offset+1][block_idx*7 + 6] = 1
        W[col_offset*2+0][block_idx*7 + 6] += 1
        
        col_offset += 2
        block_idx += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
