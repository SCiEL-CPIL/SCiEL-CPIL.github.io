"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 8
    # Decompose: 2x2x8 (using Strassen on four 2x2x2 blocks) + 2x1x8 (schoolbook)
    # 2x2x8: 4 blocks of 2x2x2, each rank 7 -> 28
    # 2x1x8: rank 16
    # Total rank = 44
    
    rank = 44
    U = [[0.0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0.0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0.0] * rank for _ in range(p * n)]  # 16 x 44
    
    # Strassen 2x2x2 decomposition
    # U entries for 2x2x2: U[i*2+j, c]
    strassen_U = [
        [1, 0, 1, 0, 1, -1, 0],  # i=0,j=0
        [0, 0, 0, 0, 1, 0, 1],   # i=0,j=1
        [0, 1, 0, 0, 0, 1, 0],   # i=1,j=0
        [1, 1, 0, 1, 0, 0, -1],  # i=1,j=1
    ]
    # V entries: V[j*2+k, c]
    strassen_V = [
        [1, 1, 0, 0, 0, 1, 0],   # j=0,k=0
        [0, 0, 0, 1, 0, 0, 1],   # j=0,k=1
        [0, 0, 1, -1, 0, 1, 0],  # j=1,k=0
        [1, 0, -1, 0, 1, 0, 1],  # j=1,k=1
    ]
    # W entries: W[k*2+i, c]
    strassen_W = [
        [1, 0, 0, 0, -1, 0, 1],  # k=0,i=0
        [0, 0, 1, 1, 1, 0, -1],  # k=0,i=1
        [0, 1, 0, 1, 0, 0, 0],   # k=1,i=0
        [1, 0, -1, 0, 0, 1, 0],  # k=1,i=1
    ]
    
    # Fill in the 2x2x8 part (j=0,1) using 4 Strassen blocks
    # Block (i_block, k_block) where i_block in {0}, k_block in {0,1,2,3}
    # For each k_block, we have k = 2*k_block, 2*k_block+1
    col_offset = 0
    for kb in range(4):  # 4 blocks of 2x2x2
        for c in range(7):
            # U: rows for i=0,1 and j=0,1 -> indices i*3+j
            for i in range(2):
                for j in range(2):
                    U[i * m + j][col_offset + c] = strassen_U[i * 2 + j][c]
            # V: rows for j=0,1 and k=2*kb, 2*kb+1 -> indices j*8 + k
            for j in range(2):
                for dk in range(2):
                    k = 2 * kb + dk
                    V[j * p + k][col_offset + c] = strassen_V[j * 2 + dk][c]
            # W: rows for k=2*kb,2*kb+1 and i=0,1 -> indices k*2+i
            for dk in range(2):
                k = 2 * kb + dk
                for i in range(2):
                    W[k * n + i][col_offset + c] = strassen_W[dk * 2 + i][c]
        col_offset += 7
    
    # Fill in the 2x1x8 part (j=2) using schoolbook: rank 2*1*8 = 16
    for k in range(p):
        for i in range(n):
            U[i * m + 2][col_offset] = 1
            V[2 * p + k][col_offset] = 1
            W[k * n + i][col_offset] = 1
            col_offset += 1
    
    # Convert to integers where possible
    for mat in [U, V, W]:
        for row in mat:
            for idx in range(len(row)):
                val = row[idx]
                if abs(val - round(val)) < 1e-10:
                    row[idx] = int(round(val))
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
