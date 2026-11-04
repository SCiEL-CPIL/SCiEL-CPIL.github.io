"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 33
    
    # Dimensions of the factor matrices
    dim_u = n * m  # 6
    dim_v = m * p  # 18
    dim_w = p * n  # 12
    
    U = np.zeros((dim_u, rank), dtype=np.int64)
    V = np.zeros((dim_v, rank), dtype=np.int64)
    W = np.zeros((dim_w, rank), dtype=np.int64)
    
    col = 0
    
    # Strassen 2x2x2 factors
    # U factors (2x2)
    S_U = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 1], [0, -1]]
    ]
    # V factors (2x2)
    S_V = [
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, -1]],
        [[-1, 0], [1, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, 0], [1, 1]]
    ]
    # W factors (2x2)
    S_W = [
        [[1, 0], [0, 1]],
        [[0, 1], [0, -1]],
        [[0, 0], [1, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 0], [0, 1]],
        [[1, 0], [0, 0]]
    ]
    
    # Standard 2x1x2 factors (for the remaining column of A and corresponding rows of B)
    # A is 2x1, B is 1x2. C is 2x2.
    # Terms: A00*B00 -> C00, A00*B01 -> C01, A10*B00 -> C10, A10*B01 -> C11
    Std_U = [
        [1], [1], [0], [0]
    ]
    Std_V = [
        [1, 0], [0, 1], [1, 0], [0, 1]
    ]
    Std_W = [
        [[1, 0], [0, 0]],
        [[0, 1], [0, 0]],
        [[0, 0], [1, 0]],
        [[0, 0], [0, 1]]
    ]
    
    # There are 3 blocks of B (cols 0-1, 2-3, 4-5)
    for b_block in range(3):
        # Strassen part: A cols 0,1 and B cols 2*b_block, 2*b_block+1
        for r in range(7):
            # U: A indices i=0,1 and j=0,1
            for i in range(2):
                for j in range(2):
                    U[i * m + j, col] = S_U[r][i][j]
            
            # V: B indices j=0,1 and k=2*b_block, 2*b_block+1
            for j in range(2):
                for k_offset in range(2):
                    k_global = 2 * b_block + k_offset
                    V[j * p + k_global, col] = S_V[r][j][k_offset]
            
            # W: C indices k=2*b_block, 2*b_block+1 and i=0,1
            for k_offset in range(2):
                k_global = 2 * b_block + k_offset
                for i in range(2):
                    W[k_global * n + i, col] = S_W[r][k_offset][i]
            
            col += 1
        
        # Standard part: A col 2 and B cols 2*b_block, 2*b_block+1
        for r in range(4):
            # U: A indices i=0,1 and j=2
            for i in range(2):
                U[i * m + 2, col] = Std_U[r][0]
            
            # V: B indices j=2 and k=2*b_block, 2*b_block+1
            for k_offset in range(2):
                k_global = 2 * b_block + k_offset
                V[2 * p + k_global, col] = Std_V[r][k_offset]
            
            # W: C indices k=2*b_block, 2*b_block+1 and i=0,1
            for k_offset in range(2):
                k_global = 2 * b_block + k_offset
                for i in range(2):
                    W[k_global * n + i, col] = Std_W[r][k_offset][i]
            
            col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
