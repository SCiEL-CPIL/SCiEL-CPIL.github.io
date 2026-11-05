"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use Strassen's algorithm for 2x2 blocks of size 3x3.
    # Base case: 3x3 schoolbook multiplication, rank 27.
    # Total rank: 7 * 27 = 189.
    
    # 1. Construct 3x3 Schoolbook Decomposition (Rank 27)
    # U3, V3, W3 are 9x27
    U3 = np.zeros((9, 27), dtype=int)
    V3 = np.zeros((9, 27), dtype=int)
    W3 = np.zeros((9, 27), dtype=int)
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i * 3 + j, col] = 1
                V3[j * 3 + k, col] = 1
                W3[k * 3 + i, col] = 1
                col += 1
                
    # 2. Construct Strassen Projections
    # A and B are 6x6, flattened to 36.
    # Local 3x3 matrices are flattened to 9.
    
    # Helper to get global index from local (i,j) and block (bi, bj)
    # bi, bj in {0, 1} for top/bottom, left/right
    def get_global_idx(i, j, bi, bj):
        return (i + bi * 3) * 6 + (j + bj * 3)

    # Define the 7 Strassen multiplications
    # Each entry is (left_blocks, right_blocks)
    # left_blocks: list of (bi, bj, coeff)
    # right_blocks: list of (bi, bj, coeff)
    
    strassen_defs = [
        # M1 = (A11 + A22) * (B11 + B22)
        {
            'L': [((0, 0), 1), ((1, 1), 1)],
            'R': [((0, 0), 1), ((1, 1), 1)]
        },
        # M2 = (A21 + A22) * B11
        {
            'L': [((1, 0), 1), ((1, 1), 1)],
            'R': [((0, 0), 1)]
        },
        # M3 = A11 * (B12 - B22)
        {
            'L': [((0, 0), 1)],
            'R': [((0, 1), 1), ((1, 1), -1)]
        },
        # M4 = A22 * (B21 - B11)
        {
            'L': [((1, 1), 1)],
            'R': [((1, 0), 1), ((0, 0), -1)]
        },
        # M5 = (A11 + A12) * B22
        {
            'L': [((0, 0), 1), ((0, 1), 1)],
            'R': [((1, 1), 1)]
        },
        # M6 = (A21 - A11) * (B11 + B12)
        {
            'L': [((1, 0), 1), ((0, 0), -1)],
            'R': [((0, 0), 1), ((0, 1), 1)]
        },
        # M7 = (A12 - A22) * (B21 + B22)
        {
            'L': [((0, 1), 1), ((1, 1), -1)],
            'R': [((1, 0), 1), ((1, 1), 1)]
        }
    ]
    
    # Output combination coefficients for C
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # P_C_k is 36x9
    # We build W by summing contributions from each M_k
    
    total_rank = 7 * 27
    U = np.zeros((36, total_rank), dtype=int)
    V = np.zeros((36, total_rank), dtype=int)
    W = np.zeros((36, total_rank), dtype=int)
    
    for k, defn in enumerate(strassen_defs):
        # Build P_A_k (9x36) and P_B_k (9x36)
        P_A = np.zeros((9, 36), dtype=int)
        P_B = np.zeros((9, 36), dtype=int)
        
        for local_i in range(3):
            for local_j in range(3):
                local_idx = local_i * 3 + local_j
                # Left side
                for (bi, bj), coeff in defn['L']:
                    g_idx = get_global_idx(local_i, local_j, bi, bj)
                    P_A[local_idx, g_idx] += coeff
                # Right side
                for (bi, bj), coeff in defn['R']:
                    g_idx = get_global_idx(local_i, local_j, bi, bj)
                    P_B[local_idx, g_idx] += coeff
                    
        # Build P_C_k (36x9)
        P_C = np.zeros((36, 9), dtype=int)
        
        # Determine which output blocks this M_k contributes to
        # M1: C11(+), C22(+)
        # M2: C21(+), C22(-)
        # M3: C12(+), C22(+)
        # M4: C11(+), C21(+)
        # M5: C11(-), C12(+)
        # M6: C22(+)
        # M7: C11(+)
        
        contributions = {
            0: { (0,0): 1, (1,1): 1 },   # M1
            1: { (1,0): 1, (1,1): -1 },  # M2
            2: { (0,1): 1, (1,1): 1 },   # M3
            3: { (0,0): 1, (1,0): 1 },   # M4
            4: { (0,0): -1, (0,1): 1 },  # M5
            5: { (1,1): 1 },             # M6
            6: { (0,0): 1 }              # M7
        }
        
        for (bi, bj), coeff in contributions[k].items():
            for local_i in range(3):
                for local_j in range(3):
                    local_idx = local_i * 3 + local_j
                    g_idx = get_global_idx(local_i, local_j, bi, bj)
                    P_C[g_idx, local_idx] += coeff
                    
        # Update U, V, W
        # For each of the 27 rank components in 3x3
        for r in range(27):
            c_idx = k * 27 + r
            U[:, c_idx] = P_A.T @ U3[:, r]
            V[:, c_idx] = P_B.T @ V3[:, r]
            W[:, c_idx] = P_C @ W3[:, r]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
