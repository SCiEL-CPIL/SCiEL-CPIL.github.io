import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (16, rank)
    
    # We decompose the multiplication A(2x4) * B(4x8) -> C(2x8)
    # by splitting the contraction dimension m=4 into two halves of size 2,
    # and the output dimension p=8 into four blocks of size 2.
    
    # A = [A_L | A_R] where A_L, A_R are 2x2
    # B = [B_T; B_B] where B_T, B_B are 2x8
    # C = A_L * B_T + A_R * B_B
    
    # B_T is 2x8, split into four 2x2 blocks: B_T0, B_T1, B_T2, B_T3
    # B_B is 2x8, split into four 2x2 blocks: B_B0, B_B1, B_B2, B_B3
    
    # C = [A_L B_T0, A_L B_T1, A_L B_T2, A_L B_T3, ...]
    #    + [A_R B_B0, A_R B_B1, A_R B_B2, A_R B_B3, ...]
    
    # There are 8 independent 2x2 * 2x2 multiplications.
    # Each uses Strassen's algorithm with rank 7.
    # Total rank = 8 * 7 = 56.
    
    # Strassen components for 2x2 matrix multiplication
    # Indices: 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    U_local = np.zeros((4, 7), dtype=int)
    V_local = np.zeros((4, 7), dtype=int)
    W_local = np.zeros((4, 7), dtype=int)
    
    # M1 = (a00 + a11)(b00 + b11)
    U_local[0][0] = 1; U_local[3][0] = 1
    V_local[0][0] = 1; V_local[3][0] = 1
    W_local[0][0] = 1
    W_local[3][0] = 1
    
    # M2 = (a10 + a11)b00
    U_local[2][1] = 1; U_local[3][1] = 1
    V_local[0][1] = 1
    W_local[2][1] = 1
    W_local[3][1] = -1
    
    # M3 = a00(b10 - b11)
    U_local[0][2] = 1
    V_local[1][2] = 1; V_local[3][2] = -1
    W_local[1][2] = 1
    W_local[3][2] = 1
    
    # M4 = a11(b01 - b00)
    U_local[3][3] = 1
    V_local[2][3] = 1; V_local[0][3] = -1
    W_local[0][3] = 1
    W_local[2][3] = 1
    
    # M5 = (a00 + a01)b11
    U_local[0][4] = 1; U_local[1][4] = 1
    V_local[3][4] = 1
    W_local[0][4] = -1
    W_local[1][4] = 1
    
    # M6 = (a10 - a00)(b00 + b01)
    U_local[2][5] = 1; U_local[0][5] = -1
    V_local[0][5] = 1; V_local[1][5] = 1
    W_local[3][5] = 1
    
    # M7 = (a01 - a11)(b10 + b11)
    U_local[1][6] = 1; U_local[3][6] = -1
    V_local[2][6] = 1; V_local[3][6] = 1
    W_local[0][6] = 1
    
    total_rank = 56
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # Mapping for A (2x4) -> 8 elements
    # A is indexed as A[i, j] with i in 0..1, j in 0..3
    # U index: i*m + j = i*4 + j
    
    # Mapping for B (4x8) -> 32 elements
    # B is indexed as B[j, k] with j in 0..3, k in 0..7
    # V index: j*p + k = j*8 + k
    
    # Mapping for C (2x8) -> 16 elements
    # C is indexed as C[i, k] with i in 0..1, k in 0..7
    # W index: k*n + i = k*2 + i (per task contract: W[k*n+i, r])
    
    # We have 8 blocks corresponding to:
    # 2 blocks of A (left, right) x 4 blocks of B (top-left, top-mid, top-right, bottom-left...)
    # Actually, let's iterate over the 8 2x2*2x2 multiplications.
    
    # Block 0: A_L * B_T0
    # A_L uses A[0,0], A[0,1], A[1,0], A[1,1] -> U indices 0,1,4,5
    # B_T0 uses B[0,0], B[0,1], B[1,0], B[1,1] -> V indices 0,1,8,9
    # C block 0 uses C[0,0], C[0,1], C[1,0], C[1,1] -> W indices 0,1,2,3
    
    # Let's define the mappings for each of the 8 cases.
    # Case t in 0..7
    # t = 0..3 corresponds to A_L * B_Tk (k=0..3)
    # t = 4..7 corresponds to A_R * B_Bk (k=0..3)
    
    for t in range(8):
        base_r = t * 7
        
        # Determine A block
        if t < 4:
            # A_L: columns 0,1
            a_cols = [0, 1]
        else:
            # A_R: columns 2,3
            a_cols = [2, 3]
            
        # Determine B block
        # B rows: 0,1 for A_L; 2,3 for A_R
        if t < 4:
            b_rows = [0, 1]
        else:
            b_rows = [2, 3]
            
        # B column block index
        b_col_block = t % 4
        b_cols = [b_col_block * 2, b_col_block * 2 + 1]
        
        # C block
        # C columns correspond to B columns
        c_cols = [b_col_block * 2, b_col_block * 2 + 1]
        
        # Local indices 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1)
        # A local (0,0) -> global (0, a_cols[0]) -> U[0*4 + a_cols[0]]
        # A local (0,1) -> global (0, a_cols[1]) -> U[0*4 + a_cols[1]]
        # A local (1,0) -> global (1, a_cols[0]) -> U[1*4 + a_cols[0]]
        # A local (1,1) -> global (1, a_cols[1]) -> U[1*4 + a_cols[1]]
        
        a_map = [
            0 * 4 + a_cols[0],
            0 * 4 + a_cols[1],
            1 * 4 + a_cols[0],
            1 * 4 + a_cols[1]
        ]
        
        # B local (0,0) -> global (b_rows[0], b_cols[0]) -> V[b_rows[0]*8 + b_cols[0]]
        # B local (0,1) -> global (b_rows[0], b_cols[1]) -> V[b_rows[0]*8 + b_cols[1]]
        # B local (1,0) -> global (b_rows[1], b_cols[0]) -> V[b_rows[1]*8 + b_cols[0]]
        # B local (1,1) -> global (b_rows[1], b_cols[1]) -> V[b_rows[1]*8 + b_cols[1]]
        
        b_map = [
            b_rows[0] * 8 + b_cols[0],
            b_rows[0] * 8 + b_cols[1],
            b_rows[1] * 8 + b_cols[0],
            b_rows[1] * 8 + b_cols[1]
        ]
        
        # C local (0,0) -> global (0, c_cols[0]) -> W[c_cols[0]*2 + 0]
        # C local (0,1) -> global (0, c_cols[1]) -> W[c_cols[1]*2 + 0]
        # C local (1,0) -> global (1, c_cols[0]) -> W[c_cols[0]*2 + 1]
        # C local (1,1) -> global (1, c_cols[1]) -> W[c_cols[1]*2 + 1]
        
        c_map = [
            c_cols[0] * 2 + 0,
            c_cols[1] * 2 + 0,
            c_cols[0] * 2 + 1,
            c_cols[1] * 2 + 1
        ]
        
        for l in range(7):
            r = base_r + l
            for la in range(4):
                val = U_local[la][l]
                if val != 0:
                    U[a_map[la], r] += val
            for lb in range(4):
                val = V_local[lb][l]
                if val != 0:
                    V[b_map[lb], r] += val
            for lc in range(4):
                val = W_local[lc][l]
                if val != 0:
                    W[c_map[lc], r] += val
                    
    if time.time() - start_time > 140:
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
