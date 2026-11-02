import numpy as np

DIMENSIONS = [4, 4, 8]

def _strassen_2x2():
    # Returns U, V, W for 2x2x2, rank 7
    # U: 4x7, V: 4x7, W: 4x7
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # M1: (a+d)(e+h)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # M2: (c+d)e
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1; W[3,1] = -1
    
    # M3: a(f-h)
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[1,2] = 1; W[3,2] = 1
    
    # M4: d(g-e)
    U[3,3] = 1
    V[0,3] = -1; V[2,3] = 1
    W[0,3] = -1; W[1,3] = 1
    
    # M5: (a+b)h
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = 1; W[2,4] = 1
    
    # M6: (c-a)(e+f)
    U[0,5] = -1; U[2,5] = 1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    
    # M7: (b-d)(g+h)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = -1
    
    return U, V, W

def _strassen_4x4():
    # Returns U, V, W for 4x4x4, rank 49
    U_22, V_22, W_22 = _strassen_2x2()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Block indices in 4x4 matrix (row-major)
    # Block (br, bc) covers rows 2*br..2*br+1, cols 2*bc..2*bc+1
    # Indices:
    # (0,0): 0,1,4,5
    # (0,1): 2,3,6,7
    # (1,0): 8,9,12,13
    # (1,1): 10,11,14,15
    
    def get_block_indices(br, bc):
        r0 = 2 * br
        c0 = 2 * bc
        return [r0 * 4 + c0, r0 * 4 + c0 + 1, (r0+1) * 4 + c0, (r0+1) * 4 + c0 + 1]
    
    # The 7 block combinations
    # Each entry is (A_blocks, B_blocks, C_blocks)
    # A_blocks: list of (br, bc, coeff)
    # B_blocks: list of (br, bc, coeff)
    # C_blocks: list of (br, bc, coeff)
    
    combinations = [
        # M1: (A00+A11)(B00+B11) -> C00+C11
        ([ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ]),
        # M2: (A10+A11)B00 -> C10-C11
        ([ (1,0,1), (1,1,1) ], [ (0,0,1) ], [ (1,0,1), (1,1,-1) ]),
        # M3: A00(B01-B11) -> C01+C11
        ([ (0,0,1) ], [ (0,1,1), (1,1,-1) ], [ (0,1,1), (1,1,1) ]),
        # M4: A11(B10-B00) -> -C00+C01
        ([ (1,1,1) ], [ (1,0,1), (0,0,-1) ], [ (0,0,-1), (0,1,1) ]),
        # M5: (A00+A01)B11 -> C00+C10
        ([ (0,0,1), (0,1,1) ], [ (1,1,1) ], [ (0,0,1), (1,0,1) ]),
        # M6: (A10-A00)(B00+B01) -> C11
        ([ (1,0,1), (0,0,-1) ], [ (0,0,1), (0,1,1) ], [ (1,1,1) ]),
        # M7: (A01-A11)(B10+B11) -> -C00
        ([ (0,1,1), (1,1,-1) ], [ (1,0,1), (1,1,1) ], [ (0,0,-1) ]),
    ]
    
    for k, (a_blocks, b_blocks, c_blocks) in enumerate(combinations):
        base_col = k * 7
        for j in range(7):
            col = base_col + j
            # U
            for br, bc, coeff in a_blocks:
                idxs = get_block_indices(br, bc)
                for i in range(4):
                    U[idxs[i], col] += coeff * U_22[i, j]
            # V
            for br, bc, coeff in b_blocks:
                idxs = get_block_indices(br, bc)
                for i in range(4):
                    V[idxs[i], col] += coeff * V_22[i, j]
            # W
            for br, bc, coeff in c_blocks:
                idxs = get_block_indices(br, bc)
                for i in range(4):
                    W[idxs[i], col] += coeff * W_22[i, j]
                    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # We are doing 4x4 * 4x8
    # Split B into two 4x4 blocks
    # Rank = 2 * Rank(4x4x4) = 2 * 49 = 98
    
    U_44, V_44, W_44 = _strassen_4x4()
    
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # First 49 columns: B1, C1
    U[:, 0:49] = U_44
    V[0:16, 0:49] = V_44
    W[0:16, 0:49] = W_44
    
    # Next 49 columns: B2, C2
    U[:, 49:98] = U_44
    V[16:32, 49:98] = V_44
    W[16:32, 49:98] = W_44
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
