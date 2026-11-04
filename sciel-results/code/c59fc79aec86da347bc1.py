import numpy as np
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition piece by piece.
    # The total rank is the sum of the ranks of the components.
    
    # Component 1: 4x4 matrix multiplication (top-left block)
    # Uses Strassen's algorithm for 4x4 (tensor product of two 2x2 Strassen)
    # Rank: 49
    U_44, V_44, W_44 = _strassen_4x4()
    
    # Component 2: 4x4 matrix times 4x1 vector (top-right block of C)
    # A_11 (4x4) * B_12 (4x1) -> C_12 (4x1)
    # Rank: 16
    U_41, V_41, W_41 = _mat_vec_4x4_4x1()
    
    # Component 3: 1x4 vector times 4x4 matrix (bottom-left block of C)
    # A_21 (1x4) * B_11 (4x4) -> C_21 (1x4)
    # Rank: 16
    U_14, V_14, W_14 = _vec_mat_1x4_4x4()
    
    # Component 4: 4x1 vector times 1x1 scalar (part of C_12)
    # A_12 (4x1) * B_22 (1x1) -> C_12 (4x1)
    # Rank: 4
    U_41_scalar, V_41_scalar, W_41_scalar = _vec_scalar_4x1_1x1()
    
    # Component 5: 1x1 scalar times 1x4 vector (part of C_21)
    # A_22 (1x1) * B_21 (1x4) -> C_21 (1x4)
    # Rank: 4
    U_14_scalar, V_14_scalar, W_14_scalar = _scalar_vec_1x1_1x4()
    
    # Component 6: 1x4 vector times 4x1 vector (part of C_22)
    # A_21 (1x4) * B_12 (4x1) -> C_22 (1x1)
    # Rank: 4
    U_dot, V_dot, W_dot = _dot_product_1x4_4x1()
    
    # Component 7: 1x1 scalar times 1x1 scalar (part of C_22)
    # A_22 (1x1) * B_22 (1x1) -> C_22 (1x1)
    # Rank: 1
    U_scalar, V_scalar, W_scalar = _scalar_product()

    # Combine all components
    U = []
    V = []
    W = []
    
    # Helper to append
    def append(U_list, V_list, W_list, U_new, V_new, W_new, offset_i, offset_j, offset_k):
        # U_new shape: (n_block, rank_block)
        # We need to map the local indices to global indices
        # Global index for U: i * m + j
        # Global index for V: j * p + k
        # Global index for W: k * n + i
        
        # This is tricky because the blocks are at specific positions.
        # It is easier to build the full matrices directly.
        pass

    # Let's build the full matrices directly to avoid index mapping errors.
    total_rank = 49 + 16 + 16 + 4 + 4 + 4 + 1
    U_full = np.zeros((25, total_rank))
    V_full = np.zeros((25, total_rank))
    W_full = np.zeros((25, total_rank))
    
    col = 0
    # 1. 4x4 block (indices 0-3)
    # U_44 is 16x49. Maps to global U rows 0-15 (i=0..3, j=0..3)
    # V_44 is 16x49. Maps to global V rows 0-15 (j=0..3, k=0..3)
    # W_44 is 16x49. Maps to global W rows 0-15 (k=0..3, i=0..3)
    for r in range(16):
        U_full[r, col:col+49] = U_44[r, :]
        V_full[r, col:col+49] = V_44[r, :]
        W_full[r, col:col+49] = W_44[r, :]
    col += 49

    # 2. A_11 * B_12 (4x4 * 4x1)
    # A_11 indices: i=0..3, j=0..3 (U rows 0-15)
    # B_12 indices: j=0..3, k=4 (V rows 0..12 with k=4? No. V index is j*p+k. p=5.
    # j=0..3, k=4 -> V rows 4, 9, 14, 19.
    # C_12 indices: i=0..3, k=4 (W rows 4, 9, 14, 19? No. W index is k*n+i.
    # k=4, i=0..3 -> W rows 20, 21, 22, 23.
    
    # U_41 is 16x16.
    # V_41 is 4x16 (only the k=4 part of B is used).
    # W_41 is 4x16.
    
    # We need to place U_41 into U_full rows 0-15.
    # Place V_41 into V_full rows corresponding to B_12.
    # Place W_41 into W_full rows corresponding to C_12.
    
    # This manual mapping is error-prone.
    # Let's use a generic function to add a block decomposition.
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

def _strassen_4x4():
    # Returns U, V, W for 4x4 matrix multiplication, rank 49.
    # U, V, W are 16x49.
    pass

# ... other helpers ...
# EVOLVE-BLOCK-END
