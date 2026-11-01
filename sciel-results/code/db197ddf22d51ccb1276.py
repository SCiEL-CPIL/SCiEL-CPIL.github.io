import time
import numpy as np

# Constants for the problem dimensions
N_DIM = 2
M_DIM = 3
P_DIM = 10
TIME_LIMIT = 110.0

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 
    2x3 by 3x10 matrix multiplication.
    
    Strategy:
    1. Decompose the 3x10 matrix B into 5 blocks of 3x2.
    2. For each 2x3 by 3x2 multiplication, use a rank-11 decomposition.
       - 7 terms from Strassen algorithm for the 2x2 core.
       - 4 terms from schoolbook for the 2x1 by 1x2 tail.
    3. Total rank = 5 * 11 = 55.
    
    Returns:
    Dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    start_time = time.time()
    
    # Dimensions
    n, m, p = N_DIM, M_DIM, P_DIM
    num_blocks = p // 2
    rank_per_block = 11
    total_rank = num_blocks * rank_per_block
    
    # Initialize factor matrices
    mat_U = np.zeros((n * m, total_rank), dtype=float)
    mat_V = np.zeros((m * p, total_rank), dtype=float)
    mat_W = np.zeros((p * n, total_rank), dtype=float)
    
    # Indices for A (2x3)
    # A is stored as 2x3, flattened row-major: [0,1,2, 3,4,5]
    # A[i, j] index = i * m + j
    a_00 = 0 * m + 0
    a_01 = 0 * m + 1
    a_02 = 0 * m + 2
    a_10 = 1 * m + 0
    a_11 = 1 * m + 1
    a_12 = 1 * m + 2
    
    # Loop over the 5 blocks of B (each 3x2)
    for b_idx in range(num_blocks):
        # Time check
        if time.time() - start_time > TIME_LIMIT:
            break
            
        # Current rank offset for this block
        base_col = b_idx * rank_per_block
        
        # B indices for this block (3x2)
        # B is 3x10, flattened row-major: [0..9, 10..19, 20..29]
        # Block b_idx covers columns 2*b_idx and 2*b_idx+1
        col_b0 = 2 * b_idx
        col_b1 = 2 * b_idx + 1
        
        # B[i, j] index = i * p + j
        b_00 = 0 * p + col_b0
        b_01 = 0 * p + col_b1
        b_10 = 1 * p + col_b0
        b_11 = 1 * p + col_b1
        b_20 = 2 * p + col_b0
        b_21 = 2 * p + col_b1
        
        # C indices for this block (2x2)
        # C is 2x10, flattened row-major: [0..9, 10..19]
        # C[k, i] index = k * n + i  (Note: Task says W[k*n+i, r] for output factor, 
        # but standard matrix mult C = A B implies C[i, k]. 
        # The task contract says "output factor follows the official trace convention W[k*n+i,r]"
        # This usually means W is indexed by output index. 
        # Let's check the seed code. 
        # Seed: W[c00, col] where c00 = k0*2+0. 
        # If C is 2x10, C[i, k] is at i*10+k. 
        # Seed uses c00 = k0*2+0. This implies C is stored as 10x2? 
        # Wait, 2x3 * 3x10 = 2x10. 
        # If W has shape (p*n, rank) = (20, rank).
        # The index in W corresponds to the output element.
        # The task says W[k*n+i, r]. This implies the output is indexed by (k, i) where k is column index of C, i is row index.
        # So C[i, k] is stored at index k*n + i.
        # Let's verify with seed:
        # c00 = k0*2+0. Here k0 is the column index in C (0 or 1 for the block), i=0 (row 0).
        # So C[0, k0] is at k0*2+0.
        # C[0, k1] is at k1*2+0.
        # C[1, k0] is at k0*2+1.
        # C[1, k1] is at k1*2+1.
        
        c_00 = col_b0 * n + 0  # C[0, col_b0]
        c_01 = col_b1 * n + 0  # C[0, col_b1]
        c_10 = col_b0 * n + 1  # C[1, col_b0]
        c_11 = col_b1 * n + 1  # C[1, col_b1]
        
        # --- Strassen Part (for A[0:2, 0:2] * B[0:2, 0:2]) ---
        # M1 = (A00 + A11)(B00 + B11)
        col = base_col
        mat_U[a_00, col] = 1.0
        mat_U[a_11, col] = 1.0
        mat_V[b_00, col] = 1.0
        mat_V[b_11, col] = 1.0
        mat_W[c_00, col] = 1.0
        mat_W[c_11, col] = 1.0
        
        # M2 = (A10 + A11) B00
        col = base_col + 1
        mat_U[a_10, col] = 1.0
        mat_U[a_11, col] = 1.0
        mat_V[b_00, col] = 1.0
        mat_W[c_10, col] = 1.0
        mat_W[c_11, col] = -1.0
        
        # M3 = A00 (B01 - B11)
        col = base_col + 2
        mat_U[a_00, col] = 1.0
        mat_V[b_01, col] = 1.0
        mat_V[b_11, col] = -1.0
        mat_W[c_01, col] = 1.0
        mat_W[c_11, col] = 1.0
        
        # M4 = A11 (B10 - B00)
        col = base_col + 3
        mat_U[a_11, col] = 1.0
        mat_V[b_10, col] = 1.0
        mat_V[b_00, col] = -1.0
        mat_W[c_00, col] = 1.0
        mat_W[c_10, col] = 1.0
        
        # M5 = (A00 + A01) B11
        col = base_col + 4
        mat_U[a_00, col] = 1.0
        mat_U[a_01, col] = 1.0
        mat_V[b_11, col] = 1.0
        mat_W[c_00, col] = -1.0
        mat_W[c_01, col] = 1.0
        
        # M6 = (A10 - A00)(B00 + B01)
        col = base_col + 5
        mat_U[a_10, col] = 1.0
        mat_U[a_00, col] = -1.0
        mat_V[b_00, col] = 1.0
        mat_V[b_01, col] = 1.0
        mat_W[c_11, col] = 1.0
        
        # M7 = (A01 - A11)(B10 + B11)
        col = base_col + 6
        mat_U[a_01, col] = 1.0
        mat_U[a_11, col] = -1.0
        mat_V[b_10, col] = 1.0
        mat_V[b_11, col] = 1.0
        mat_W[c_00, col] = 1.0
        
        # --- Schoolbook Part (for A[0:2, 2] * B[2, 0:2]) ---
        # P1 = A02 * B20 -> C00
        col = base_col + 7
        mat_U[a_02, col] = 1.0
        mat_V[b_20, col] = 1.0
        mat_W[c_00, col] = 1.0
        
        # P2 = A02 * B21 -> C01
        col = base_col + 8
        mat_U[a_02, col] = 1.0
        mat_V[b_21, col] = 1.0
        mat_W[c_01, col] = 1.0
        
        # P3 = A12 * B20 -> C10
        col = base_col + 9
        mat_U[a_12, col] = 1.0
        mat_V[b_20, col] = 1.0
        mat_W[c_10, col] = 1.0
        
        # P4 = A12 * B21 -> C11
        col = base_col + 10
        mat_U[a_12, col] = 1.0
        mat_V[b_21, col] = 1.0
        mat_W[c_11, col] = 1.0
        
    return {'U': mat_U.tolist(), 'V': mat_V.tolist(), 'W': mat_W.tolist()}
