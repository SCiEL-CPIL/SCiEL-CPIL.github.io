import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x9 matrix.
    
    Uses a hybrid approach:
    1. Apply Kronecker product structure to split 3 = 3*1, 9 = 3*3
    2. For the 2x3x3 core, use 9 rank-1 terms (direct outer products)
    3. Verify and potentially improve with Strassen on sub-blocks
    
    Returns: dict with 'U', 'V', 'W' factor matrices
    """
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    # Strategy: Direct decomposition using all 6*9=54 possible products,
    # then apply Strassen to 2x2x2 blocks where beneficial.
    
    # The multiplication tensor T is defined by:
    # C[k*n + i, r] = sum_j A[i,j] * B[j,k*n+r]
    # Equivalently: C[i, r] = sum_{j=0}^{2} sum_{k=0}^{8} A[i,j] * B[j, k] * delta_{r, k}
    
    # Direct rank: 54 (one rank-1 term for each (i,j,k) triple)
    # But we can group by k: for fixed k, we have a 2x3 rank-1 outer product A * B[:,k]
    # Actually, for fixed k, C[:,k] = A * B[:,k], which is a matrix product.
    
    # Let's use the Kronecker structure: B has shape 3x9 = 3x(3*3)
    # Reshape B as B_res[3,3,3] where B_res[j,k,l] = B[j, 3*k+l]
    # Then C[i, 3*k+l] = sum_j A[i,j] * B_res[j,k,l]
    
    # For fixed k, C[:, 3*k : 3*k+3] = A * B_res[:, k, :]
    # This is a 2x3 * 3x3 = 2x3 matrix product, rank = 9 for direct, 
    # but Strassen gives rank 7 for the 2x2x2 part.
    
    # Let's split A's columns into 2+1 and B's rows into 2+1 for each block.
    
    # Approach: For each of the 3 k-blocks:
    #   Use A[0:2, 0:2] * B[0:2, :] (2x2 * 2x3) with Strassen (rank 7)
    #   Use A[0:2, 2] * B[2, :] (2x1 * 1x3) with 6 direct terms
    #   Total per block: 7 + 6 = 13
    #   Total: 3 * 13 = 39
    
    rank = 39
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    
    for k_block in range(3):
        # Columns 3*k_block to 3*k_block+2 of C
        c_start = 3 * k_block
        
        # Part A: A[0:2,0:2] * B[0:2, 3*k_block:3*k_block+3]
        # This is 2x2 * 2x3, decompose 2x3 as two 2x2 blocks? No, it's 2x2*2x3.
        # Actually, we can treat the 2x3 output as: first 2 columns use Strassen on 2x2x2,
        # last column uses 2x2x1 direct (rank 4).
        # But 2x2*2x3 isn't directly Strassen-able. Let's use:
        # Split the 3 output columns into 2+1.
        # First 2 columns: 2x2 * 2x2, use Strassen (rank 7)
        # Last 1 column: 2x2 * 2x1, use 4 direct multiplications (rank 4)
        # Total: 11 for this sub-block.
        
        # B block: B[0:2, 3*k_block : 3*k_block+3] has shape 2x3
        # B_block_col0 = B[0, 3*k_block], B_block_col1 = B[1, 3*k_block]
        # B_block_col0b = B[0, 3*k_block+1], B_block_col1b = B[1, 3*k_block+1]
        # B_block_col0c = B[0, 3*k_block+2], B_block_col1c = B[1, 3*k_block+2]
        
        # For the 2x2x2 Strassen part (columns 0,1 of this block):
        u_a = 0  # A[0,0]
        u_b = 1  # A[0,1]
        u_c = 3  # A[1,0]  (0*3+0)
        u_d = 4  # A[1,1]  (1*3+1)
        
        v_e = 0 * p + (3 * k_block)       # B[0, 3*k_block]
        v_f = 0 * p + (3 * k_block + 1)   # B[0, 3*k_block+1]
        v_g = 1 * p + (3 * k_block)       # B[1, 3*k_block]
        v_h = 1 * p + (3 * k_block + 1)   # B[1, 3*k_block+1]
        
        # Strassen 7 multiplications:
        # M1 = (a+d)(e+h)
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        
        # M2 = (c+d)e
        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        
        # M3 = a(f-h)
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_h][col+2] -= 1
        
        # M4 = d(g-e)
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        V[v_e][col+3] -= 1
        
        # M5 = (a+b)h
        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        
        # M6 = (c-a)(e+f)
        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1
        
        # M7 = (b-d)(g+h)
        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g][col+6] += 1
        V[v_h][col+6] += 1
        
        # W indices for C[0, 3*k_block] and C[0, 3*k_block+1], C[1, 3*k_block] and C[1, 3*k_block+1]
        w_00 = (3 * k_block) * n + 0   # C[0, 3*k_block]
        w_01 = (3 * k_block + 1) * n + 0  # C[0, 3*k_block+1]
        w_10 = (3 * k_block) * n + 1   # C[1, 3*k_block]
        w_11 = (3 * k_block + 1) * n + 1  # C[1, 3*k_block+1]
        
        # C[0, 3*k_block] = M1 + M4 - M5 + M7
        W[w_00][col] += 1
        W[w_00][col+3] += 1
        W[w_00][col+4] -= 1
        W[w_00][col+6] += 1
        
        # C[0, 3*k_block+1] = M3 + M5
        W[w_01][col+2] += 1
        W[w_01][col+4] += 1
        
        # C[1, 3*k_block] = M2 + M4
        W[w_10][col+1] += 1
        W[w_10][col+3] += 1
        
        # C[1, 3*k_block+1] = M1 - M2 + M3 + M6
        W[w_11][col] += 1
        W[w_11][col+1] -= 1
        W[w_11][col+2] += 1
        W[w_11][col+5] += 1
        
        col += 7
        
        # Part B: 2x2 * 2x1 direct (column 3*k_block+2)
        # A[0:2,0:2] * B[0:2, 3*k_block+2]
        v_i = 0 * p + (3 * k_block + 2)  # B[0, 3*k_block+2]
        v_j = 1 * p + (3 * k_block + 2)  # B[1, 3*k_block+2]
        
        # P1 = a*e
        U[u_a][col] += 1
        V[v_i][col] += 1
        
        # P2 = b*f
        U[u_b][col+1] += 1
        V[v_j][col+1] += 1
        
        # P3 = c*e
        U[u_c][col+2] += 1
        V[v_i][col+2] += 1
        
        # P4 = d*f
        U[u_d][col+3] += 1
        V[v_j][col+3] += 1
        
        w_02 = (3 * k_block + 2) * n + 0  # C[0, 3*k_block+2]
        w_12 = (3 * k_block + 2) * n + 1  # C[1, 3*k_block+2]
        
        # C[0, 3*k_block+2] = P1 + P2
        W[w_02][col] += 1
        W[w_02][col+1] += 1
        
        # C[1, 3*k_block+2] = P3 + P4
        W[w_12][col+2] += 1
        W[w_12][col+3] += 1
        
        col += 4
        
        # Part C: A[0:2, 2] * B[2, 3*k_block:3*k_block+3] (2x1 * 1x3 outer product)
        # This is 2*3 = 6 scalar multiplications
        u_ar0 = 0 * m + 2  # A[0,2]
        u_ar1 = 1 * m + 2  # A[1,2]
        
        for l in range(3):
            v_b = 2 * p + (3 * k_block + l)  # B[2, 3*k_block+l]
            
            # Q_l = A[0,2] * B[2, 3*k_block+l]
            U[u_ar0][col] += 1
            V[v_b][col] += 1
            W[(3 * k_block + l) * n + 0][col] += 1
            
            # R_l = A[1,2] * B[2, 3*k_block+l]
            U[u_ar1][col+1] += 1
            V[v_b][col+1] += 1
            W[(3 * k_block + l) * n + 1][col+1] += 1
            
            col += 2
    
    # Verify col == rank
    assert col == rank, f"Column count mismatch: {col} != {rank}"
    
    # Convert to lists as required by the interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
