"""Program for finding a rank-minimal exact tensor decomposition for 2x4*4x5 matrix multiplication."""
import numpy as np
import random
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-36 tensor decomposition for 2x4 * 4x5 matrix multiplication.
    
    This construction exploits the direct sum structure of the tensor:
    M_{2,4,5} = M_{2,2,5} \oplus M_{2,2,5}
    where the split is along the middle dimension (j).
    
    Each M_{2,2,5} is decomposed as:
    M_{2,2,5} = M_{2,2,2} \oplus M_{2,2,2} \oplus M_{2,2,1}
    using Strassen's algorithm for the 2x2x2 parts (rank 7) 
    and schoolbook for the 2x2x1 part (rank 4).
    Total rank per M_{2,2,5} = 7 + 7 + 4 = 18.
    Total rank for M_{2,4,5} = 18 + 18 = 36.
    """
    n, m, p = DIMENSIONS  # 2, 4, 5
    rank = 36
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a decomposition block
    # u_offsets: indices in U (i*m + j)
    # v_offsets: indices in V (j*p + k)
    # w_offsets: indices in W (k*n + i)
    # local_u, local_v, local_w: factor matrices for the sub-problem
    
    def add_block(u_offset, v_offset, w_offset, local_u, local_v, local_w):
        r = local_u.shape[1]
        for c in range(r):
            for i in range(local_u.shape[0]):
                U[u_offset[i] + i, c + u_offset[0]] = local_u[i, c]
            for j in range(local_v.shape[0]):
                V[v_offset[j] + j, c + v_offset[0]] = local_v[j, c]
            for k in range(local_w.shape[0]):
                W[w_offset[k] + k, c + w_offset[0]] = local_w[k, c]
        
    # This helper is tricky with offsets. Let's just build the full matrices directly.
    # We will construct the factors for one M_{2,2,5} block, then copy it.
    
    # --- Construct M_{2,2,5} factors (Rank 18) ---
    # Local indices for M_{2,2,5}:
    # A: 2x2 (indices 0,1 for rows, 0,1 for cols)
    # B: 2x5 (indices 0,1 for rows, 0..4 for cols)
    # C: 2x5 (indices 0,1 for rows, 0..4 for cols)
    
    # Sub-block 1: M_{2,2,2} using Strassen (Rank 7)
    # A cols 0,1; B cols 0,1
    # Strassen factors for 2x2x2:
    # U_s[i, c], V_s[j, c], W_s[k, c]
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B12)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((10, 7)) # Wait, B is 2x2 here, so 4 entries? No, V is (m*p, rank).
    # In the sub-problem M_{2,2,2}, m=2, p=2. V size is 2*2=4.
    # But in the full problem, V is (4*5, rank).
    # Let's define local factors for the sub-problems and then map them.
    
    # Let's define the full U, V, W by filling columns.
    
    # Column 0-6: M_{2,2,2} for A[:, 0:2] and B[0:2, 0:2]
    # A indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3 (in local 2x2)
    # In full U (2x4), A row 0 is indices 0,1 (j=0,1). A row 1 is indices 2,3 (j=0,1).
    # So U[0, c] corresponds to A[0,0], U[1, c] to A[0,1], etc.
    
    # Let's use a mapping function.
    # Full U index: i*m + j. i in 0..1, j in 0..3.
    # Full V index: j*p + k. j in 0..3, k in 0..4.
    # Full W index: k*n + i. k in 0..4, i in 0..1.
    
    # Block 1: j in {0,1}, k in {0,1}
    # Strassen for this block.
    # Local A: 2x2. Local B: 2x2.
    
    # Strassen U (4x7):
    # Row 0: A11, Row 1: A12, Row 2: A21, Row 3: A22
    U_s = np.array([
        [1, 1, 1, 0, 1, -1, 0],
        [0, 0, 0, 1, 1, 0, -1],
        [0, 1, 0, 0, 0, 1, 1],
        [1, 0, 0, 1, 0, 0, 0]
    ])
    
    # Strassen V (4x7):
    # Row 0: B11, Row 1: B12, Row 2: B21, Row 3: B22
    V_s = np.array([
        [1, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 1],
        [1, 0, -1, 0, 0, 0, 1],
        [1, 0, 0, -1, 1, 0, 0]
    ])
    
    # Strassen W (4x7):
    # Row 0: C11, Row 1: C12, Row 2: C21, Row 3: C22
    W_s = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0]
    ])
    # Wait, W_s in Strassen is not identity.
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    W_s = np.array([
        [1, 0, 0, -1, 1, 0, -1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Map Strassen to full matrices for Block 1 (j=0,1; k=0,1)
    # U rows: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # V rows: B[0,0]=0, B[0,1]=1, B[1,0]=5, B[1,1]=6 (j*p+k: 0*5+0=0, 0*5+1=1, 1*5+0=5, 1*5+1=6)
    # W rows: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3 (k*n+i: 0*2+0=0, 1*2+0=2? No, W[k*n+i].
    # C[i,k]. i=0, k=0 -> 0*2+0=0.
    # C[i,k]. i=0, k=1 -> 1*2+0=2.
    # C[i,k]. i=1, k=0 -> 0*2+1=1.
    # C[i,k]. i=1, k=1 -> 1*2+1=3.
    
    # Let's re-verify W indexing.
    # Task: W[k*n+i, r].
    # So W[0, r] is C[0,0].
    # W[1, r] is C[1,0]? No, k=0, i=1 -> 0*2+1=1.
    # W[2, r] is C[0,1]? k=1, i=0 -> 1*2+0=2.
    # W[3, r] is C[1,1]? k=1, i=1 -> 1*2+1=3.
    
    # So W indices for C[i,k]:
    # C[0,0] -> 0
    # C[1,0] -> 1
    # C[0,1] -> 2
    # C[1,1] -> 3
    
    # Strassen W_s rows correspond to C11, C12, C21, C22.
    # C11 is C[0,0] -> idx 0
    # C12 is C[0,1] -> idx 2
    # C21 is C[1,0] -> idx 1
    # C22 is C[1,1] -> idx 3
    
    # So we need to permute W_s rows.
    W_s_perm = W_s[[0, 2, 1, 3]]
    
    # Fill Block 1
    for c in range(7):
        U[0, c] = U_s[0, c]
        U[1, c] = U_s[1, c]
        U[2, c] = U_s[2, c]
        U[3, c] = U_s[3, c]
        
        V[0, c] = V_s[0, c]
        V[1, c] = V_s[1, c]
        V[5, c] = V_s[2, c]
        V[6, c] = V_s[3, c]
        
        W[0, c] = W_s_perm[0, c]
        W[2, c] = W_s_perm[1, c]
        W[1, c] = W_s_perm[2, c]
        W[3, c] = W_s_perm[3, c]
        
    # Block 2: M_{2,2,2} for A[:, 0:2] and B[0:2, 2:4]
    # j in {0,1}, k in {2,3}
    # V indices: B[0,2]=2, B[0,3]=3, B[1,2]=7, B[1,3]=8
    # W indices: C[0,2]=4, C[0,3]=5, C[1,2]=6, C[1,3]=7
    
    for c in range(7):
        # U is same as Block 1
        U[0, c+7] = U_s[0, c]
        U[1, c+7] = U_s[1, c]
        U[2, c+7] = U_s[2, c]
        U[3, c+7] = U_s[3, c]
        
        V[2, c+7] = V_s[0, c]
        V[3, c+7] = V_s[1, c]
        V[7, c+7] = V_s[2, c]
        V[8, c+7] = V_s[3, c]
        
        W[4, c+7] = W_s_perm[0, c]
        W[5, c+7] = W_s_perm[1, c]
        W[6, c+7] = W_s_perm[2, c]
        W[7, c+7] = W_s_perm[3, c]
        
    # Block 3: M_{2,2,1} for A[:, 0:2] and B[0:2, 4]
    # j in {0,1}, k in {4}
    # Schoolbook for 2x2x1.
    # C[0,4] = A[0,0]B[0,4] + A[0,1]B[1,4]
    # C[1,4] = A[1,0]B[0,4] + A[1,1]B[1,4]
    # 4 multiplications.
    
    # U rows: A[0,0], A[0,1], A[1,0], A[1,1]
    # V rows: B[0,4]=4, B[1,4]=9
    # W rows: C[0,4]=8, C[1,4]=9
    
    # Col 14: A[0,0] * B[0,4] -> C[0,4]
    U[0, 14] = 1
    V[4, 14] = 1
    W[8, 14] = 1
    
    # Col 15: A[0,1] * B[1,4] -> C[0,4]
    U[1, 15] = 1
    V[9, 15] = 1
    W[8, 15] = 1
    
    # Col 16: A[1,0] * B[0,4] -> C[1,4]
    U[2, 16] = 1
    V[4, 16] = 1
    W[9, 16] = 1
    
    # Col 17: A[1,1] * B[1,4] -> C[1,4]
    U[3, 17] = 1
    V[9, 17] = 1
    W[9, 17] = 1
    
    # Now, we need to add the second half of A (j=2,3)
    # This is another M_{2,2,5} block.
    # A rows 0,1 but columns 2,3.
    # U indices: A[0,2]=4, A[0,3]=5, A[1,2]=6, A[1,3]=7
    # V indices: B[2,0]=10, B[2,1]=11, B[2,2]=12, B[2,3]=13, B[2,4]=14
    #             B[3,0]=15, B[3,1]=16, B[3,2]=17, B[3,3]=18, B[3,4]=19
    # W indices: Same as before, C[0,0..4] and C[1,0..4]
    
    # Block 4: M_{2,2,2} for A[:, 2:4] and B[2:4, 0:2]
    # j in {2,3}, k in {0,1}
    
    for c in range(7):
        U[4, c+18] = U_s[0, c]
        U[5, c+18] = U_s[1, c]
        U[6, c+18] = U_s[2, c]
        U[7, c+18] = U_s[3, c]
        
        V[10, c+18] = V_s[0, c]
        V[11, c+18] = V_s[1, c]
        V[15, c+18] = V_s[2, c]
        V[16, c+18] = V_s[3, c]
        
        W[0, c+18] = W_s_perm[0, c]
        W[2, c+18] = W_s_perm[1, c]
        W[1, c+18] = W_s_perm[2, c]
        W[3, c+18] = W_s_perm[3, c]
        
    # Block 5: M_{2,2,2} for A[:, 2:4] and B[2:4, 2:4]
    # j in {2,3}, k in {2,3}
    
    for c in range(7):
        U[4, c+25] = U_s[0, c]
        U[5, c+25] = U_s[1, c]
        U[6, c+25] = U_s[2, c]
        U[7, c+25] = U_s[3, c]
        
        V[12, c+25] = V_s[0, c]
        V[13, c+25] = V_s[1, c]
        V[17, c+25] = V_s[2, c]
        V[18, c+25] = V_s[3, c]
        
        W[4, c+25] = W_s_perm[0, c]
        W[5, c+25] = W_s_perm[1, c]
        W[6, c+25] = W_s_perm[2, c]
        W[7, c+25] = W_s_perm[3, c]
        
    # Block 6: M_{2,2,1} for A[:, 2:4] and B[2:4, 4]
    # j in {2,3}, k in {4}
    
    # Col 32: A[0,2] * B[2,4] -> C[0,4]
    U[4, 32] = 1
    V[14, 32] = 1
    W[8, 32] = 1
    
    # Col 33: A[0,3] * B[3,4] -> C[0,4]
    U[5, 33] = 1
    V[19, 33] = 1
    W[8, 33] = 1
    
    # Col 34: A[1,2] * B[2,4] -> C[1,4]
    U[6, 34] = 1
    V[14, 34] = 1
    W[9, 34] = 1
    
    # Col 35: A[1,3] * B[3,4] -> C[1,4]
    U[7, 35] = 1
    V[19, 35] = 1
    W[9, 35] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
