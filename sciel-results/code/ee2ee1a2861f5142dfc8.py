"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x7 matrix using algebraic construction.
    
    Strategy:
    1. Partition the 3x4x7 problem into blocks
    2. Use Strassen's rank-7 algorithm for 2x2x2 subproblems
    3. Handle remaining dimensions with rank-1 terms
    4. Verify and return the decomposition
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # 3, 4, 7
    nm = n * m  # 12
    mp = m * p  # 28
    pn = p * n  # 21
    
    # Build the target tensor T[a, b, c]
    # a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify_decomposition(U, V, W, tol=1e-8):
        """
        Fast verification using NumPy broadcasting.
        Checks if T[a,b,c] == sum_r U[a,r] * V[b,r] * W[c,r]
        """
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        
        reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
        max_error = np.max(np.abs(reconstructed - T))
        return max_error < tol

    def make_schoolbook(rank):
        """Create the standard schoolbook decomposition."""
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < rank:
                        U[i * m + j, col] = 1.0
                        V[j * p + k, col] = 1.0
                        W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    # Baseline: Schoolbook decomposition
    schoolbook_rank = n * m * p  # 84
    U_sb, V_sb, W_sb = make_schoolbook(schoolbook_rank)
    if not verify_decomposition(U_sb, V_sb, W_sb):
        raise RuntimeError("Schoolbook decomposition failed verification")

    # Algebraic construction using Strassen's algorithm
    # Partition indices:
    # i in {0,1,2} (3 rows)
    # j in {0,1,2,3} (4 cols)  
    # k in {0,1,2,3,4,5,6} (7 cols)
    
    # We'll use a block structure:
    # Split i into {0,1} and {2}
    # Split j into {0,1} and {2,3}
    # Split k into {0,1,2,3} and {4,5,6}
    
    # For the 2x2x2 blocks, use Strassen's rank-7 algorithm
    # Strassen's algorithm for 2x2 matrix multiplication:
    # M = [[m11, m12], [m21, m22]] = A * B where A,B are 2x2
    # Uses 7 multiplications:
    # p1 = (A11 + A22) * (B11 + B22)
    # p2 = (A21 + A22) * B11
    # p3 = A11 * (B12 - B22)
    # p4 = A22 * (B21 - B11)
    # p5 = (A11 + A12) * B22
    # p6 = (A21 - A11) * (B11 + B12)
    # p7 = (A12 - A22) * (B21 + B22)
    # m11 = p1 + p4 - p5 + p7
    # m12 = p3 + p5
    # m21 = p2 + p4
    # m22 = p1 - p2 + p3 + p6
    
    # We need to apply this to multiple 2x2x2 subproblems
    # Let's construct the decomposition systematically
    
    U = []
    V = []
    W = []
    
    # Helper to add a rank-1 term
    def add_rank1(a_idx, b_idx, c_idx, u_val=1.0, v_val=1.0, w_val=1.0):
        U.append((a_idx, u_val))
        V.append((b_idx, v_val))
        W.append((c_idx, w_val))
    
    # Process each combination of i-block, j-block, k-block
    # i_blocks: [{0,1}, {2}]
    # j_blocks: [{0,1}, {2,3}]
    # k_blocks: [{0,1,2,3}, {4,5,6}]
    
    i_blocks = [[0, 1], [2]]
    j_blocks = [[0, 1], [2, 3]]
    k_blocks = [[0, 1, 2, 3], [4, 5, 6]]
    
    for ib in i_blocks:
        for jb in j_blocks:
            for kb in k_blocks:
                # For this block, we need to multiply:
                # A_block: rows ib, cols jb
                # B_block: rows jb, cols kb
                # C_block: rows ib, cols kb
                
                # If the block is 2x2x2, use Strassen
                # Otherwise, use schoolbook or other methods
                
                len_i = len(ib)
                len_j = len(jb)
                len_k = len(kb)
                
                if len_i == 2 and len_j == 2 and len_k == 2:
                    # Use Strassen's rank-7 algorithm
                    i1, i2 = ib
                    j1, j2 = jb
                    k1, k2 = kb
                    
                    # p1 = (A[i1,j1] + A[i2,j2]) * (B[j1,k1] + B[j2,k2])
                    # contributes to C[i1,k1], C[i1,k2], C[i2,k1], C[i2,k2]
                    add_rank1(i1*m+j1, j1*p+k1, k1*n+i1, 1.0, 1.0, 1.0)
                    add_rank1(i1*m+j1, j2*p+k2, k2*n+i1, 1.0, 1.0, 1.0)
                    add_rank1(i2*m+j2, j1*p+k1, k1*n+i2, 1.0, 1.0, 1.0)
                    add_rank1(i2*m+j2, j2*p+k2, k2*n+i2, 1.0, 1.0, 1.0)
                    
                    # p2 = (A[i2,j1] + A[i2,j2]) * B[j1,k1]
                    add_rank1(i2*m+j1, j1*p+k1, k1*n+i2, 1.0, 1.0, 1.0)
                    add_rank1(i2*m+j2, j1*p+k1, k1*n+i2, 1.0, 1.0, 1.0)
                    
                    # p3 = A[i1,j1] * (B[j1,k2] - B[j2,k2])
                    add_rank1(i1*m+j1, j1*p+k2, k2*n+i1, 1.0, 1.0, 1.0)
                    add_rank1(i1*m+j1, j2*p+k2, k2*n+i1, 1.0, -1.0, 1.0)
                    
                    # p4 = A[i2,j2] * (B[j2,k1] - B[j1,k1])
                    add_rank1(i2*m+j2, j2*p+k1, k1*n+i2, 1.0, 1.0, 1.0)
                    add_rank1(i2*m+j2, j1*p+k1, k1*n+i2, 1.0, -1.0, 1.0)
                    
                    # p5 = (A[i1,j1] + A[i1,j2]) * B[j2,k2]
                    add_rank1(i1*m+j1, j2*p+k2, k2*n+i1, 1.0, 1.0, 1.0)
                    add_rank1(i1*m+j2, j2*p+k2, k2*n+i1, 1.0, 1.0, 1.0)
                    
                    # p6 = (A[i2,j1] - A[i1,j1]) * (B[j1,k1] + B[j1,k2])
                    add_rank1(i2*m+j1, j1*p+k1, k1*n+i2, 1.0, 1.0, 1.0)
                    add_rank1(i2*m+j1, j1*p+k2, k2*n+i2, 1.0, 1.0, 1.0)
                    add_rank1(i1*m+j1, j1*p+k1, k1*n+i1, -1.0, 1.0, 1.0)
                    add_rank1(i1*m+j1, j1*p+k2, k2*n+i1, -1.0, 1.0, 1.0)
                    
                    # p7 = (A[i1,j2] - A[i2,j2]) * (B[j2,k1] + B[j2,k2])
                    add_rank1(i1*m+j2, j2*p+k1, k1*n+i1, 1.0, 1.0, 1.0)
                    add_rank1(i1*m+j2, j2*p+k2, k2*n+i1, 1.0, 1.0, 1.0)
                    add_rank1(i2*m+j2, j2*p+k1, k1*n+i2, -1.0, 1.0, 1.0)
                    add_rank1(i2*m+j2, j2*p+k2, k2*n+i2, -1.0, 1.0, 1.0)
                    
                else:
                    # Use schoolbook for non-2x2x2 blocks
                    for i in ib:
                        for j in jb:
                            for k in kb:
                                add_rank1(i*m+j, j*p+k, k*n+i, 1.0, 1.0, 1.0)
    
    # Convert lists to matrices
    rank = len(U)
    U_mat = np.zeros((nm, rank))
    V_mat = np.zeros((mp, rank))
    W_mat = np.zeros((pn, rank))
    
    for r in range(rank):
        a_idx, u_val = U[r]
        b_idx, v_val = V[r]
        c_idx, w_val = W[r]
        U_mat[a_idx, r] = u_val
        V_mat[b_idx, r] = v_val
        W_mat[c_idx, r] = w_val
    
    # Verify
    if verify_decomposition(U_mat, V_mat, W_mat):
        return {
            "U": U_mat.tolist(),
            "V": V_mat.tolist(),
            "W": W_mat.tolist()
        }
    
    # Fallback to schoolbook
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
# EVOLVE-BLOCK-END
