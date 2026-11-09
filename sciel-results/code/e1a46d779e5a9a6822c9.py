import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def _strassen_2x2():
    """Return (U, V, W) each 4x7 for 2x2 matrix multiplication."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # M1 = (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    # M2 = (A10+A11)B00
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1; W[3,1]=-1
    # M3 = A00(B10-B11)
    U[0,2]=1
    V[2,2]=1; V[3,2]=-1
    W[1,2]=1; W[3,2]=1
    # M4 = A11(B01-B00)
    U[3,3]=1
    V[1,3]=1; V[0,3]=-1
    W[0,3]=1; W[2,3]=1
    # M5 = (A00+A01)B11
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=-1; W[1,4]=1
    # M6 = (A10-A00)(B00+B01)
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[3,5]=1
    # M7 = (A01-A11)(B10+B11)
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,6]=1
    return U, V, W

def _strassen_4x4x4():
    """
    Build Strassen decomposition for 4x4x4 using block structure.
    A is 4x4, B is 4x4, C is 4x4.
    Split into 2x2 blocks of size 2x2.
    Uses 7 Strassen products, each of which is a 2x2x2 multiplication (rank 7).
    Total rank: 7*7 = 49.
    """
    U2, V2, W2 = _strassen_2x2()
    
    # Block indices: bi, bj in {0,1} for A blocks (row, col)
    # A blocks: A[bi*2:(bi+1)*2, bj*2:(bj+1)*2]
    # B blocks: B[bj*2:(bj+1)*2, bk*2:(bk+1)*2]
    # C blocks: C[bi*2:(bi+1)*2, bk*2:(bk+1)*2]
    
    # Strassen formula for 2x2 block matrices:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M5 + M6
    # C10 = M2 + M3
    # C11 = M1 - M2 + M3 + M4
    
    # For each Strassen step s (0..6), we need to compute a product of two 2x2 "super-matrices"
    # where each entry is a 2x2 block.
    # The product of two 2x2 matrices of 2x2 blocks is itself a 2x2 matrix of 2x2 blocks.
    # Each such product has rank 7 (via Strassen 2x2x2).
    
    # Let's define the Strassen steps for the block level:
    # Step s: U_s (4x7 for block-level A), V_s (4x7 for block-level B), W_s (4x7 for block-level C)
    # But at the block level, "4" means 4 blocks (2x2 arrangement).
    
    # Actually, U2, V2, W2 are for 2x2 matrix mult, with indices 0..3 for the 4 elements.
    # At the block level, each "element" is a 2x2 block.
    # So for Strassen step s, the block-level U has 4 entries (for the 4 block positions).
    # Each block position corresponds to a 2x2 submatrix.
    
    # Let's build the full 4x4x4 decomposition.
    # U_full: 16 x 49 (4x4 A matrix flattened to 16)
    # V_full: 16 x 49 (4x4 B matrix flattened to 16)
    # W_full: 16 x 49 (4x4 C matrix flattened to 16)
    
    U_full = np.zeros((16, 49))
    V_full = np.zeros((16, 49))
    W_full = np.zeros((16, 49))
    
    # For each Strassen step s (0..6), and for each 2x2x2 Strassen sub-step t (0..6):
    # The column index is s*7 + t
    
    # The block-level Strassen defines, for step s:
    #   u_s[bi, bj] for the A block at position (bi, bj)
    #   v_s[bj, bk] for the B block at position (bj, bk)
    #   w_s[bi, bk] for the C block at position (bi, bk)
    # where bi, bj, bk in {0, 1}
    
    # But U2, V2, W2 give us the coefficients for the 2x2 matrix multiplication.
    # U2[i, s] where i is the element index in the 2x2 A matrix.
    # Element index i = 2*bi + bj for A (row bi, col bj)
    # Element index j = 2*bj + bk for B (row bj, col bk)
    # Element index k = 2*bi + bk for C (row bi, col bk)
    
    # Wait, I need to be careful with the indexing.
    # For a 2x2 matrix, element (r,c) has index 2*r + c.
    # A: element (bi, bj) -> index 2*bi + bj
    # B: element (bj, bk) -> index 2*bj + bk
    # C: element (bi, bk) -> index 2*bi + bk
    
    # So for Strassen step s, the block-level coefficients are:
    # u_block[bi, bj] = U2[2*bi + bj, s]
    # v_block[bj, bk] = V2[2*bj + bk, s]
    # w_block[bi, bk] = W2[2*bi + bk, s]
    
    # Now, for each block position, we need to compute a 2x2 matrix product using Strassen.
    # The 2x2 matrix product has rank 7 (using U2, V2, W2 again).
    
    # So for Strassen step s and sub-step t:
    #   Column = s*7 + t
    #   For each A block position (bi, bj):
    #     u_block_coeff = U2[2*bi + bj, s]
    #     If u_block_coeff != 0:
    #       For each element (ai, aj) in the 2x2 block at (bi, bj):
    #         Global A index = 2*(2*bi + ai) + (2*bj + aj) = 4*bi + 2*ai + 2*bj + aj
    #         U_full[global_a_idx, col] += u_block_coeff * U2[2*ai + aj, t]
    
    #   Similarly for B and C.
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            # A part
            for bi in range(2):
                for bj in range(2):
                    u_block_coeff = U2[2*bi + bj, s]
                    if u_block_coeff != 0:
                        for ai in range(2):
                            for aj in range(2):
                                global_a_idx = 4*bi + 2*ai + 2*bj + aj
                                U_full[global_a_idx, col] += u_block_coeff * U2[2*ai + aj, t]
            # B part
            for bj in range(2):
                for bk in range(2):
                    v_block_coeff = V2[2*bj + bk, s]
                    if v_block_coeff != 0:
                        for bi in range(2):
                            for bk_inner in range(2):
                                global_b_idx = 4*bj + 2*bi + 2*bk + bk_inner
                                V_full[global_b_idx, col] += v_block_coeff * V2[2*bi + bk_inner, t]
            # C part
            for bi in range(2):
                for bk in range(2):
                    w_block_coeff = W2[2*bi + bk, s]
                    if w_block_coeff != 0:
                        for ai in range(2):
                            for ak in range(2):
                                global_c_idx = 4*bi + 2*ai + 2*bk + ak
                                W_full[global_c_idx, col] += w_block_coeff * W2[2*ai + ak, t]
    
    # Verify
    # The tensor T[i,j,k] = delta_{i,j,k} for matrix mult
    # Our decomposition: sum_s sum_t U[i, s*7+t] * V[j, s*7+t] * W[k, s*7+t]
    # should equal 1 if i,j,k correspond to a valid matrix mult term, 0 otherwise.
    
    return U_full, V_full, W_full

def _schoolbook_4x4x3():
    """Schoolbook decomposition for 4x4x3. Rank = 48."""
    U = np.zeros((16, 48))
    V = np.zeros((28, 48))
    W = np.zeros((28, 48))
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U[i*4 + j, col] = 1
                V[j*4 + k, col] = 1
                W[k*4 + i, col] = 1
                col += 1
    return U, V, W

def _combine_decompositions(U1, V1, W1, U2, V2, W2):
    """Combine two decompositions by concatenating columns."""
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    U = np.hstack([U1, U2])
    V = np.hstack([V1, V2])
    W = np.hstack([W1, W2])
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    # The tensor for matrix mult: T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[k,i]
    # In our convention: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # The reconstruction: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    # for all i,j,k where the term exists (always, for matrix mult), and 0 for invalid combos.
    # Actually, for matrix mult, every combination (i,j,k) is valid in the sense that
    # A[i,j]*B[j,k] contributes to C[k,i]. So the tensor is T[i,j,k] = 1 for all i,j,k.
    # Wait, that's not right. The tensor T is defined such that the bilinear form is:
    # sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * C_out[k,i]
    # For matrix mult, T[i,j,k] = 1 for all i,j,k.
    
    # Let's verify by computing the tensor and checking.
    # Tensor shape: (n*m, m*p, p*n)
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[a, b, c] = 1 if the indices are consistent, 0 otherwise.
    # Actually, for matrix multiplication, the tensor is:
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # And 0 otherwise.
    
    # Let's check our decomposition:
    # sum_r U[a, r] * V[b, r] * W[c, r] should equal T[a, b, c]
    
    # Build the true tensor (sparse)
    # For efficiency, let's just check a subset or use the structure.
    
    # Actually, let's verify by matrix multiplication:
    # For random A, B, check that sum_r (A_flat @ U[:,r]) * (B_flat @ V[:,r]) * (W[:,r] @ C_flat) 
    # No, that's not how it works.
    
    # The correct verification:
    # Let A be n x m, B be m x p.
    # C = A @ B is n x p.
    # Our decomposition computes: sum_r (A_flat @ U[:,r]) * (B_flat @ V[:,r]) * W[:,r]
    # where A_flat is n*m vector, B_flat is m*p vector.
    # This should equal C_flat (p*n vector) in the trace convention.
    
    # Let's test with random matrices.
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C = A @ B
    
    A_flat = A.flatten()  # length n*m
    B_flat = B.flatten()  # length m*p
    C_flat = C.flatten()  # length n*p, but we need p*n convention
    
    # Trace convention: W[k*n+i, r] means C[i,k] is at index k*n+i
    # So C_flat_trace[k*n+i] = C[i,k]
    C_trace = np.zeros(p*n)
    for i in range(n):
        for k in range(p):
            C_trace[k*n + i] = C[i,k]
    
    # Compute using decomposition
    U_A = U.T @ A_flat  # length rank
    V_B = V.T @ B_flat  # length rank
    result = (U_A * V_B) @ W  # length p*n
    
    diff = np.abs(result - C_trace).max()
    return diff < tol

def _try_reduce_rank(U, V, W, target_rank, max_iters=100, tol=1e-9):
    """
    Try to reduce the rank of the decomposition using alternating least squares.
    This is a heuristic and may not find a valid low-rank decomposition.
    """
    # This is complex and may not work well. For now, return the input unchanged.
    return U, V, W

def construct():
    start_time = time.time()
    deadline = 140.0  # seconds
    
    n, m, p = DIMENSIONS
    
    # Step 1: Get Strassen 4x4x4 (rank 49)
    U_s, V_s, W_s = _strassen_4x4x4()
    
    # Step 2: Get schoolbook 4x4x3 (rank 48)
    U_sb, V_sb, W_sb = _schoolbook_4x4x3()
    
    # Step 3: Combine
    U, V, W = _combine_decompositions(U_s, V_s, W_s, U_sb, V_sb, W_sb)
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook for the whole thing
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    current_rank = U.shape[1]
    
    # Step 4: Try to reduce rank if time permits
    # For now, just return the combined decomposition
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

