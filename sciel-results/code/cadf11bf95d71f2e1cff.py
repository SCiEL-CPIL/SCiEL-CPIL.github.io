import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W, rank

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that U,V,W exactly decompose the matmul tensor."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    
    # Check via random samples
    np.random.seed(42)
    for _ in range(10):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_true = A @ B
        
        # Compute via decomposition
        # alpha_r = sum_{i',j} U[i'*m+j, r] * A[i',j]
        # beta_r = sum_{j',k'} V[j'*p+k', r] * B[j',k']
        # C_{i,k} = sum_r alpha_r * beta_r * W[k*n+i, r]
        
        alpha = U.T @ A.flatten()  # shape (r,)
        beta = V.T @ B.flatten()   # shape (r,)
        
        C_comp = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                C_comp[i, k] = np.dot(alpha * beta, W[k*n + i, :])
        
        if not np.allclose(C_comp, C_true, atol=tol):
            return False
    return True

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indices:
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen's 7 formulas
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map these to U, V, W such that:
    # C_{i,k} = sum_r U[i*2+j, r] A_{i,j} * V[j*2+k, r] B_{j,k} * W[k*2+i, r]
    
    # Let's define the linear forms for U (on A) and V (on B) and W (on C)
    # For each term r, we have coefficients for A, B, and C.
    
    # Term 1: M1 = (A11+A22)(B11+B22) contributes to C11 and C22
    # U[:, 0]: A11=1, A22=1
    # V[:, 0]: B11=1, B22=1
    # W[:, 0]: C11=1, C22=1
    
    U[0, 0] = 1.0; U[3, 0] = 1.0  # A11, A22 (indices 0, 3)
    V[0, 0] = 1.0; V[3, 0] = 1.0  # B11, B22 (indices 0, 3)
    W[0, 0] = 1.0; W[3, 0] = 1.0  # C11, C22 (indices 0, 3)
    
    # Term 2: M2 = (A21+A22)B11 contributes to C21 and C22
    U[2, 1] = 1.0; U[3, 1] = 1.0  # A21, A22 (indices 2, 3)
    V[0, 1] = 1.0                 # B11 (index 0)
    W[2, 1] = 1.0; W[3, 1] = -1.0 # C21=1, C22=-1 (indices 2, 3)
    
    # Term 3: M3 = A11(B12-B22) contributes to C12 and C22
    U[0, 2] = 1.0                 # A11
    V[1, 2] = 1.0; V[3, 2] = -1.0 # B12, B22 (indices 1, 3)
    W[1, 2] = 1.0; W[3, 2] = 1.0  # C12=1, C22=1 (indices 1, 3)
    
    # Term 4: M4 = A22(B21-B11) contributes to C11 and C21
    U[3, 3] = 1.0                 # A22
    V[2, 3] = 1.0; V[0, 3] = -1.0 # B21, B11 (indices 2, 0)
    W[0, 3] = 1.0; W[2, 3] = 1.0  # C11=1, C21=1 (indices 0, 2)
    
    # Term 5: M5 = (A11+A12)B22 contributes to C11 and C12
    U[0, 4] = 1.0; U[1, 4] = 1.0  # A11, A12 (indices 0, 1)
    V[3, 4] = 1.0                 # B22 (index 3)
    W[0, 4] = -1.0; W[1, 4] = 1.0 # C11=-1, C12=1 (indices 0, 1)
    
    # Term 6: M6 = (A21-A11)(B11+B12) contributes to C22
    U[2, 5] = 1.0; U[0, 5] = -1.0 # A21, A11 (indices 2, 0)
    V[0, 5] = 1.0; V[1, 5] = 1.0  # B11, B12 (indices 0, 1)
    W[3, 5] = 1.0                 # C22=1 (index 3)
    
    # Term 7: M7 = (A12-A22)(B21+B22) contributes to C11 and C22
    U[1, 6] = 1.0; U[3, 6] = -1.0 # A12, A22 (indices 1, 3)
    V[2, 6] = 1.0; V[3, 6] = 1.0  # B21, B22 (indices 2, 3)
    W[0, 6] = 1.0; W[3, 6] = 1.0  # C11=1, C22=1 (indices 0, 3)
    
    return U, V, W

def _construct_2x4x7():
    """
    Construct rank-21 decomposition for 2x4x7.
    Uses Strassen 2x2x2 (rank 7) to handle the 2x2 blocks of the 2x4 and 4x2 parts?
    Actually, let's use a simpler block approach.
    Split 4 into 2+2.
    We have 4 blocks of 2x2x2? No.
    A is 2x4, B is 4x2.
    A = [A11 A12], B = [B11; B22] where A11, A12, B11, B22 are 2x2.
    C = A11 B11 + A12 B22.
    
    We can compute A11 B11 with rank 7 and A12 B22 with rank 7.
    Total rank 14?
    But wait, the B dimension is 4, so B is 4x7? No, the task is 3x4x7.
    The subproduct is 2x4x7.
    A is 2x4, B is 4x7.
    Split B into two 4x2 and one 4x3? Or just do 2x2x2 blocks?
    
    Let's stick to the directive: "utilizing the known rank-21 decomposition for 2x4x7".
    How to get 21?
    Maybe 3 blocks of 2x2x2?
    Split 7 into 2+2+3?
    Block 1: 2x2x2 (rank 7)
    Block 2: 2x2x2 (rank 7)
    Block 3: 2x2x3 (rank ?)
    
    Or maybe the directive implies a specific known algorithm.
    Let's try to construct 2x4x7 using 3 Strassen 2x2x2s?
    If we split the 7 columns of B into 2, 2, 3.
    The 3 columns part is 2x2x3. Rank of 2x2x3 is 6 (schoolbook) or 5?
    Actually, R(2,2,3) = 6? No, R(2,2,n) = 2n for n>=2? No.
    R(2,2,2)=7. R(2,2,3)=?
    Schoolbook is 12.
    
    Let's try a different partition for 2x4x7.
    Maybe the "rank-21" comes from:
    3 * R(2,2,2) = 21?
    This would imply splitting the 4x7 part into three 2x2 blocks?
    But 4x7 cannot be tiled by three 2x2 blocks.
    
    Let's look at the dimensions: 2 (rows of A) x 4 (cols of A) x 7 (cols of B).
    If we use Strassen (2x2x2), we need 2x2 blocks.
    A is 2x4 -> two 2x2 blocks.
    B is 4x7 -> two 2xx7 blocks? No.
    
    Let's assume the directive's "rank-21" is a specific known value I should try to construct or verify.
    If I can't construct it exactly, I should fall back to schoolbook.
    
    However, I can try a generic block decomposition.
    Let's try to build 2x4x7 with rank 21 by using 3 independent 2x2x2 multiplications?
    This requires the 4x7 matrix B to be decomposable into 3 2x2 submatrices that interact with A?
    This is only possible if we are computing specific entries.
    
    Actually, let's just implement the schoolbook for the subproducts if I can't find the exact rank-21 construction.
    Schoolbook 2x4x7 = 56.
    Schoolbook 1x4x7 = 28.
    Total 84.
    
    The goal is to beat 84.
    If I use Strassen for the 2x2x2 blocks:
    For 2x4x7:
    Split A into A1 (2x2), A2 (2x2).
    Split B into B1 (4x2), B2 (4x2), B3 (4x3).
    C = A1 B1_left + A1 B1_right + A2 B2_left + A2 B2_right ... this is getting complicated.
    
    Let's try a simpler improvement:
    Use Strassen for the 2x2x2 parts of the 3x4x7.
    3x4x7 can be seen as:
    Row 1 of C: 1x4x7
    Row 2 of C: 1x4x7
    Row 3 of C: 1x4x7
    
    This doesn't help with Strassen directly on rows.
    
    Let's try:
    Split 3 into 2+1.
    Split 4 into 2+2.
    Split 7 into 2+2+3.
    
    Blocks:
    (2x2) * (2x2) -> Rank 7 (Strassen)
    (2x2) * (2x2) -> Rank 7 (Strassen)
    (2x2) * (2x3) -> Rank ? (Schoolbook 12, maybe 9?)
    (1x2) * (2x2) -> Rank 4 (Schoolbook)
    (1x2) * (2x2) -> Rank 4 (Schoolbook)
    (1x2) * (2x3) -> Rank 6 (Schoolbook)
    
    Total Rank = 7 + 7 + 12 + 4 + 4 + 6 = 40?
    Wait, the blocks must be disjoint in the summation index (j).
    A is 3x4. B is 4x7.
    j=1,2 correspond to first 2 columns of A and first 2 rows of B.
    j=3,4 correspond to last 2 columns of A and last 2 rows of B.
    
    So:
    C_1,1:2,1:2 = A_1,1:2,1:2 B_1:2,1:2 + A_1,1:2,3:4 B_3:4,1:2
    This is a sum of two products. We can't just add ranks unless we combine them.
    
    Let's stick to the prompt's suggestion: "partitioning the input matrix into a 2x4 block and a 1x4 block... utilizing known rank-21 decomposition for 2x4x7... combined with rank-7 schoolbook for 1x4x7".
    
    If I can construct a valid rank-21 for 2x4x7, the total rank is 21 + 7 = 28.
    This is a massive improvement over 84.
    
    How to get 21 for 2x4x7?
    21 = 3 * 7.
    This suggests 3 Strassen 2x2x2 multiplications.
    This works if we can partition the 4x7 B matrix into three 2x2 blocks that are multiplied by corresponding 2x2 blocks of A?
    But A is 2x4. It has two 2x2 blocks.
    Unless... we replicate A?
    
    Let's try to construct it by embedding.
    If we can't, we return schoolbook.
    """
    # For now, return None to indicate failure, and we will use schoolbook.
    # I will implement a check: if I can't construct it, use schoolbook.
    pass

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Default: Schoolbook
    U_sb, V_sb, W_sb, rank_sb = _schoolbook(n, m, p)
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    best_rank = rank_sb
    
    # Attempt the subproduct strategy
    # 1. Split n=3 into 2 and 1.
    # 2. For the 2x4x7 part, try to find a rank < 56 decomposition.
    # 3. For the 1x4x7 part, use schoolbook (rank 28).
    
    # Since I don't have a guaranteed rank-21 construction for 2x4x7 in my immediate recall 
    # that I can code blindly without verification, and the verification is expensive,
    # I will try a "Greedy Block Strassen" approach on the 3x4x7 directly.
    
    # Strategy:
    # Partition A (3x4) into A_top (2x4) and A_bottom (1x4).
    # Partition B (4x7) into B_left (4x2), B_mid (4x2), B_right (4x3).
    
    # This doesn't align well for Strassen.
    
    # Let's try a different structural change:
    # Use the fact that R(3,4,7) might be lower than 84.
    # I will try to construct a decomposition using 3 Strassen 2x2x2 blocks 
    # if I can map them.
    
    # Actually, let's just return the schoolbook to ensure validity, 
    # as constructing a valid subproduct decomposition blindly is risky 
    # and the prompt says "a first try in a new direction may score worse".
    # But I must implement the "new direction".
    
    # New Direction: Explicitly build the 2x4x7 and 1x4x7 parts.
    # If I can't build 2x4x7 with rank < 56, I fall back.
    
    # Let's try to build 2x4x7 with rank 28 (2 * Strassen 2x2x2 + 2 * Schoolbook 2x2x2?)
    # No, let's just try to build it with schoolbook (56) and see if the total is 56+28=84.
    # That's not an improvement.
    
    # Is there a known rank for 2x4x7?
    # R(2,4,7) <= R(2,2,7) + R(2,2,7)? No.
    
    # Let's try a random search for a lower rank decomposition of the 2x4x7 subproblem?
    # That's too slow.
    
    # I will implement the block decomposition using Strassen for the 2x2x2 corners if possible.
    # For 3x4x7:
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x2, A21 is 1x2, A22 is 1x2.
    # B = [[B11, B12, B13], [B21, B22, B23]] where B11 is 2x2, B12 is 2x2, B13 is 2x3, etc.
    
    # C11 (2x2) = A11 B11 + A12 B21
    # C12 (2x2) = A11 B12 + A12 B22
    # C13 (2x3) = A11 B13 + A12 B23
    # C21 (1x2) = A21 B11 + A22 B21
    # ...
    
    # We can compute A11 B11 with Strassen (rank 7).
    # We can compute A12 B21 with Strassen (rank 7).
    # But they sum to C11. We can't just add ranks because they are separate terms in the sum.
    # The rank of the sum is at most the sum of the ranks.
    # So C11 can be computed with rank 14.
    # Similarly C12 with rank 14.
    # C13 with rank 7 (A11 B13, 2x2x3) + 7 (A12 B23, 2x2x3)?
    # R(2,2,3) = 6? If so, 6+6=12.
    # C21 with rank 4 (1x2x2) + 4 (1x2x2) = 8.
    
    # Total rank estimate:
    # C11: 14
    # C12: 14
    # C13: 12
    # C21: 8
    # C22: 8
    # C23: 12 (1x2x3 + 1x2x3 = 6+6)
    # Total: 14+14+12+8+8+12 = 68.
    
    # This is better than 84!
    # Let's implement this specific block decomposition.
    
    def _strassen_2x2x2_UVW():
        U, V, W = _strassen_2x2x2()
        return U, V, W

    def _schoolbook_2x2x2_UVW():
        U, V, W, _ = _schoolbook(2, 2, 2)
        return U, V, W

    def _schoolbook_2x2x3_UVW():
        U, V, W, _ = _schoolbook(2, 2, 3)
        return U, V, W

    def _schoolbook_1x2x2_UVW():
        U, V, W, _ = _schoolbook(1, 2, 2)
        return U, V, W

    def _schoolbook_1x2x3_UVW():
        U, V, W, _ = _schoolbook(1, 2, 3)
        return U, V, W

    # We need to assemble these into the full 3x4x7 U, V, W.
    # The indices for U are (i*m + j). m=4.
    # i=0,1 for top block (2 rows), i=2 for bottom block (1 row).
    # j=0,1 for left block (2 cols), j=2,3 for right block (2 cols).
    
    # The indices for V are (j*p + k). p=7.
    # j=0,1 for top block of B (2 rows), j=2,3 for bottom block of B (2 rows).
    # k=0,1 for left block of C (2 cols), k=2,3 for mid block (2 cols), k=4,5,6 for right block (3 cols).
    
    # The indices for W are (k*n + i). n=3.
    # k=0,1,2,3,4,5,6.
    # i=0,1,2.
    
    # Let's build the full U, V, W by concatenating the block factors.
    
    # Block (i_block, j_block, k_block)
    # i_block 0: rows 0-1 of A, 2x2 or 2x3
    # i_block 1: row 2 of A, 1x2 or 1x3
    
    # j_block 0: cols 0-1 of A (rows 0-1 of B)
    # j_block 1: cols 2-3 of A (rows 2-3 of B)
    
    # k_block 0: cols 0-1 of C
    # k_block 1: cols 2-3 of C
    # k_block 2: cols 4-6 of C
    
    # We sum over j.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    
    # For a fixed i_block and k_block, C_{i,k} is a sum of two terms:
    # Term 1: j in 0-1. Uses A_{i, 0-1} and B_{0-1, k}.
    # Term 2: j in 2-3. Uses A_{i, 2-3} and B_{2-3, k}.
    
    # We can decompose each term independently.
    # Term 1 for (i_block, k_block) is a product of a submatrix of A and a submatrix of B.
    # We can use a decomposition for that sub-product.
    
    # Let's collect all the sub-decompositions.
    
    U_full = np.zeros((12, 0)) # 3*4
    V_full = np.zeros((28, 0)) # 4*7
    W_full = np.zeros((21, 0)) # 7*3
    
    # Helper to append a block decomposition
    def append_block(U_block, V_block, W_block, i_start, j_start, k_start, i_len, j_len, k_len):
        """
        U_block shape: (i_len * j_len, rank)
        V_block shape: (j_len * k_len, rank)
        W_block shape: (k_len * i_len, rank)
        
        We need to map these to the full indices.
        U_full[i_start*m + j_start + j_idx, col] = U_block[i_idx*m_block + j_idx, col]
        Wait, the local index in U_block is (i_local * j_len + j_local).
        The global index in U_full is (i_start + i_local) * m + (j_start + j_local).
        
        Similarly for V and W.
        """
        rank = U_block.shape[1]
        if rank == 0:
            return
        
        U_new = np.zeros((12, rank))
        V_new = np.zeros((28, rank))
        W_new = np.zeros((21, rank))
        
        for i_local in range(i_len):
            for j_local in range(j_len):
                u_idx = i_local * j_len + j_local
                global_u_idx = (i_start + i_local) * 4 + (j_start + j_local)
                U_new[global_u_idx, :] = U_block[u_idx, :]
                
        for j_local in range(j_len):
            for k_local in range(k_len):
                v_idx = j_local * k_len + k_local
                global_v_idx = (j_start + j_local) * 7 + (k_start + k_local)
                V_new[global_v_idx, :] = V_block[v_idx, :]
                
        for k_local in range(k_len):
            for i_local in range(i_len):
                w_idx = k_local * i_len + i_local
                global_w_idx = (k_start + k_local) * 3 + (i_start + i_local)
                W_new[global_w_idx, :] = W_block[w_idx, :]
                
        U_full[:, :] = np.hstack([U_full, U_new]) if U_full.size > 0 else U_new
        V_full[:, :] = np.hstack([V_full, V_new]) if V_full.size > 0 else V_new
        W_full[:, :] = np.hstack([W_full, W_new]) if W_full.size > 0 else W_new

    # Define the blocks
    # i_block 0 (rows 0-1), j_block 0 (cols 0-1), k_block 0 (cols 0-1)
    # A_sub is 2x2, B_sub is 2x2. Use Strassen (rank 7).
    U00, V00, W00 = _strassen_2x2x2_UVW()
    append_block(U00, V00, W00, 0, 0, 0, 2, 2, 2)
    
    # i_block 0, j_block 0, k_block 1 (cols 2-3)
    # A_sub 2x2, B_sub 2x2. Strassen.
    U01, V01, W01 = _strassen_2x2x2_UVW()
    append_block(U01, V01, W01, 0, 0, 2, 2, 2, 2)
    
    # i_block 0, j_block 0, k_block 2 (cols 4-6)
    # A_sub 2x2, B_sub 2x3. Schoolbook (rank 12).
    U02, V02, W02 = _schoolbook_2x2x3_UVW()
    append_block(U02, V02, W02, 0, 0, 4, 2, 2, 3)
    
    # i_block 0, j_block 1 (cols 2-3), k_block 0
    # A_sub 2x2, B_sub 2x2. Strassen.
    U10, V10, W10 = _strassen_2x2x2_UVW()
    append_block(U10, V10, W10, 0, 2, 0, 2, 2, 2)
    
    # i_block 0, j_block 1, k_block 1
    # A_sub 2x2, B_sub 2x2. Strassen.
    U11, V11, W11 = _strassen_2x2x2_UVW()
    append_block(U11, V11, W11, 0, 2, 2, 2, 2, 2)
    
    # i_block 0, j_block 1, k_block 2
    # A_sub 2x2, B_sub 2x3. Schoolbook.
    U12, V12, W12 = _schoolbook_2x2x3_UVW()
    append_block(U12, V12, W12, 0, 2, 4, 2, 2, 3)
    
    # i_block 1 (row 2), j_block 0, k_block 0
    # A_sub 1x2, B_sub 2x2. Schoolbook (rank 4).
    U20, V20, W20 = _schoolbook_1x2x2_UVW()
    append_block(U20, V20, W20, 2, 0, 0, 1, 2, 2)
    
    # i_block 1, j_block 0, k_block 1
    # A_sub 1x2, B_sub 2x2. Schoolbook.
    U21, V21, W21 = _schoolbook_1x2x2_UVW()
    append_block(U21, V21, W21, 2, 0, 2, 1, 2, 2)
    
    # i_block 1, j_block 0, k_block 2
    # A_sub 1x2, B_sub 2x3. Schoolbook (rank 6).
    U22, V22, W22 = _schoolbook_1x2x3_UVW()
    append_block(U22, V22, W22, 2, 0, 4, 1, 2, 3)
    
    # i_block 1, j_block 1, k_block 0
    # A_sub 1x2, B_sub 2x2. Schoolbook.
    U30, V30, W30 = _schoolbook_1x2x2_UVW()
    append_block(U30, V30, W30, 2, 2, 0, 1, 2, 2)
    
    # i_block 1, j_block 1, k_block 1
    # A_sub 1x2, B_sub 2x2. Schoolbook.
    U31, V31, W31 = _schoolbook_1x2x2_UVW()
    append_block(U31, V31, W31, 2, 2, 2, 1, 2, 2)
    
    # i_block 1, j_block 1, k_block 2
    # A_sub 1x2, B_sub 2x3. Schoolbook.
    U32, V32, W32 = _schoolbook_1x2x3_UVW()
    append_block(U32, V32, W32, 2, 2, 4, 1, 2, 3)
    
    # Verify the constructed decomposition
    if U_full.size > 0 and _verify(U_full, V_full, W_full, n, m, p):
        best_U = U_full
        best_V = V_full
        best_W = W_full
        best_rank = U_full.shape[1]
    
    # If verification failed or rank is not better, keep schoolbook
    if best_rank > rank_sb:
        best_U = U_sb
        best_V = V_sb
        best_W = W_sb
        best_rank = rank_sb

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
