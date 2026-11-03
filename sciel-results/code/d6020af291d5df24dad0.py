import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _schoolbook(n, m, p):
    """Schoolbook decomposition for n x m times m x p -> n x p.
    Returns U (n*m x n*m*p), V (m*p x n*m*p), W (p*n x n*m*p) as lists.
    Convention: U indexed by (i,j) row-major, V by (j,k), W by (k,i).
    C[i,i_out] = sum_{j,k} A[i,j]*B[j,k]
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """Strassen 2x2x2 decomposition, rank 7.
    A is 2x2, B is 2x2, C = A*B is 2x2.
    U indexed by (i,j) for A, V by (j,k) for B, W by (k,i) for C (trace convention).
    """
    # Strassen formulas:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    #
    # U[i*m+j] for A element (i,j), V[j*p+k] for B element (j,k), W[k*n+i] for C element (k,i)
    # n=m=p=2, so U has 4 rows, V has 4 rows, W has 4 rows.
    # U row indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # V row indices: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    # W row indices: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3  (W[k*n+i]: k=0,i=0 -> 0; k=0,i=1 -> 1; k=1,i=0 -> 2; k=1,i=1 -> 3)
    
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1 = (A11+A22)(B11+B22)
    # A indices: A11=(0,0)=row0, A22=(1,1)=row3
    # B indices: B11=(0,0)=row0, B22=(1,1)=row3
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1
    # C11 += M1, C12 += M1, C21 -= M1, C22 += M1
    # W[k*n+i]: C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[2], C[1,1]=W[3]
    W[0][0] = 1; W[1][0] = 1; W[2][0] = -1; W[3][0] = 1
    
    # M2 = (A21+A22)B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    # C21 += M2, C22 -= M2
    
    # M3 = A11(B12-B22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    # C12 += M3, C22 += M3
    
    # M4 = A22(B21-B11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    # C11 += M4, C21 += M4
    
    # M5 = (A11+A12)B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    # C11 -= M5, C12 += M5
    
    # M6 = (A21-A11)(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    # C22 += M6
    
    # M7 = (A12-A22)(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    # C11 += M7
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify tensor decomposition: sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] == delta(i,i) for all i,j,k."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    S = np.einsum('ijt,jkt,kit->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    # S should be 1 where i==i (always true since i is the same index), 0 otherwise
    # Actually S[i,j,k] should be 1 for all i,j,k since the matmul tensor T[i,j,k] = delta_{i,i} = 1
    # Wait: T[i,j,k] = 1 if the (i,k) entry of A*B uses A[i,j]*B[j,k], so T[i,j,k]=1 for all i,j,k
    # No! The tensor is T[i,j,k] = 1 for all valid i,j,k. It's the identity in the sense that
    # C[i,k] = sum_j A[i,j] B[j,k]. The tensor has T[i,j,k]=1 for all i,j,k.
    
    if not np.allclose(S, 1.0, atol=1e-9):
        return False
    return True

def _kron_decomp(U1, V1, W1, U2, V2, W2):
    """Kronecker product of two decompositions.
    If decomp1 is for (n1,m1,p1) and decomp2 is for (n2,m2,p2),
    the product is for (n1*n2, m1*m2, p1*p2).
    U row index: (i1*n2+i2)*m1*m2 + (j1*m2+j2)
    V row index: (j1*m2+j2)*p1*p2 + (k1*p2+k2)  
    W row index: (k1*p2+k2)*n1*n2 + (i1*n2+i2)
    """
    r1 = len(U1[0])
    r2 = len(U2[0])
    rank = r1 * r2
    n1, m1, p1 = len(U1) // m1, m1, len(V1) // m1  # infer from shapes
    # Actually let's pass dimensions explicitly
    return None  # We'll handle this differently

def _combine_blocks(block_decomps, n, m, p, block_structure):
    """
    Combine block decompositions into a full 4x4x9 decomposition.
    
    block_structure: list of (n_b, m_b, p_b, U_b, V_b, W_b) for each block.
    The blocks tile the 4x4x9 space.
    
    For a block decomposition of size (n_b, m_b, p_b), the U_b has shape (n_b*m_b, r_b),
    V_b has shape (m_b*p_b, r_b), W_b has shape (p_b*n_b, r_b).
    
    We need to embed each block into the full 4x4x9 space.
    Block (a, b, c) occupies:
    - A rows: a*n_b to (a+1)*n_b - 1, A cols: b*m_b to (b+1)*m_b - 1
    - B rows: b*m_b to (b+1)*m_b - 1, B cols: c*p_b to (c+1)*p_b - 1  
    - C rows: c*p_b to (c+1)*p_b - 1, C cols: a*n_b to (a+1)*n_b - 1
    
    Full U index: (i_full * m + j_full) where i_full = a*n_b + i_b, j_full = b*m_b + j_b
    Full V index: (j_full * p + k_full) where j_full = b*m_b + j_b, k_full = c*p_b + k_b
    Full W index: (k_full * n + i_full) where k_full = c*p_b + k_b, i_full = a*n_b + i_b
    """
    total_rank = sum(len(bd[3][0]) for bd in block_decomps)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for (a, b, c, n_b, m_b, p_b, U_b, V_b, W_b) in block_decomps:
        r_b = len(U_b[0])
        for i_b in range(n_b):
            for j_b in range(m_b):
                u_row = (a * n_b + i_b) * m + (b * m_b + j_b)
                for t in range(r_b):
                    U[u_row][col_offset + t] = U_b[i_b * m_b + j_b][t]
        for j_b in range(m_b):
            for k_b in range(p_b):
                v_row = (b * m_b + j_b) * p + (c * p_b + k_b)
                for t in range(r_b):
                    V[v_row][col_offset + t] = V_b[j_b * p_b + k_b][t]
        for k_b in range(p_b):
            for i_b in range(n_b):
                w_row = (c * p_b + k_b) * n + (a * n_b + i_b)
                for t in range(r_b):
                    W[w_row][col_offset + t] = W_b[k_b * n_b + i_b][t]
        col_offset += r_b
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    start_time = time.time()
    deadline = start_time + 120
    
    # Build base decompositions
    sb_2x2x1 = _schoolbook(2, 2, 1)  # rank 4
    sb_2x2x2 = _schoolbook(2, 2, 2)  # rank 8
    st_2x2x2 = _strassen_2x2x2()     # rank 7
    
    # Verify base decompositions
    assert _verify(sb_2x2x1[0], sb_2x2x1[1], sb_2x2x1[2], 2, 2, 1), "sb_2x2x1 failed"
    assert _verify(sb_2x2x2[0], sb_2x2x2[1], sb_2x2x2[2], 2, 2, 2), "sb_2x2x2 failed"
    assert _verify(st_2x2x2[0], st_2x2x2[1], st_2x2x2[2], 2, 2, 2), "strassen failed"
    
    # Strategy: Partition 4x4x9 into blocks and use best decomposition for each.
    # The 4x4 A matrix and 4x9 B matrix.
    # We can partition rows of A (4 rows) and columns of B (9 cols).
    # Block structure: A is partitioned into row-blocks, B into col-blocks.
    # For each pair (row_block_of_A, col_block_of_B), we compute a sub-product.
    # But we also need to partition the intermediate dimension (4).
    # 
    # Simplest: partition the 4x4 A into 2x2 blocks (2 row-blocks x 2 col-blocks),
    # and the 4x9 B into row-blocks matching A's col-blocks and col-blocks.
    #
    # Let's use: A partitioned as 2+2 rows, 2+2 cols. B partitioned as 2+2 rows, 9 cols.
    # Actually B is 4x9, so 4 rows = 2+2, 9 cols.
    # 
    # Block (a,b,c): A block is rows [a*2:(a+1)*2], cols [b*2:(b+1)*2] -> 2x2
    # B block is rows [b*2:(b+1)*2], cols [c:...] 
    # 
    # Let's partition 9 cols as 3+3+3.
    # So blocks are 2x2 x 2x3 sub-problems.
    # Number of blocks: 2 (A rows) * 2 (A cols/B rows) * 3 (B cols) = 12 blocks
    # Each block: 2x2 A, 2x3 B, 2x3 C. Rank of 2x2x2x3 = ?
    # 
    # For 2x2 * 2x3 -> 2x3:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Schoolbook rank = 2*2*3 = 12.
    # Can we do better? 
    # Actually, let's just use schoolbook for 2x2x3 (rank 12) for each block.
    # Total rank = 12 * 12 = 144. That's the same as full schoolbook.
    #
    # Better: use Strassen-like for the 2x2 part.
    # 2x2x3: We can write C = A*B where A is 2x2, B is 2x3.
    # C[:, 0:3] = A * B[:, 0:3]
    # This is like two 2x2x1 problems sharing the same A.
    # If we use Strassen for the 2x2 structure:
    # M1 = (A11+A22)(B11+B21+B31) -- no, that doesn't work cleanly.
    #
    # Let me think differently. For 2x2 * 2x3:
    # We can compute:
    # P1 = (A11+A22) * B1  (where B1 is first row of B, a 1x3 vector) -- 3 products
    # P2 = (A21+A22) * B1  -- 3 products  
    # P3 = A11 * (B2-B1)  -- 3 products
    # P4 = A22 * (B2-B1)  -- 3 products
    # P5 = (A11+A12) * B2  -- 3 products
    # P6 = (A21-A11) * B1  -- 3 products
    # P7 = (A12-A22) * B2  -- 3 products
    # Total: 21 products. Worse than 12.
    #
    # Actually the optimal for 2x2x3 is 12 (schoolbook) since it's just two independent
    # 2x2x1 computations. Wait no, they share A.
    # C[:,j] = A * B[:,j] for j=0,1,2.
    # Each C[:,j] is a 2-vector computed from A (2x2) and B[:,j] (2-vector).
    # C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j]
    # The products are: A[i,a]*B[a,j] for i in {0,1}, a in {0,1}, j in {0,1,2}
    # That's 2*2*3 = 12 distinct products, and each output C[i,j] is a sum of 2 products.
    # So rank is exactly 12 (you can't share products across different j since B[a,j] differs).
    # Actually wait - you CAN share. A[i,0]*B[0,j] for different j are different products.
    # So yes, rank 12 for 2x2x3.
    #
    # Alternative: partition 4x4x9 differently.
    # Use 4x4x4 (Strassen rank 49) + 4x4x1 (schoolbook rank 16) + 4x4x4 (Strassen rank 49)
    # Total: 49 + 16 + 49 = 114.
    # Or: 4x4x3 (rank?) + 4x4x3 (rank?) + 4x4x3 (rank?)
    #
    # For 4x4x3: partition A as 2+2 rows, 2+2 cols; B as 2+2 rows, 3 cols.
    # Blocks: 2*2*1 = 4 blocks of 2x2 x 2x3, each rank 12. Total 48.
    # Or use Strassen on the 2x2: for each (a,b) pair, we have a 2x2 A block and 2x3 B block.
    # That's still 2x2x3 per block = rank 12. 4 blocks * 12 = 48.
    #
    # Better for 4x4x3: Use the fact that it's (2x2) ⊗ (2x1) for A and (2x2) ⊗ (2x3) for B?
    # No, that's not a clean Kronecker.
    #
    # Let's just go with: 4x4x9 = 4x4x4 + 4x4x4 + 4x4x1
    # Using Strassen (rank 49) for each 4x4x4 and schoolbook (rank 16) for 4x4x1.
    # Total rank = 49 + 49 + 16 = 114.
    
    # Build 4x4x4 Strassen decomposition
    def _strassen_4x4x4():
        """4x4x4 using Kronecker product of two Strassen 2x2x2.
        Rank = 7 * 7 = 49.
        A is 4x4 = (2x2) ⊗ (2x2), B is 4x4 = (2x2) ⊗ (2x2), C is 4x4.
        """
        U1, V1, W1 = st_2x2x2
        U2, V2, W2 = st_2x2x2
        r1 = len(U1[0])  # 7
        r2 = len(U2[0])  # 7
        rank = r1 * r2  # 49
        
        n, m, p = 4, 4, 4
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for i1 in range(2):
            for j1 in range(2):
                for i2 in range(2):
                    for j2 in range(2):
                        u_row = (i1 * 2 + i2) * m + (j1 * 2 + j2)
                        for t1 in range(r1):
                            for t2 in range(r2):
                                t = t1 * r2 + t2
                                U[u_row][t] = U1[i1 * 2 + j1][t1] * U2[i2 * 2 + j2][t2]
        
        for j1 in range(2):
            for k1 in range(2):
                for j2 in range(2):
                    for k2 in range(2):
                        v_row = (j1 * 2 + j2) * p + (k1 * 2 + k2)
                        for t1 in range(r1):
                            for t2 in range(r2):
                                t = t1 * r2 + t2
                                V[v_row][t] = V1[j1 * 2 + k1][t1] * V2[j2 * 2 + k2][t2]
        
        for k1 in range(2):
            for i1 in range(2):
                for k2 in range(2):
                    for i2 in range(2):
                        w_row = (k1 * 2 + k2) * n + (i1 * 2 + i2)
                        for t1 in range(r1):
                            for t2 in range(r2):
                                t = t1 * r2 + t2
                                W[w_row][t] = W1[k1 * 2 + i1][t1] * W2[k2 * 2 + i2][t2]
        
        return U, V, W
    
    def _schoolbook_4x4x1():
        """4x4 * 4x1 -> 4x1, schoolbook rank 16."""
        return _schoolbook(4, 4, 1)
    
    # Build the combined 4x4x9 decomposition
    # 4x4x9 = 4x4x4 (cols 0-3) + 4x4x4 (cols 4-7) + 4x4x1 (col 8)
    
    U_444_1, V_444_1, W_444_1 = _strassen_4x4x4()
    U_444_2, V_444_2, W_444_2 = _strassen_4x4x4()
    U_441, V_441, W_441 = _schoolbook_4x4x1()
    
    r1 = len(U_444_1[0])  # 49
    r2 = len(U_444_2[0])  # 49
    r3 = len(U_441[0])    # 16
    total_rank = r1 + r2 + r3  # 114
    
    U_full = [[0] * total_rank for _ in range(n * m)]
    V_full = [[0] * total_rank for _ in range(m * p)]
    W_full = [[0] * total_rank for _ in range(p * n)]
    
    # Block 1: 4x4x4, cols 0-3 of B, cols 0-3 of C
    for i in range(4):
        for j in range(4):
            for t in range(r1):
                U_full[i * m + j][t] = U_444_1[i * m + j][t]
    for j in range(4):
        for k in range(4):
            for t in range(r1):
                V_full[j * p + k][t] = V_444_1[j * m + k][t]
    for k in range(4):
        for i in range(4):
            for t in range(r1):
                W_full[k * n + i][t] = W_444_1[k * n + i][t]
    
    # Block 2: 4x4x4, cols 4-7 of B, cols 4-7 of C
    off2 = r1
    for i in range(4):
        for j in range(4):
            for t in range(r2):
                U_full[i * m + j][off2 + t] = U_444_2[i * m + j][t]
    for j in range(4):
        for k in range(4):
            for t in range(r2):
                V_full[j * p + (k + 4)][off2 + t] = V_444_2[j * m + k][t]
    for k in range(4):
        for i in range(4):
            for t in range(r2):
                W_full[(k + 4) * n + i][off2 + t] = W_444_2[k * n + i][t]
    
    # Block 3: 4x4x1, col 8 of B, col 8 of C
    off3 = r1 + r2
    for i in range(4):
        for j in range(4):
            for t in range(r3):
                U_full[i * m + j][off3 + t] = U_441[i * m + j][t]
    for j in range(4):
        for t in range(r3):
            V_full[j * p + 8][off3 + t] = V_441[j * m][t]
    for i in range(4):
        for t in range(r3):
            W_full[8 * n + i][off3 + t] = W_441[8 * n + i][t]
    
    # Verify
    if not _verify(U_full, V_full, W_full, n, m, p):
        # Fallback to schoolbook
        U_sb, V_sb, W_sb = _schoolbook(n, m, p)
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    best = {"U": U_full, "V": V_full, "W": W_full}
    best_rank = total_rank
    
    # Try alternative partition: 4x4x3 + 4x4x3 + 4x4x3
    # Each 4x4x3: partition as 2x2 A blocks (2x2) and 2x3 B blocks
    # 4 blocks of 2x2x2x3, each rank 12, total 48 per 4x4x3.
    # Total: 48*3 = 144. Worse than 114.
    
    # Try: 4x4x9 as 2x4x9 + 2x4x9 (split A rows)
    # Each 2x4x9: schoolbook rank 2*4*9 = 72. Total 144. Worse.
    
    # Try: Use Strassen for 4x4x4 blocks but with different partition of 9
    # 4x4x9 = 4x4x2 + 4x4x2 + 4x4x2 + 4x4x2 + 4x4x1
    # 4x4x2: partition A as 2x2, B as 2x2 (two col-blocks of 1) -> no, B is 4x2.
    # 4x4x2: A is 4x4, B is 4x2. 
    # Partition: 2x2 A blocks, 2x1 B blocks. 4 A-blocks * 2 B-blocks = 8 sub-blocks of 2x2x2x1.
    # Each 2x2x1: schoolbook rank 4. Total 8*4 = 32 per 4x4x2.
    # Or use Strassen: 2x2x2 has rank 7, but 2x2x1 is just schoolbook rank 4.
    # 5 blocks * 32 = 160. Worse.
    
    # The 114-rank solution (2x Strassen 4x4x4 + 1x schoolbook 4x4x1) is our best.
    # Let's try to see if we can do 4x4x5 better than 49+16=65.
    # 4x4x5 = 4x4x4 + 4x4x1: 49 + 16 = 65.
    # Alternative: partition 4x4x5 as 2x4x5 + 2x4x5.
    # 2x4x5: A is 2x4, B is 4x5. Schoolbook rank 2*4*5 = 40. Two of them: 80. Worse.
    # Or 4x4x5 as 4x2x5 + 4x2x5.
    # 4x2x5: A is 4x2, B is 2x5. Schoolbook rank 4*2*5 = 40. Two: 80. Worse.
    
    # Try to improve 4x4x4 below 49?
    # Known: rank of 4x4x4 is at most 49 (Strassen), and it's known that 
    # the rank is between 49 and 72. We'll stick with 49.
    
    # Check time and try random search for small improvements
    if time.time() < deadline - 5:
        # Try random perturbation of the 4x4x1 block to see if we can save rank
        # This is unlikely to work but let's try a few quick searches
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            # Skip - random search for exact integer decompositions is infeasible
            break
    
    return best
