"""Tensor decomposition for 4x5 * 5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return (U, V, W) for 2x2 * 2x2 -> 2x2 using Strassen's 7 products.
    U shape (4,7), V shape (4,7), W shape (4,7)
    Row indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1 = (a11+a22)*(b11+b22)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1  # c11, c22
    
    # P2 = (a21+a22)*b11
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[1,1] = 1; W[3,1] = -1  # c21, c22
    
    # P3 = a11*(b12-b22)
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[2,2] = 1; W[3,2] = 1  # c12, c22
    
    # P4 = a22*(b21-b11)
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[1,3] = 1  # c11, c21
    
    # P5 = (a11+a12)*b22
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[2,4] = 1  # c11, c12
    
    # P6 = (a21-a11)*(b11+b12)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1  # c22
    
    # P7 = (a12-a22)*(b21+b22)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1  # c11
    
    return U, V, W

def _schoolbook(n, m, p):
    """
    Schoolbook decomposition for n x m * m x p -> n x p.
    Returns U (n*m, n*m*p), V (m*p, n*m*p), W (p*n, n*m*p)
    """
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
    return U, V, W

def _block_strassen(n, m, p, block_n, block_m, block_p):
    """
    Build decomposition for n x m * m x p using block Strassen.
    Splits into blocks of size block_n x block_m x block_p.
    Uses Strassen at the block level (2x2 blocks) combined with 
    inner decompositions for each block.
    
    For now, we handle the specific case where n=block_n*2, m=block_m*2, p=block_p*2
    and use 2x2 Strassen at block level with schoolbook or Strassen inside.
    """
    # This is a general-purpose block decomposition builder
    # For the specific case of 4x5x6, we'll use a custom approach
    
    # Actually, let's use a simpler approach: direct construction
    pass

def _build_4x5x6():
    """
    Build the 4x5 * 5x6 -> 4x6 decomposition.
    
    Strategy: Split the middle dimension (5) into 4+1.
    - 4x4 * 4x6 part: Use block Strassen (2x2 blocks of 2x2)
      This gives 4x4x4 (rank 49) + 4x4x2 (rank 28) = 77
    - 4x1 * 1x6 part: Schoolbook, rank 24
    Total: 101
    
    Alternative: Split differently or use other methods.
    
    Let me try: 4x5 * 5x6
    Split m=5 as 2+3:
    - 4x2 * 2x6: rank = 4*2*6 = 48 (schoolbook) or better
    - 4x3 * 3x6: rank = 4*3*6 = 72 (schoolbook) or better
    Total schoolbook: 120
    
    Better: Use Strassen where possible.
    4x4 * 4x6: 
      Split p=6 as 4+2
      4x4x4: Strassen rank 49
      4x4x2: Block Strassen rank 28
      Total for 4x4x6: 77
    4x1 * 1x6: Schoolbook rank 24
    Total: 101
    """
    U2, V2, W2 = _strassen_2x2()
    
    n, m, p = 4, 5, 6
    
    # Part 1: 4x4 * 4x4 -> 4x4 (Strassen, rank 49)
    # Part 2: 4x4 * 4x2 -> 4x2 (Block Strassen, rank 28)  
    # Part 3: 4x1 * 1x6 -> 4x6 (Schoolbook, rank 24)
    
    total_rank = 49 + 28 + 24
    U = np.zeros((n*m, total_rank))  # 20 x total_rank
    V = np.zeros((m*p, total_rank))  # 30 x total_rank
    W = np.zeros((p*n, total_rank))  # 24 x total_rank
    
    col_offset = 0
    
    # Part 1: 4x4x4 Strassen (i in 0..3, j in 0..3, k in 0..3)
    U1, V1, W1 = _strassen_4x4()
    U[:, col_offset:col_offset+49] = U1
    V[:16, col_offset:col_offset+49] = V1  # j in 0..3, k in 0..3 -> rows 0..15
    W[:16, col_offset:col_offset+49] = W1  # k in 0..3, i in 0..3 -> rows 0..15
    col_offset += 49
    
    # Part 2: 4x4x2 (i in 0..3, j in 0..3, k in 4..5)
    U2p, V2p, W2p = _strassen_4x4x2()
    U[:, col_offset:col_offset+28] = U2p
    # V rows: j in 0..3, k in 4..5 -> j*p+k = j*6+4, j*6+5
    # V2p has shape (8, 28) where row = j*2+k_local, k_local in 0..1
    # Global row = j*6 + (4+k_local)
    for j in range(4):
        for kl in range(2):
            V[j*6 + 4 + kl, col_offset:col_offset+28] = V2p[j*2+kl, :]
    # W rows: k in 4..5, i in 0..3 -> k*n+i = 4*4+i, 5*4+i
    # W2p has shape (8, 28) where row = k_local*4+i, k_local in 0..1
    # Global row = (4+k_local)*4 + i
    for kl in range(2):
        for i in range(4):
            W[(4+kl)*4 + i, col_offset:col_offset+28] = W2p[kl*4+i, :]
    col_offset += 28
    
    # Part 3: 4x1x6 (i in 0..3, j in 4, k in 0..5)
    U3, V3, W3 = _schoolbook(4, 1, 6)
    # U3 shape (4, 24), row = i
    # Global U row = i*5 + 4
    for i in range(4):
        U[i*5+4, col_offset:col_offset+24] = U3[i, :]
    # V3 shape (6, 24), row = k
    # Global V row = 4*6 + k = 24+k
    for k in range(6):
        V[24+k, col_offset:col_offset+24] = V3[k, :]
    # W3 shape (24, 24), row = k*4+i
    # Global W row = k*4+i (same since k in 0..5, i in 0..3)
    W[:, col_offset:col_offset+24] = W3
    
    return U.tolist(), V.tolist(), W.tolist()

def _strassen_4x4():
    """
    Build 4x4x4 decomposition using 2x2 Strassen at block level.
    Returns U (16,49), V (16,49), W (16,49)
    """
    U2, V2, W2 = _strassen_2x2()
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # 7 Strassen products, each with 7 inner terms
    # P1 = (A11+A22)*(B11+B22) -> C11 += 1, C22 += 1
    # P2 = (A21+A22)*B11 -> C21 += 1, C22 -= 1
    # P3 = A11*(B12-B22) -> C12 += 1, C22 += 1
    # P4 = A22*(B21-B11) -> C11 += 1, C21 += 1
    # P5 = (A11+A12)*B22 -> C11 -= 1, C12 += 1
    # P6 = (A21-A11)*(B11+B12) -> C22 += 1
    # P7 = (A12-A22)*(B21+B22) -> C11 += 1
    
    strassen_defs = [
        # (a_blocks, b_blocks, c_blocks)
        # Each block is (block_row, block_col, sign)
        ([((0,0),1), ((1,1),1)],
         [((0,0),1), ((1,1),1)],
         [((0,0),1), ((1,1),1)]),
        ([((1,0),1), ((1,1),1)],
         [((0,0),1)],
         [((1,0),1), ((1,1),-1)]),
        ([((0,0),1)],
         [((0,1),1), ((1,1),-1)],
         [((0,1),1), ((1,1),1)]),
        ([((1,1),1)],
         [((1,0),1), ((0,0),-1)],
         [((0,0),1), ((1,0),1)]),
        ([((0,0),1), ((0,1),1)],
         [((1,1),1)],
         [((0,0),-1), ((0,1),1)]),
        ([((1,0),1), ((0,0),-1)],
         [((0,0),1), ((0,1),1)],
         [((1,1),1)]),
        ([((0,1),1), ((1,1),-1)],
         [((1,0),1), ((1,1),1)],
         [((0,0),1)]),
    ]
    
    for s, (a_blocks, b_blocks, c_blocks) in enumerate(strassen_defs):
        for t in range(7):
            col = s * 7 + t
            
            # A contributions
            for la_i in range(2):
                for la_j in range(2):
                    local_a_row = la_i * 2 + la_j
                    coeff_a = U2[local_a_row, t]
                    if coeff_a != 0:
                        for (bi, bj), sigma in a_blocks:
                            global_i = bi * 2 + la_i
                            global_j = bj * 2 + la_j
                            u_row = global_i * 4 + global_j
                            U[u_row, col] += coeff_a * sigma
            
            # B contributions
            for lb_j in range(2):
                for lb_k in range(2):
                    local_b_row = lb_j * 2 + lb_k
                    coeff_b = V2[local_b_row, t]
                    if coeff_b != 0:
                        for (bj, bk), sigma in b_blocks:
                            global_j = bj * 2 + lb_j
                            global_k = bk * 2 + lb_k
                            v_row = global_j * 4 + global_k
                            V[v_row, col] += coeff_b * sigma
            
            # C contributions
            for lc_k in range(2):
                for lc_i in range(2):
                    local_c_row = lc_k * 2 + lc_i
                    coeff_c = W2[local_c_row, t]
                    if coeff_c != 0:
                        for (bi, bk), sigma in c_blocks:
                            global_k = bk * 2 + lc_k
                            global_i = bi * 2 + lc_i
                            w_row = global_k * 4 + global_i
                            W[w_row, col] += coeff_c * sigma
    
    return U, V, W

def _strassen_4x4x2():
    """
    Build 4x4x2 decomposition.
    A(4x4) * B(4x2) -> C(4x2)
    
    We can view this as two 4x4x1 problems or use block structure.
    Split into 2x2 blocks: A has 4 blocks, B has 2 blocks, C has 2 blocks.
    
    C1 = A11*B11 + A12*B21
    C2 = A21*B11 + A22*B21
    
    Each 2x2*2x2->2x2 product uses Strassen (rank 7).
    4 such products: rank 28.
    
    Returns U (16,28), V (8,28), W (8,28)
    """
    U2, V2, W2 = _strassen_2x2()
    rank = 28
    U = np.zeros((16, rank))
    V = np.zeros((8, rank))
    W = np.zeros((8, rank))
    
    # 4 products:
    # Product 0: A11 * B11 -> C1 (top half of C)
    # Product 1: A12 * B21 -> C1
    # Product 2: A21 * B11 -> C2 (bottom half of C)
    # Product 3: A22 * B21 -> C2
    
    products = [
        # (a_row_start, a_col_start, b_row_start, c_row_start)
        (0, 0, 0, 0),   # A11 * B11 -> C1
        (0, 2, 2, 0),   # A12 * B21 -> C1
        (2, 0, 0, 2),   # A21 * B11 -> C2
        (2, 2, 2, 2),   # A22 * B21 -> C2
    ]
    
    for s, (a_rs, a_cs, b_rs, c_rs) in enumerate(products):
        for t in range(7):
            col = s * 7 + t
            
            # A: 2x2 block at (a_rs, a_cs)
            for la_i in range(2):
                for la_j in range(2):
                    local_a_row = la_i * 2 + la_j
                    coeff_a = U2[local_a_row, t]
                    if coeff_a != 0:
                        global_i = a_rs + la_i
                        global_j = a_cs + la_j
                        u_row = global_i * 4 + global_j
                        U[u_row, col] += coeff_a
            
            # B: 2x2 block at (b_rs, 0) - B is 4x2, so 2 blocks vertically
            for lb_j in range(2):
                for lb_k in range(2):
                    local_b_row = lb_j * 2 + lb_k
                    coeff_b = V2[local_b_row, t]
                    if coeff_b != 0:
                        global_j = b_rs + lb_j
                        global_k = lb_k
                        v_row = global_j * 2 + global_k
                        V[v_row, col] += coeff_b
            
            # C: 2x2 block at (c_rs, 0) - C is 4x2, so 2 blocks vertically
            for lc_k in range(2):
                for lc_i in range(2):
                    local_c_row = lc_k * 2 + lc_i
                    coeff_c = W2[local_c_row, t]
                    if coeff_c != 0:
                        global_k = lc_k
                        global_i = c_rs + lc_i
                        w_row = global_k * 4 + global_i
                        W[w_row, col] += coeff_c
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    try:
        U, V, W = _build_4x5x6()
        
        # Verify within time budget
        if time.time() < deadline - 10:
            # Quick verification with random matrices
            U_np = np.array(U)
            V_np = np.array(V)
            W_np = np.array(W)
            
            # Test with a few random matrices
            for trial in range(3):
                A = np.random.randn(n, m)
                B = np.random.randn(m, p)
                C_expected = A @ B
                
                # Compute via decomposition
                # Flatten A and B
                a_flat = A.flatten()  # row-major: i*m+j
                b_flat = B.flatten()  # row-major: j*p+k
                
                # u = U^T @ a_flat, v = V^T @ b_flat
                u = U_np.T @ a_flat
                v = V_np.T @ b_flat
                
                # c_flat = W^T @ (u * v)
                c_flat = W_np.T @ (u * v)
                
                C_computed = c_flat.reshape(n, p)
                
                if not np.allclose(C_computed, C_expected, atol=1e-10):
                    # Fallback to schoolbook
                    rank = n * m * p
                    U = np.zeros((n*m, rank))
                    V = np.zeros((m*p, rank))
                    W = np.zeros((p*n, rank))
                    col = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                U[i*m+j, col] = 1
                                V[j*p+k, col] = 1
                                W[k*n+i, col] = 1
                                col += 1
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return {"U": U, "V": V, "W": W}
        
    except Exception:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
