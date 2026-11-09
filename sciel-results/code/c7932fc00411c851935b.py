"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _strassen_2x2():
    """Return the 7 rank-1 terms for 2x2x2 matrix multiplication.
    
    Each term is (u, v, w) where u, v, w are 4D vectors representing
    the 2x2 matrices flattened in row-major order.
    """
    # Strassen's 7 products for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Flattened 2x2 matrices: [a11, a12, a21, a22]
    terms = []
    
    # M1 = (A11+A22)(B11+B22)
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, 1]
    terms.append((u1, v1, w1))
    
    # M2 = (A21+A22)B11
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 0, 1, 0]
    terms.append((u2, v2, w2))
    
    # M3 = A11(B21-B22)
    u3 = [1, 0, 0, 0]
    v3 = [0, 0, 1, -1]
    w3 = [0, 1, 0, 0]
    terms.append((u3, v3, w3))
    
    # M4 = A22(B12-B11)
    u4 = [0, 0, 0, 1]
    v4 = [0, 1, -1, 0]
    w4 = [0, 0, 0, 1]
    terms.append((u4, v4, w4))
    
    # M5 = (A11+A12)B22
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [0, 0, 0, 1]
    terms.append((u5, v5, w5))
    
    # M6 = (A21-A11)(B11+B12)
    u6 = [0, 0, 1, 0]
    v6 = [1, 1, 0, 0]
    w6 = [0, 0, -1, 0]
    terms.append((u6, v6, w6))
    
    # M7 = (A12-A22)(B21+B22)
    u7 = [0, 1, 0, 0]
    v7 = [0, 0, 1, 1]
    w7 = [0, 1, 0, 1]
    terms.append((u7, v7, w7))
    
    return terms

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # For 4x4x9: Split into 2x2 blocks in the i,j dimensions
    # A is 4x4: split into 2x2 blocks (A11, A12; A21, A22)
    # B is 4x9: split into 2x2 blocks of columns (B11, B12, B13; B21, B22, B23)
    # C is 4x9: split into 2x2 blocks (C11, C12, C13; C21, C22, C23)
    
    # Each C block (2x2) = sum over j-blocks of A_block * B_block
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C13 = A11*B13 + A12*B23
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # C23 = A21*B13 + A22*B23
    
    # We need to compute 12 products of 2x2 matrices:
    # A11*B11, A11*B12, A11*B13, A12*B21, A12*B22, A12*B23,
    # A21*B11, A21*B12, A21*B13, A22*B21, A22*B22, A22*B23
    
    # Using Strassen for each 2x2 product would give 7*12 = 84 terms,
    # but we can share computations.
    
    # Alternative: Use the fact that we're multiplying 4x4 by 4x9.
    # Let's use a different approach: decompose the 4x4 matrix into
    # 2x2 blocks and use Strassen recursively.
    
    # Actually, let's think more carefully. The tensor for 4x4x9 has
    # rank at most 4*4*9 = 144 (schoolbook).
    
    # Using Strassen-style: Split 4x4 into 2x2 blocks.
    # A = [A11 A12; A21 A22] where each is 2x2
    # B = [B11 B12 B13; B21 B22 B23] where each is 2x3
    
    # C = [A11*B11 + A12*B21, A11*B12 + A12*B22, A11*B13 + A12*B23;
    #      A21*B11 + A22*B21, A21*B12 + A22*B22, A21*B13 + A22*B23]
    
    # Each 2x2 * 2x3 product can be done with rank at most 7*3=21
    # (using Strassen for the 2x2 part and schoolbook for the 3 columns)
    # But we can do better.
    
    # Let's use a direct approach: enumerate all possible rank-1 terms
    # and try to find a minimal decomposition.
    
    # For now, let's implement a structured decomposition that combines
    # Strassen sub-blocks.
    
    # Strategy: Use 7 terms per 2x2x2 block, and we have 4 row-blocks 
    # and 3 col-blocks in B, so 12 2x2x2 sub-problems.
    # But we can share: A11 appears in 3 products, A12 in 3, etc.
    
    # Let's build the decomposition explicitly.
    # Indexing: A[i,j] with i in 0..3, j in 0..3
    # B[j,k] with j in 0..3, k in 0..8
    # C[i,k] with i in 0..3, k in 0..8
    
    # Block structure:
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
    # A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # B11 = B[0:2, 0:3], B12 = B[0:2, 3:6], B13 = B[0:2, 6:9]
    # B21 = B[2:4, 0:3], B22 = B[2:4, 3:6], B23 = B[2:4, 6:9]
    # C11 = C[0:2, 0:3], C12 = C[0:2, 3:6], C13 = C[0:2, 6:9]
    # C21 = C[2:4, 0:3], C22 = C[2:4, 3:6], C23 = C[2:4, 6:9]
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C13 = A11*B13 + A12*B23
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # C23 = A21*B13 + A22*B23
    
    # For each 2x2 * 2x3 product, we can use a rank-7 decomposition
    # for the 2x2 part applied to each column of the 2x3 matrix.
    
    # Let's define a helper to get the rank-1 terms for 2x2 * 2x3.
    
    def _strassen_2x2_times_2x3():
        """Return rank-1 terms for 2x2 * 2x3 matrix multiplication.
        
        Each term is (u, v, w) where:
        - u is a 4D vector (flattened 2x2 matrix A)
        - v is a 6D vector (flattened 2x3 matrix B)  
        - w is a 6D vector (flattened 2x3 matrix C)
        
        The decomposition is: C = sum_t u_t * v_t^T * w_t
        where u_t is the coefficient for A, v_t for B, w_t for C.
        """
        terms = []
        # Use Strassen's 7 products, each applied to each of the 3 columns
        # But we can do better by sharing.
        
        # Actually, for 2x2 * 2x3, we can use the 7 Strassen products
        # where each product is a scalar, and then we reconstruct.
        
        # Let me think differently. The tensor for 2x2*2x3 has dimension
        # 4x6x6 = 144. Schoolbook rank is 12 (2*2*3).
        # Can we do better?
        
        # For now, use schoolbook for the 2x2*2x3 sub-problems.
        # That gives 12 terms per sub-problem, 6 sub-problems = 72 total.
        
        # Wait, that's already better than 144!
        
        # Let's use schoolbook for each 2x2*2x3 block.
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    u = [0]*4
                    v = [0]*6
                    w = [0]*6
                    u[i*2+j] = 1
                    v[j*3+k] = 1
                    w[i*3+k] = 1
                    terms.append((u, v, w))
        
        return terms
    
    # Now build the full decomposition.
    # We have 6 sub-problems, each with 12 terms = 72 total terms.
    
    # Map global indices to local indices.
    # A[i,j]: i in 0..3, j in 0..3
    # B[j,k]: j in 0..3, k in 0..8
    # C[i,k]: i in 0..3, k in 0..8
    
    # Sub-problem 1: A11*B11 -> C11
    # A11[i,j] = A[i,j] for i,j in 0..1
    # B11[j,k] = B[j,k] for j in 0..1, k in 0..2
    # C11[i,k] = C[i,k] for i in 0..1, k in 0..2
    
    # Sub-problem 2: A12*B21 -> C11
    # A12[i,j] = A[i,j+2] for i in 0..1, j in 0..1
    # B21[j,k] = B[j+2,k] for j in 0..1, k in 0..2
    # C11[i,k] = C[i,k] for i in 0..1, k in 0..2
    
    # And so on...
    
    # Let's build the full set of terms.
    all_terms = []
    
    # Define the 6 sub-problems:
    # (A_block_start_i, A_block_start_j, B_block_start_j, B_block_start_k, C_block_start_i, C_block_start_k)
    subproblems = [
        (0, 0, 0, 0, 0, 0),   # A11*B11 -> C11
        (0, 2, 2, 0, 0, 0),   # A12*B21 -> C11
        (0, 0, 0, 3, 0, 3),   # A11*B12 -> C12
        (0, 2, 2, 3, 0, 3),   # A12*B22 -> C12
        (0, 0, 0, 6, 0, 6),   # A11*B13 -> C13
        (0, 2, 2, 6, 0, 6),   # A12*B23 -> C13
        (2, 0, 0, 0, 2, 0),   # A21*B11 -> C21
        (2, 2, 2, 0, 2, 0),   # A22*B21 -> C21
        (2, 0, 0, 3, 2, 3),   # A21*B12 -> C22
        (2, 2, 2, 3, 2, 3),   # A22*B22 -> C22
        (2, 0, 0, 6, 2, 6),   # A21*B13 -> C23
        (2, 2, 2, 6, 2, 6),   # A22*B23 -> C23
    ]
    
    for (ai, aj, bj, bk, ci, ck) in subproblems:
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    u = [0] * (n*m)  # 16
                    v = [0] * (m*p)  # 36
                    w = [0] * (p*n)  # 36
                    
                    # A index: (ai+i, aj+j)
                    a_idx = (ai + i) * m + (aj + j)
                    u[a_idx] = 1
                    
                    # B index: (bj+j, bk+k)
                    b_idx = (bj + j) * p + (bk + k)
                    v[b_idx] = 1
                    
                    # C index: (ci+i, ck+k)
                    c_idx = (ck + k) * n + (ci + i)
                    w[c_idx] = 1
                    
                    all_terms.append((u, v, w))
    
    rank = len(all_terms)
    
    # Build the U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u, v, w) in enumerate(all_terms):
        for i in range(n * m):
            U[i][col] = u[i]
        for j in range(m * p):
            V[j][col] = v[j]
        for k in range(p * n):
            W[k][col] = w[k]
    
    # Verify correctness (optional, but good for debugging)
    if time.time() < deadline:
        # Check that the decomposition is correct
        # C[i,k] = sum_j A[i,j] * B[j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Wait, the convention is W[k*n+i, c]
        # So C[i,k] = sum_c sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # Let's verify with a random test
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for c in range(rank):
            U_col = np.array([U[i][c] for i in range(n*m)]).reshape(n, m)
            V_col = np.array([V[j][c] for j in range(m*p)]).reshape(m, p)
            W_col = np.array([W[k][c] for k in range(p*n)]).reshape(p, n)
            # C += U_col * V_col * W_col? No...
            # The tensor is T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # And C[i,k] = sum_j T[i,j,k]
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        C_computed[i,k] += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # Fallback to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
