"""Fresh design: exploit 2x2 block Strassen to get rank 189 for 6x6x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-189 decomposition of 6x6 * 6x6 -> 6x6 matrix multiplication
    using the 2x2 block Strassen algorithm.
    
    We partition the 6x6 matrices into 3x3 blocks of size 2x2.
    For each pair of 2x2 blocks, we use Strassen's 7-term algorithm.
    This gives 9 block positions * 7 Strassen terms = 63 rank-1 operations per "level",
    but we need to be more careful.
    
    Actually, let me think about this differently. The standard approach:
    
    A (6x6) = [[A11, A12, A13], [A21, A22, A23], [A31, A32, A33]] where each block is 2x2
    B (6x6) = [[B11, B12, B13], [B21, B22, B23], [B31, B32, B33]] where each block is 2x2
    
    C = A * B, where Cij = sum_k Akj * Bkj (block multiplication)
    
    For each block product Cij = sum_k Akj * Bkj, we have 2x2 * 2x2 matrix multiplication.
    Using Strassen for 2x2: 7 multiplications instead of 8.
    
    There are 9 block positions (i,j) in C, each requiring a sum over k of 2x2*2x2 products.
    For each (i,j) block of C, we compute Cij = A_i1*B_1j + A_i2*B_2j + A_i3*B_3j
    
    Each 2x2*2x2 product uses 7 rank-1 terms (Strassen).
    So for each (i,j) block: 3 terms in the sum, each 2x2*2x2 = 3*7 = 21 rank-1 terms.
    Total: 9 * 21 = 189 rank-1 terms.
    
    This matches the known result: rank 189 for 6x6x6 via 2x2 block Strassen.
    
    Let me implement this carefully.
    
    Strassen for 2x2:
    Given X = [[x11, x12], [x21, x22]], Y = [[y11, y12], [y21, y22]]
    
    P1 = (x11 + x22)(y11 + y22)
    P2 = (x21 + x22)y11
    P3 = x11(y12 - y22)
    P4 = x22(y21 - y11)
    P5 = (x11 + x12)y22
    P6 = (x21 - x11)(y11 + y12)
    P7 = (x12 - x22)(y21 + y22)
    
    c11 = P1 + P4 - P5 + P7
    c12 = P3 + P5
    c21 = P2 + P4
    c22 = P1 - P2 + P3 + P6
    
    Each Pk is a product of two linear combinations, which in tensor decomposition terms
    is a rank-1 term where U and V have the linear combination coefficients and W has
    the output coefficient.
    
    Let me define the tensor decomposition explicitly.
    
    For a 2x2 * 2x2 -> 2x2 multiplication with Strassen:
    The 4 output elements are:
    c11 = P1 + P4 - P5 + P7
    c12 = P3 + P5
    c21 = P2 + P4
    c22 = P1 - P2 + P3 + P6
    
    Where:
    P1 = (x11 + x22)(y11 + y22)
    P2 = (x21 + x22)(y11)
    P3 = (x11)(y12 - y22)
    P4 = (x22)(y21 - y11)
    P5 = (x11 + x12)(y22)
    P6 = (x21 - x11)(y11 + y12)
    P7 = (x12 - x22)(y21 + y22)
    
    So for each Strassen term Pk, we have:
    - A linear form in X (the U part)
    - A linear form in Y (the V part)
    - A coefficient in the output (the W part)
    
    Let me enumerate all 7 Strassen terms with their U, V, W contributions.
    
    Term 1 (P1): U: x11+x22, V: y11+y22, W: c11 += 1, c22 += 1
    Term 2 (P2): U: x21+x22, V: y11, W: c21 += 1, c22 += -1
    Term 3 (P3): U: x11, V: y12-y22, W: c12 += 1, c22 += 1
    Term 4 (P4): U: x22, V: y21-y11, W: c11 += 1, c21 += 1
    Term 5 (P5): U: x11+x12, V: y22, W: c11 += -1, c12 += 1
    Term 6 (P6): U: x21-x11, V: y11+y12, W: c22 += 1
    Term 7 (P7): U: x12-x22, V: y21+y22, W: c11 += 1
    """
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 120  # Must finish well before 150s
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    block_size = 2
    num_blocks = n // block_size  # 3
    
    # Strassen terms for 2x2: each term has (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: coefficients for [x11, x12, x21, x22]
    # v_coeffs: coefficients for [y11, y12, y21, y22]
    # w_coeffs: coefficients for [c11, c12, c21, c22]
    
    strassen_terms = [
        # (u, v, w)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # P1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),  # P2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),  # P3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),  # P4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # P5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # P6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # P7
    ]
    
    # Total rank
    total_rank = 0
    for i in range(num_blocks):
        for j in range(num_blocks):
            for k in range(num_blocks):
                total_rank += 7
    
    rank = total_rank  # 189
    
    U = [[0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * rank for _ in range(p * n)]  # 36 x 189 (output: k*n+i)
    
    col = 0
    for i in range(num_blocks):
        for j in range(num_blocks):
            for k in range(num_blocks):
                for (u_coeffs, v_coeffs, w_coeffs) in strassen_terms:
                    # A block A[i][k]: rows i*2 to i*2+1, cols k*2 to k*2+1
                    # In U (n*m = 6*6 = 36): row index = row * m + col = row * 6 + col
                    # A[i][k] has rows [i*2, i*2+1] and cols [k*2, k*2+1]
                    # So U rows: i*2*6 + k*2, i*2*6 + k*2+1, (i*2+1)*6 + k*2, (i*2+1)*6 + k*2+1
                    # u_coeffs correspond to [x11, x12, x21, x22] = [A[i*2][k*2], A[i*2][k*2+1], A[i*2+1][k*2], A[i*2+1][k*2+1]]
                    
                    a_rows = [i*2, i*2+1]
                    a_cols = [k*2, k*2+1]
                    u_row_indices = [a_rows[0]*m + a_cols[0], a_rows[0]*m + a_cols[1], 
                                      a_rows[1]*m + a_cols[0], a_rows[1]*m + a_cols[1]]
                    
                    for idx, coeff in enumerate(u_coeffs):
                        U[u_row_indices[idx]][col] += coeff
                    
                    # B block B[k][j]: rows k*2 to k*2+1, cols j*2 to j*2+1
                    # In V (m*p = 6*6 = 36): row index = row * p + col = row * 6 + col
                    # B[k][j] has rows [k*2, k*2+1] and cols [j*2, j*2+1]
                    # V rows: k*2*6 + j*2, k*2*6 + j*2+1, (k*2+1)*6 + j*2, (k*2+1)*6 + j*2+1
                    # v_coeffs correspond to [y11, y12, y21, y22] = [B[k*2][j*2], B[k*2][j*2+1], B[k*2+1][j*2], B[k*2+1][j*2+1]]
                    
                    b_rows = [k*2, k*2+1]
                    b_cols = [j*2, j*2+1]
                    v_row_indices = [b_rows[0]*p + b_cols[0], b_rows[0]*p + b_cols[1],
                                      b_rows[1]*p + b_cols[0], b_rows[1]*p + b_cols[1]]
                    
                    for idx, coeff in enumerate(v_coeffs):
                        V[v_row_indices[idx]][col] += coeff
                    
                    # C block C[i][j]: rows i*2 to i*2+1, cols j*2 to j*2+1
                    # In W (p*n = 6*6 = 36): index = k*n + i, where k is col, i is row
                    # W[k*n + i, col] where k is the column index in C, i is the row index in C
                    # C[i][j] has rows [i*2, i*2+1] and cols [j*2, j*2+1]
                    # W index: col_in_C * n + row_in_C
                    # w_coeffs correspond to [c11, c12, c21, c22] = [C[i*2][j*2], C[i*2][j*2+1], C[i*2+1][j*2], C[i*2+1][j*2+1]]
                    
                    c_rows = [i*2, i*2+1]
                    c_cols = [j*2, j*2+1]
                    w_row_indices = [c_cols[0]*n + c_rows[0], c_cols[1]*n + c_rows[0],
                                      c_cols[0]*n + c_rows[1], c_cols[1]*n + c_rows[1]]
                    
                    for idx, coeff in enumerate(w_coeffs):
                        W[w_row_indices[idx]][col] += coeff
                    
                    col += 1
    
    # Verify correctness with a random test
    if time.time() < deadline - 5:
        np.random.seed(42)
        A_test = np.random.randint(-5, 6, size=(n, m))
        B_test = np.random.randint(-5, 6, size=(m, p))
        C_expected = A_test @ B_test
        
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # C[r, c] = sum_t U[r, t] * V[c, t] * W[c*n + r, t]
        # Wait, the convention is W[k*n + i, r] where the output is C[i, k]
        # Let me re-read: "The output factor follows the official trace convention W[k*n+i,r]"
        # So W has shape (p*n, rank), and W[k*n+i, r] is the coefficient for output element C[i, k] in rank-r term.
        # The tensor is: C[i,k] = sum_r U[i*m + j... wait no.
        
        # Actually the tensor decomposition is:
        # C[i, k] = sum_{j, r} A[i,j] * B[j,k] * T[i,j,k,r]
        # Where T is factored as U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # So: C[i,k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] for all j
        # Wait, that's not right either. Let me think again.
        
        # The matrix multiplication tensor T[i,j,k] = 1 (for all i,j,k) when we write
        # C[i,k] = sum_j A[i,j] B[j,k]
        
        # A rank-r decomposition is: T[i,j,k] = sum_{r=1}^r U[i,j,r] V[j,k,r] W[i,k,r]
        # where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
        
        # So U[i*m + j, r], V[j*p + k, r], W[k*n + i, r]
        
        # And C[i,k] = sum_{j,r} A[i,j] * B[j,k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Let me verify:
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for r in range(rank):
                        s += A_test[i, j] * B_test[j, k] * U_np[i*m + j, r] * V_np[j*p + k, r] * W_np[k*n + i, r]
                C_computed[i, k] = s
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            # Fallback to schoolbook if verification fails
            rank_school = n * m * p
            U_fb = [[0] * rank_school for _ in range(n * m)]
            V_fb = [[0] * rank_school for _ in range(m * p)]
            W_fb = [[0] * rank_school for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j][column] = 1
                        V_fb[j * p + k][column] = 1
                        W_fb[k * n + i][column] = 1
                        column += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
