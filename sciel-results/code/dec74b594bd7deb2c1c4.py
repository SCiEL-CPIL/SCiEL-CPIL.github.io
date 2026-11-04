import time
import numpy as np
import random

DIMENSIONS = [3, 4, 8]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor T[u, v, w] = sum_r U[u,r]*V[v,r]*W[w,r] should equal 1
    iff there exist i,k,j with u=i*m+k, v=k*p+j, w=j*n+i (and consistent i,k,j).
    """
    rank = U.shape[1]
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                u = i * m + k
                v = k * p + j
                w = j * n + i
                T_target[u, v, w] = 1.0
    
    T_decomp = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_decomp += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
    
    error = np.max(np.abs(T_decomp - T_target))
    return error < tol, error

def _strassen_block(U, V, W, col, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, m, p, n):
    """
    Adds 7 rank-1 terms corresponding to a Strassen decomposition of a 2x2x2 block.
    """
    u_a = a_rows[0] * m + a_cols[0]
    u_b = a_rows[0] * m + a_cols[1]
    u_c = a_rows[1] * m + a_cols[0]
    u_d = a_rows[1] * m + a_cols[1]
    
    v_e = b_rows[0] * p + b_cols[0]
    v_f = b_rows[0] * p + b_cols[1]
    v_g = b_rows[1] * p + b_cols[0]
    v_h = b_rows[1] * p + b_cols[1]
    
    w_00 = c_cols[0] * n + c_rows[0]
    w_01 = c_cols[1] * n + c_rows[0]
    w_10 = c_cols[0] * n + c_rows[1]
    w_11 = c_cols[1] * n + c_rows[1]
    
    # M1: (a+d)(e+h)
    U[u_a, col] += 1.0
    U[u_d, col] += 1.0
    V[v_e, col] += 1.0
    V[v_h, col] += 1.0
    W[w_00, col] += 1.0
    W[w_11, col] += 1.0
    col += 1
    
    # M2: (c+d)e
    U[u_c, col] += 1.0
    U[u_d, col] += 1.0
    V[v_e, col] += 1.0
    W[w_10, col] += 1.0
    W[w_11, col] += -1.0
    col += 1
    
    # M3: a(f-h)
    U[u_a, col] += 1.0
    V[v_f, col] += 1.0
    V[v_h, col] += -1.0
    W[w_01, col] += 1.0
    W[w_11, col] += 1.0
    col += 1
    
    # M4: d(g-e)
    U[u_d, col] += 1.0
    V[v_g, col] += 1.0
    V[v_e, col] += -1.0
    W[w_00, col] += 1.0
    W[w_10, col] += 1.0
    col += 1
    
    # M5: (a+b)h
    U[u_a, col] += 1.0
    U[u_b, col] += 1.0
    V[v_h, col] += 1.0
    W[w_00, col] += -1.0
    W[w_01, col] += 1.0
    col += 1
    
    # M6: (c-a)(e+f)
    U[u_c, col] += 1.0
    U[u_a, col] += -1.0
    V[v_e, col] += 1.0
    V[v_f, col] += 1.0
    W[w_11, col] += 1.0
    col += 1
    
    # M7: (b-d)(g+h)
    U[u_b, col] += 1.0
    U[u_d, col] += -1.0
    V[v_g, col] += 1.0
    V[v_h, col] += 1.0
    W[w_00, col] += 1.0
    col += 1
    
    return col

def _bcrl_3x3x3_block(U, V, W, col, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, m, p, n):
    """
    Adds 19 rank-1 terms corresponding to a BCRL decomposition of a 3x3x3 block.
    BCRL decomposition of the 3x3x3 matrix multiplication tensor with rank 19.
    
    The BCRL scheme uses 19 bilinear forms. Here I implement a known rank-19 scheme.
    For a 3x3x3 multiplication C = A * B where A, B, C are 3x3:
    
    A = [a b c]
        [d e f]
        [g h i]
    
    B = [j k l]
        [m n o]
        [p q r]
    
    C = A * B
    
    The 19 bilinear forms from the BCRL decomposition:
    """
    # For simplicity, let's use a different approach.
    # Actually, let me think about this more carefully.
    # The current rank-88 solution uses Strassen (rank 7) for 2x2x2 blocks.
    # For the 3x3x3 case, the BCRL rank is 19.
    # 
    # Let me try a different decomposition strategy:
    # Split the 3x4x8 problem differently.
    # 
    # Actually, let me try to use the Laderman decomposition or other known schemes.
    # 
    # For now, let me implement a simple but potentially better decomposition.
    # 
    # Alternative: Use the fact that we can decompose the 3x4x8 tensor as:
    # - 3 copies of 3x4x(8/?) ... no, that doesn't work cleanly.
    #
    # Let me try: partition A into rows [0:2] and [2:3], B into columns [0:4] and [4:8]
    # 
    # Actually, let me try a different approach: use Strassen more cleverly.
    #
    # The 3x4x8 problem: C[i,j] = sum_k A[i,k] * B[k,j]
    # i in 0..2, k in 0..3, j in 0..7
    #
    # Let me try to use a rank-19 BCRL scheme for 3x3x3.
    # Here's a known rank-19 decomposition from the literature.
    
    # For the 3x3x3 BCRL decomposition, I'll use a scheme based on the following idea:
    # The 19 bilinear forms are:
    
    # Let me index the elements:
    # A[i,k] for i,k in 0..2
    # B[k,j] for k,j in 0..2
    # C[i,j] for i,j in 0..2
    
    a = [a_rows[0]*m+a_cols[0], a_rows[0]*m+a_cols[1], a_rows[0]*m+a_cols[2],
         a_rows[1]*m+a_cols[0], a_rows[1]*m+a_cols[1], a_rows[1]*m+a_cols[2],
         a_rows[2]*m+a_cols[0], a_rows[2]*m+a_cols[1], a_rows[2]*m+a_cols[2]]
    
    b = [b_rows[0]*p+b_cols[0], b_rows[0]*p+b_cols[1], b_rows[0]*p+b_cols[2],
         b_rows[1]*p+b_cols[0], b_rows[1]*p+b_cols[1], b_rows[1]*p+b_cols[2],
         b_rows[2]*p+b_cols[0], b_rows[2]*p+b_cols[1], b_rows[2]*p+b_cols[2]]
    
    c = [c_cols[0]*n+c_rows[0], c_cols[0]*n+c_rows[1], c_cols[0]*n+c_rows[2],
         c_cols[1]*n+c_rows[0], c_cols[1]*n+c_rows[1], c_cols[1]*n+c_rows[2],
         c_cols[2]*n+c_rows[0], c_cols[2]*n+c_rows[1], c_cols[2]*n+c_rows[2]]
    
    # This is getting complex. Let me use a simpler approach for now.
    # I'll fall back to schoolbook for 3x3x3 (rank 9) which is worse than BCRL (rank 19) 
    # but wait, 9 < 19, so schoolbook is actually better for 3x3x3!
    # No wait, that's wrong. The tensor rank of 3x3x3 matrix multiplication is 19,
    # not 9. Schoolbook uses 9 scalar multiplications but that's not the tensor rank.
    # The tensor rank is the minimum number of rank-1 tensors needed.
    # For 3x3x3, the rank is 19 (BCRL).
    
    # Actually, I need to be more careful. Let me just use a known rank-19 scheme.
    # 
    # Here's a simpler approach: I'll use the fact that for the 3x4x8 problem,
    # I can try different partitions and see which gives the best rank.
    
    # For now, let me implement the schoolbook decomposition for 3x3x3 (which uses 9 terms,
    # but each term is a product of sums, so it's not directly rank-9 in the tensor sense).
    # Actually, schoolbook for 3x3x3 is: C[i,j] = sum_k A[i,k]*B[k,j]
    # This can be written as 9 rank-1 tensors: for each (i,k,j), the tensor has 1 at positions
    # (i*m+k, k*p+j, j*n+i) and 0 elsewhere. So rank 9.
    
    # Wait, but that's the same as what we'd get from the trivial decomposition.
    # The BCRL rank 19 is for the full 3x3x3 tensor, which means we need 19 rank-1 terms
    # to decompose it exactly.
    
    # Let me just use the trivial rank-9 decomposition for 3x3x3 for now, and see if
    # the overall rank is better.
    
    # For each (i, k, j) in 0..2:
    for i in range(3):
        for k in range(3):
            for j in range(3):
                U[a[i*3+k], col] = 1.0
                V[b[k*3+j], col] = 1.0
                W[c[j*3+i], col] = 1.0
                col += 1
    
    return col

def _construct_rank_88():
    """
    Construct the rank-88 decomposition using Strassen blocks.
    C = A * B where A is 3x4, B is 4x8, C is 3x8.
    
    Strategy:
    - Rows 0-1 of A (2x4) with all of B (4x8): use Strassen on 2x2x2 sub-blocks
      Split B into 4 column pairs, for each pair split A cols into two 2-col groups
      4 pairs * 2 A-col-groups * 7 = 56 terms
    - Row 2 of A (1x4) with all of B (4x8): trivial 32 terms
    Total: 56 + 32 = 88
    """
    n, m, p = DIMENSIONS  # 3, 4, 8
    rank = 88
    U = np.zeros((n * m, rank), dtype=float)  # (12, 88)
    V = np.zeros((m * p, rank), dtype=float)  # (32, 88)
    W = np.zeros((p * n, rank), dtype=float)  # (24, 88)
    
    col = 0
    
    # Part 1: A[0:2, :] * B[:, :] using Strassen on 2x2x2 blocks
    for j_start in range(0, p, 2):
        for k_start in range(0, m, 2):
            col = _strassen_block(
                U, V, W, col,
                a_rows=[0, 1], a_cols=[k_start, k_start + 1],
                b_rows=[k_start, k_start + 1], b_cols=[j_start, j_start + 1],
                c_rows=[0, 1], c_cols=[j_start, j_start + 1],
                m=m, p=p, n=n
            )
    
    # Part 2: A[2, :] * B[:, :] trivial
    for k in range(m):
        for j in range(p):
            U[2 * m + k, col] = 1.0
            V[k * p + j, col] = 1.0
            W[j * n + 2, col] = 1.0
            col += 1
    
    if col != rank:
        raise Exception(f"Rank mismatch: expected {rank}, got {col}")
    
    valid, error = _verify_decomposition(U, V, W, n, m, p)
    if not valid:
        raise Exception(f"Verification failed with error {error}")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _construct_rank_84():
    """
    Try a rank-84 decomposition using a different partitioning.
    
    Split A into [0:2] and [2:3], B into [0:6] and [6:8].
    - A[0:2] * B[:, 0:6]: 2x4x6, use Strassen on 2x2x2 blocks: 3 col-pairs * 2 col-groups * 7 = 42
    - A[0:2] * B[:, 6:8]: 2x4x2, use Strassen: 1 col-pair * 2 col-groups * 7 = 14
    - A[2:3] * B[:, 0:6]: 1x4x6, schoolbook: 24
    - A[2:3] * B[:, 6:8]: 1x4x2, schoolbook: 8
    Total: 42 + 14 + 24 + 8 = 88. Same as before.
    
    Let me try another partition:
    - A[0:2] * B[:, :]: 2x4x8, Strassen: 4 col-pairs * 2 col-groups * 7 = 56
    - A[2:3] * B[:, 0:6]: 1x4x6, schoolbook: 24
    - A[2:3] * B[:, 6:8]: 1x4x2, schoolbook: 8
    Total: 56 + 24 + 8 = 88. Still 88.
    
    How about using a 3x3x3 BCRL decomposition?
    - A[0:3, 0:3] * B[0:3, 0:8]: 3x3x8
      Split B into [0:3], [3:6], [6:8]:
      - 3x3x3 (BCRL rank 19) + 3x3x3 (BCRL rank 19) + 3x3x2 (schoolbook 18) = 56
    - A[0:3, 3:4] * B[3:4, 0:8]: 3x1x8, schoolbook: 24
    Total: 56 + 24 = 80.
    
    Let me implement this.
    """
    n, m, p = DIMENSIONS  # 3, 4, 8
    # Estimate rank: 56 + 24 = 80
    rank = 80
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Part 1: A[0:3, 0:3] * B[0:3, :]
    # Split B[0:3, :] into B[0:3, 0:3], B[0:3, 3:6], B[0:3, 6:8]
    
    # Sub-part 1a: A[0:3, 0:3] * B[0:3, 0:3] -> C[0:3, 0:3], rank 19 (BCRL)
    # For now, use schoolbook (rank 9) as a placeholder, then improve
    # Actually, let's use the trivial rank-9 decomposition
    for i in range(3):
        for k in range(3):
            for j in range(3):
                U[i * m + k, col] = 1.0
                V[k * p + j, col] = 1.0
                W[j * n + i, col] = 1.0
                col += 1
    
    # Sub-part 1b: A[0:3, 0:3] * B[0:3, 3:6] -> C[0:3, 3:6], rank 9 (schoolbook)
    for i in range(3):
        for k in range(3):
            for j in range(3):
                U[i * m + k, col] = 1.0
                V[k * p + (j + 3), col] = 1.0
                W[(j + 3) * n + i, col] = 1.0
                col += 1
    
    # Sub-part 1c: A[0:3, 0:3] * B[0:3, 6:8] -> C[0:3, 6:8], rank 6 (schoolbook)
    for i in range(3):
        for k in range(3):
            for j in range(2):
                U[i * m + k, col] = 1.0
                V[k * p + (j + 6), col] = 1.0
                W[(j + 6) * n + i, col] = 1.0
                col += 1
    
    # Part 2: A[0:3, 3:4] * B[3:4, :] -> C[0:3, :], rank 24 (schoolbook)
    for i in range(3):
        for j in range(8):
            U[i * m + 3, col] = 1.0
            V[3 * p + j, col] = 1.0
            W[j * n + i, col] = 1.0
            col += 1
    
    if col != rank:
        raise Exception(f"Rank mismatch: expected {rank}, got {col}")
    
    valid, error = _verify_decomposition(U, V, W, n, m, p)
    if not valid:
        raise Exception(f"Verification failed with error {error}")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Try the rank-80 solution first
    try:
        result = _construct_rank_84()
        return result
    except Exception as e:
        print(f"Rank-80 solution failed: {e}")
    
    # Fall back to rank-88
    result = _construct_rank_88()
    return result
