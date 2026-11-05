"""Matrix multiplication 3x5x6 - Strassen-based block decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def _verify(U, V, W, n, m, p):
    """Verify that sum_c U[:,c] * V[:,c] * W[:,c]^T equals the multiplication tensor."""
    rank = len(U[0])
    # Build the full tensor as a matrix of shape (n*m, m*p) and check against expected
    # T[i,j; j2,k] = 1 if j == j2 else 0
    # Actually: tensor T has T[(i,j),(j,k)] = 1 iff middle indices match
    # The decomposition: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] should equal 1 when j=j2
    # More precisely: for output element C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor: T[(i,j),(j,k)] = delta_{j,j} = 1
    # Rank-1: sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Build as a 3D tensor and check
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Sum over c of outer products
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This is a 3D tensor of shape (n, m, p)
    
    # Use einsum or manual
    T = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        # U[:, c] has shape (n*m,), V[:, c] has shape (m*p,), W[:, c] has shape (p*n,)
        u_col = U_arr[:, c].reshape(n, m)
        v_col = V_arr[:, c].reshape(m, p)
        w_col = W_arr[:, c].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u_col, v_col, w_col)
    
    # Expected: T[i,j,k] = 1 for all i,j,k
    expected = np.ones((n, m, p), dtype=float)
    return np.allclose(T, expected, atol=1e-9)

def _strassen_2x2x2():
    """
    Return rank-7 Strassen decomposition for 2x2 matrix multiplication.
    U: shape (4, 7) for A indices (i,j)
    V: shape (4, 7) for B indices (j,k)
    W: shape (4, 7) for C indices (k,i)
    """
    # Strassen 7 multiplications for 2x2x2
    # C = sum_{r=1}^{7} U_r * V_r * W_r^T where W_r[k,i] gives output position
    #
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M7
    #
    # U[i,j,r]: coefficient of A[i,j] in M_r
    # V[j,k,r]: coefficient of B[j,k] in M_r
    # W[k,i,r]: coefficient of C[k,i] in M_r (note: W[k*n+i, r])
    
    U = [[0]*7 for _ in range(4)]  # indices: i*2+j
    V = [[0]*7 for _ in range(4)]  # indices: j*2+k
    W = [[0]*7 for _ in range(4)]  # indices: k*2+i
    
    # M1 = (A00+A11)(B00+B11)
    U[0*2+0][0] = 1  # A00
    U[1*2+1][0] = 1  # A11
    V[0*2+0][0] = 1  # B00
    V[1*2+1][0] = 1  # B11
    # C00 += M1, C11 += M1
    W[0*2+0][0] = 1  # C[0,0] -> k=0,i=0
    W[1*2+1][0] = 1  # C[1,1] -> k=1,i=1
    
    # M2 = (A10+A11)B00
    U[1*2+0][1] = 1  # A10
    U[1*2+1][1] = 1  # A11
    V[0*2+0][1] = 1  # B00
    # C10 += M2
    W[0*2+1][1] = 1  # C[0,1] -> k=0,i=1
    
    # M3 = A00(B01-B11)
    U[0*2+0][2] = 1  # A00
    V[0*2+1][2] = 1  # B01
    V[1*2+1][2] = -1 # B11
    # C01 += M3, C11 -= M3
    W[1*2+0][2] = 1  # C[1,0] -> k=1,i=0
    W[1*2+1][2] = -1 # C[1,1] -> k=1,i=1
    
    # M4 = A11(B10-B00)
    U[1*2+1][3] = 1  # A11
    V[1*2+0][3] = 1  # B10
    V[0*2+0][3] = -1 # B00
    # C10 += M4, C00 -= M4
    W[0*2+1][3] = 1  # C[0,1] -> k=0,i=1
    W[0*2+0][3] = -1 # C[0,0] -> k=0,i=0
    
    # M5 = (A00+A01)B11
    U[0*2+0][4] = 1  # A00
    U[0*2+1][4] = 1  # A01
    V[1*2+1][4] = 1  # B11
    # C01 += M5
    W[1*2+0][4] = 1  # C[1,0] -> k=1,i=0
    
    # M6 = (A10-A00)(B00+B01)
    U[1*2+0][5] = 1  # A10
    U[0*2+0][5] = -1 # A00
    V[0*2+0][5] = 1  # B00
    V[0*2+1][5] = 1  # B01
    # C00 += M6
    W[0*2+0][5] = 1  # C[0,0] -> k=0,i=0
    
    # M7 = (A01-A11)(B10+B11)
    U[0*2+1][6] = 1  # A01
    U[1*2+1][6] = -1 # A11
    V[1*2+0][6] = 1  # B10
    V[1*2+1][6] = 1  # B11
    # C11 += M7
    W[1*2+1][6] = 1  # C[1,1] -> k=1,i=1
    
    return U, V, W

def _block_strassen_decomposition():
    """
    Use Strassen on 2x2 sub-blocks to reduce rank.
    
    A is 3x5, B is 5x6.
    We can partition:
    - A into rows: [2, 1] and cols [2, 2, 1]
    - B into rows: [2, 2, 1] and cols [2, 2, 2]
    
    Actually, let's think about this differently.
    
    For 3x5x6, the schoolbook rank is 90.
    
    We can try to use Strassen on overlapping 2x2 blocks.
    
    A better approach: partition the indices.
    A: i in {0,1,2}, j in {0,1,2,3,4}
    B: j in {0,1,2,3,4}, k in {0,1,2,3,4,5}
    
    Let's partition j into groups: {0,1}, {2,3}, {4}
    i into: {0,1}, {2}
    k into: {0,1}, {2,3}, {4,5}
    
    Then we have blocks:
    A blocks: A[{0,1},{0,1}], A[{0,1},{2,3}], A[{0,1},{4}], A[{2},{0,1}], A[{2},{2,3}], A[{2},{4}]
    B blocks: B[{0,1},{0,1}], B[{0,1},{2,3}], B[{0,1},{4,5}], B[{2,3},{0,1}], B[{2,3},{2,3}], B[{2,3},{4,5}], B[{4},{0,1}], B[{4},{2,3}], B[{4},{4,5}]
    
    Output C blocks: C[{0,1},{0,1}], C[{0,1},{2,3}], C[{0,1},{4,5}], C[{2},{0,1}], C[{2},{2,3}], C[{2},{4,5}]
    
    For each pair of A-block (rows i1, cols j1) and B-block (rows j2, cols k1),
    the contribution to C-block (rows i1, cols k1) is sum over j where j1=j2.
    
    So we need j1 == j2 for the multiplication to contribute.
    
    Block pairs that contribute:
    - A[{0,1},{0,1}] * B[{0,1},{0,1}] -> C[{0,1},{0,1}] (2x2 * 2x2 = 2x2, rank 7 via Strassen)
    - A[{0,1},{2,3}] * B[{2,3},{0,1}] -> C[{0,1},{0,1}] (2x2 * 2x2 = 2x2, rank 7)
    - A[{0,1},{2,3}] * B[{2,3},{2,3}] -> C[{0,1},{2,3}] (2x2 * 2x2 = 2x2, rank 7)
    - A[{0,1},{2,3}] * B[{2,3},{4,5}] -> C[{0,1},{4,5}] (2x2 * 2x2 = 2x2, rank 7)
    - A[{0,1},{4}] * B[{4},{0,1}] -> C[{0,1},{0,1}] (2x1 * 1x2 = 2x2, rank 4 schoolbook)
    - A[{0,1},{4}] * B[{4},{2,3}] -> C[{0,1},{2,3}] (2x1 * 1x2 = 2x2, rank 4)
    - A[{0,1},{4}] * B[{4},{4,5}] -> C[{0,1},{4,5}] (2x1 * 1x2 = 2x2, rank 4)
    - A[{2},{0,1}] * B[{0,1},{0,1}] -> C[{2},{0,1}] (1x2 * 2x2 = 1x2, rank 2)
    - A[{2},{0,1}] * B[{0,1},{2,3}] -> C[{2},{2,3}] (1x2 * 2x2 = 1x2, rank 2)
    - A[{2},{0,1}] * B[{0,1},{4,5}] -> C[{2},{4,5}] (1x2 * 2x2 = 1x2, rank 2)
    - A[{2},{2,3}] * B[{2,3},{0,1}] -> C[{2},{0,1}] (1x2 * 2x2 = 1x2, rank 2)
    - A[{2},{2,3}] * B[{2,3},{2,3}] -> C[{2},{2,3}] (1x2 * 2x2 = 1x2, rank 2)
    - A[{2},{2,3}] * B[{2,3},{4,5}] -> C[{2},{4,5}] (1x2 * 2x2 = 1x2, rank 2)
    - A[{2},{4}] * B[{4},{0,1}] -> C[{2},{0,1}] (1x1 * 1x2 = 1x2, rank 1... actually rank 2 since 1*2)
    - A[{2},{4}] * B[{4},{2,3}] -> C[{2},{2,3}] (1x1 * 1x2 = 1x2, rank 2)
    - A[{2},{4}] * B[{4},{4,5}] -> C[{2},{4,5}] (1x1 * 1x2 = 1x2, rank 2)
    
    Total rank: 4*7 + 3*4 + 6*2 + 3*2 = 28 + 12 + 12 + 6 = 58
    
    Wait, let me recount:
    Strassen 2x2x2 blocks: 4 blocks, each rank 7 = 28
    2x1 * 1x2 blocks: 3 blocks, each rank 4 (schoolbook for 2x1x2) = 12
    1x2 * 2x2 blocks: 6 blocks, each rank 2 (schoolbook for 1x2x2) = 12
    1x1 * 1x2 blocks: 3 blocks, each rank 2 (schoolbook for 1x1x2) = 6
    
    Total: 28 + 12 + 12 + 6 = 58
    
    That's much better than 90!
    """
    n, m, p = 3, 5, 6
    
    U = [[0]*58 for _ in range(n*m)]  # 15 x 58
    V = [[0]*58 for _ in range(m*p)]  # 30 x 58
    W = [[0]*58 for _ in range(p*n)]  # 18 x 58
    
    col = 0
    sU, sV, sW = _strassen_2x2x2()
    
    def add_strassen_block(ai, aj, bj, bk):
        """Add a Strassen 2x2x2 block.
        ai: list of 2 row indices for A
        aj: list of 2 col indices for A (and row indices for B)
        bj: list of 2 row indices for B (same as aj)
        bk: list of 2 col indices for B
        """
        nonlocal col
        for r in range(7):
            # U entries
            for local_i, global_i in enumerate(ai):
                for local_j, global_j in enumerate(aj):
                    if sU[local_i*2 + local_j][r] != 0:
                        U[global_i*m + global_j][col] += sU[local_i*2 + local_j][r]
            # V entries
            for local_j, global_j in enumerate(bj):
                for local_k, global_k in enumerate(bk):
                    if sV[local_j*2 + local_k][r] != 0:
                        V[global_j*p + global_k][col] += sV[local_j*2 + local_k][r]
            # W entries
            for local_k, global_k in enumerate(bk):
                for local_i, global_i in enumerate(ai):
                    if sW[local_k*2 + local_i][r] != 0:
                        W[global_k*n + global_i][col] += sW[local_k*2 + local_i][r]
            col += 1
    
    def add_schoolbook_block(ai, aj, bj, bk):
        """Add a schoolbook block multiplication.
        ai: list of row indices for A
        aj: list of col indices for A (= row indices for B)
        bj: list of row indices for B
        bk: list of col indices for B
        """
        nonlocal col
        for i in ai:
            for j in aj:
                for k in bk:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
    
    # Strassen blocks (2x2 x 2x2 -> 2x2)
    # A rows {0,1}, cols {0,1}; B rows {0,1}, cols {0,1} -> C rows {0,1}, cols {0,1}
    add_strassen_block([0,1], [0,1], [0,1], [0,1])
    # A rows {0,1}, cols {2,3}; B rows {2,3}, cols {0,1} -> C rows {0,1}, cols {0,1}
    add_strassen_block([0,1], [2,3], [2,3], [0,1])
    # A rows {0,1}, cols {2,3}; B rows {2,3}, cols {2,3} -> C rows {0,1}, cols {2,3}
    add_strassen_block([0,1], [2,3], [2,3], [2,3])
    # A rows {0,1}, cols {2,3}; B rows {2,3}, cols {4,5} -> C rows {0,1}, cols {4,5}
    add_strassen_block([0,1], [2,3], [2,3], [4,5])
    
    # 2x1 * 1x2 blocks (schoolbook)
    add_schoolbook_block([0,1], [4], [4], [0,1])
    add_schoolbook_block([0,1], [4], [4], [2,3])
    add_schoolbook_block([0,1], [4], [4], [4,5])
    
    # 1x2 * 2x2 blocks (schoolbook)
    add_schoolbook_block([2], [0,1], [0,1], [0,1])
    add_schoolbook_block([2], [0,1], [0,1], [2,3])
    add_schoolbook_block([2], [0,1], [0,1], [4,5])
    add_schoolbook_block([2], [2,3], [2,3], [0,1])
    add_schoolbook_block([2], [2,3], [2,3], [2,3])
    add_schoolbook_block([2], [2,3], [2,3], [4,5])
    
    # 1x1 * 1x2 blocks (schoolbook)
    add_schoolbook_block([2], [4], [4], [0,1])
    add_schoolbook_block([2], [4], [4], [2,3])
    add_schoolbook_block([2], [4], [4], [4,5])
    
    assert col == 58, f"Expected 58 columns, got {col}"
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    result = _block_strassen_decomposition()
    
    # Verify again
    n, m, p = DIMENSIONS
    if not _verify(result["U"], result["V"], result["W"], n, m, p):
        # Fallback
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    return result
