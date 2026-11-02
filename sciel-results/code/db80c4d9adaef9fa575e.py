import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2 matrix multiplication (rank 7).
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen's 7 products for 2x2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # p1 = (a+d)(e+h), p2 = (c+d)e, p3 = a(f-h), p4 = d(g-e)
    # p5 = (a+b)h, p6 = (c-a)(e+f), p7 = (b-d)(g+h)
    # C = [[p1+p4-p5+p7, p3+p5], [p2+p4, p1-p2+p3+p6]]
    
    # U[i*2+j, r]: row index = i*2+j (i=row of A, j=col of A)
    # V[j*2+k, r]: row index = j*2+k (j=row of B, k=col of B)  
    # W[k*2+i, r]: row index = k*2+i (k=col of B, i=row of A)
    
    # Let me index: a=A[0,0]=U[0], b=A[0,1]=U[1], c=A[1,0]=U[2], d=A[1,1]=U[3]
    # e=B[0,0]=V[0], f=B[0,1]=V[1], g=B[1,0]=V[2], h=B[1,1]=V[3]
    # C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[2], C[1,1]=W[3]
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # p1 = (a+d)(e+h): U contributes a+d to rows 0,3; V contributes e+h to rows 0,3
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # C[0,0] += p1, C[1,1] += p1
    
    # p2 = (c+d)e: U contributes c+d to rows 2,3; V contributes e to row 0
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # C[1,0] += p2
    
    # p3 = a(f-h): U contributes a to row 0; V contributes f-h to rows 1,3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1  # C[0,1] += p3
    
    # p4 = d(g-e): U contributes d to row 3; V contributes g-e to rows 2,0
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1  # C[0,0] += p4, C[1,0] += p4
    
    # p5 = (a+b)h: U contributes a+b to rows 0,1; V contributes h to row 3
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = -1; W[0][4] = -1  # C[0,1] -= p5, C[0,0] -= p5
    
    # p6 = (c-a)(e+f): U contributes c-a to rows 2,0; V contributes e+f to rows 0,1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C[1,1] += p6
    
    # p7 = (b-d)(g+h): U contributes b-d to rows 1,3; V contributes g+h to rows 2,3
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C[0,0] += p7
    
    return U, V, W

def _schoolbook_block(n, m, p):
    """Schoolbook decomposition for n x m x p. Returns U(nm, nmp), V(mp, nmp), W(pn, nmp)."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _combine_blocks(block_list, n, m, p):
    """Combine a list of (n_b, m_b, p_b, U_b, V_b, W_b, i_off, j_off, k_off) into full U, V, W.
    Each block computes C[i_off:i_off+n_b, k_off:k_off+p_b] += A[i_off:i_off+n_b, j_off:j_off+m_b] @ B[j_off:j_off+m_b, k_off:k_off+p_b]
    """
    total_rank = sum(len(b[3][0]) for b in block_list)
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col_offset = 0
    for (n_b, m_b, p_b, U_b, V_b, W_b, i_off, j_off, k_off) in block_list:
        rank_b = len(U_b[0])
        for i in range(n_b):
            for j in range(m_b):
                for r in range(rank_b):
                    U[(i_off+i)*m + (j_off+j)][col_offset + r] = U_b[i*m_b+j][r]
        for j in range(m_b):
            for k in range(p_b):
                for r in range(rank_b):
                    V[(j_off+j)*p + (k_off+k)][col_offset + r] = V_b[j*p_b+k][r]
        for k in range(p_b):
            for i in range(n_b):
                for r in range(rank_b):
                    W[(k_off+k)*n + (i_off+i)][col_offset + r] = W_b[k*n_b+i][r]
        col_offset += rank_b
    
    return U, V, W

def _verify_decomposition(U, V, W, tol=1e-10):
    n, m, p = DIMENSIONS
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U_np[a, :] * V_np[b, :] * W_np[c, :])
                err = abs(val - 1.0)
                max_err = max(max_err, err)
                if max_err > tol:
                    return False
    
    # Check off-diagonal
    for a in range(n * m):
        i1 = a // m
        j1 = a % m
        for b in range(m * p):
            j2 = b // p
            k1 = b % p
            if j1 != j2:
                continue
            for c in range(p * n):
                k2 = c // n
                i2 = c % n
                if j1 != j2 or k1 != k2 or i1 != i2:
                    val = np.sum(U_np[a, :] * V_np[b, :] * W_np[c, :])
                    err = abs(val)
                    max_err = max(max_err, err)
                    if max_err > tol:
                        return False
    
    return max_err <= tol

def _construct_rank68():
    """Construct rank-68 decomposition for 3x4x6.
    Split: rows 0-1 (2 rows) use Strassen-enhanced, row 2 (1 row) uses schoolbook.
    For the 2x4x6 part: split into 2x2x3 blocks (4 blocks), each 2x2x3 has rank... 
    Actually, let's split differently.
    
    Strategy: Split middle dimension j into j=0,1 (Strassen on 2x2x2 sub-blocks) and j=2,3 (schoolbook).
    For j=0,1: A is 3x2, B is 2x6. Split further: for each pair of rows in A and 2 cols in B, use Strassen.
    
    Simpler: Split A into A_top (2x4) and A_bot (1x4).
    Split B into B_left (4x3) and B_right (4x3).
    
    A_top @ B_left: 2x4x3. Split j into 2+2: two 2x2x3 blocks.
    2x2x3: schoolbook rank 12. But we can do better? 2x2x2 + 2x2x1 = 7 + 4 = 11? No, 2x2x1 is rank 4.
    Actually 2x2x3: split k into 2+1: 2x2x2 (Strassen rank 7) + 2x2x1 (schoolbook rank 4) = 11.
    So 2x4x3 = 2 * 11 = 22.
    
    A_top @ B_right: same, 22.
    A_bot @ B_left: 1x4x3. Split j into 2+2: two 1x2x3 blocks, each schoolbook rank 6. Total 12.
    A_bot @ B_right: same, 12.
    
    Total: 22 + 22 + 12 + 12 = 68. Yes!
    """
    n, m, p = 3, 4, 6
    
    blocks = []
    
    # A_top (rows 0-1) @ B_left (cols 0-2): 2x4x3
    # Split j=0,1 and j=2,3
    # j=0,1: 2x2x3 -> split k=0,1 (Strassen 7) + k=2 (schoolbook 4)
    U_b, V_b, W_b = _strassen_2x2x2()
    blocks.append((2, 2, 2, U_b, V_b, W_b, 0, 0, 0))  # i=0-1, j=0-1, k=0-1
    
    U_b, V_b, W_b = _schoolbook_block(2, 2, 1)
    blocks.append((2, 2, 1, U_b, V_b, W_b, 0, 0, 2))  # i=0-1, j=0-1, k=2
    
    U_b, V_b, W_b = _strassen_2x2x2()
    blocks.append((2, 2, 2, U_b, V_b, W_b, 0, 2, 0))  # i=0-1, j=2-3, k=0-1
    
    U_b, V_b, W_b = _schoolbook_block(2, 2, 1)
    blocks.append((2, 2, 1, U_b, V_b, W_b, 0, 2, 2))  # i=0-1, j=2-3, k=2
    
    # A_top (rows 0-1) @ B_right (cols 3-5): 2x4x3
    U_b, V_b, W_b = _strassen_2x2x2()
    blocks.append((2, 2, 2, U_b, V_b, W_b, 0, 0, 3))  # i=0-1, j=0-1, k=3-4
    
    U_b, V_b, W_b = _schoolbook_block(2, 2, 1)
    blocks.append((2, 2, 1, U_b, V_b, W_b, 0, 0, 5))  # i=0-1, j=0-1, k=5
    
    U_b, V_b, W_b = _strassen_2x2x2()
    blocks.append((2, 2, 2, U_b, V_b, W_b, 0, 2, 3))  # i=0-1, j=2-3, k=3-4
    
    U_b, V_b, W_b = _schoolbook_block(2, 2, 1)
    blocks.append((2, 2, 1, U_b, V_b, W_b, 0, 2, 5))  # i=0-1, j=2-3, k=5
    
    # A_bot (row 2) @ B_left (cols 0-2): 1x4x3
    # Split j=0,1 and j=2,3: two 1x2x3 blocks, schoolbook
    U_b, V_b, W_b = _schoolbook_block(1, 2, 3)
    blocks.append((1, 2, 3, U_b, V_b, W_b, 2, 0, 0))  # i=2, j=0-1, k=0-2
    
    U_b, V_b, W_b = _schoolbook_block(1, 2, 3)
    blocks.append((1, 2, 3, U_b, V_b, W_b, 2, 2, 0))  # i=2, j=2-3, k=0-2
    
    # A_bot (row 2) @ B_right (cols 3-5): 1x4x3
    U_b, V_b, W_b = _schoolbook_block(1, 2, 3)
    blocks.append((1, 2, 3, U_b, V_b, W_b, 2, 0, 3))  # i=2, j=0-1, k=3-5
    
    U_b, V_b, W_b = _schoolbook_block(1, 2, 3)
    blocks.append((1, 2, 3, U_b, V_b, W_b, 2, 2, 3))  # i=2, j=2-3, k=3-5
    
    U, V, W = _combine_blocks(blocks, n, m, p)
    return {"U": U, "V": V, "W": W}

def _construct_schoolbook():
    n, m, p = DIMENSIONS
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

def construct():
    deadline = time.time() + 140
    
    # Try the rank-68 structured decomposition
    try:
        result = _construct_rank68()
        if _verify_decomposition(result['U'], result['V'], result['W']):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _construct_schoolbook()
