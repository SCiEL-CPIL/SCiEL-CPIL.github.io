import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def _verify_fast(U, V, W, n=2, m=4, p=8, trials=50, tol=1e-9):
    rank = len(U[0]) if U else 0
    if rank == 0:
        return True
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)

    U_r = U_mat.reshape(n, m, rank)
    V_r = V_mat.reshape(m, p, rank)
    W_r = W_mat.reshape(p, n, rank)

    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        factor_A = np.einsum('ij,ijc->ic', A, U_r)
        factor_B = np.einsum('jk,jkc->kc', B, V_r)
        C_computed = np.einsum('ic,kc,kic->ik', factor_A, factor_B, W_r)
        if np.max(np.abs(C_computed - C_expected)) > tol:
            return False
    return True

def _build_block_strassen():
    n, m, p = 2, 4, 8
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    U_s = [[1,0,0,1],[0,0,1,1],[1,0,0,0],[0,0,0,1],[1,1,0,0],[0,0,1,-1],[0,1,0,1]]
    V_s = [[1,0,0,1],[0,0,0,1],[1,0,-1,0],[0,-1,0,1],[0,0,1,0],[1,1,0,0],[0,0,1,1]]
    W_s = [[1,0,0,1,-1,0,1],[0,0,1,0,1,0,0],[0,1,0,0,0,1,0],[1,-1,1,0,0,1,0]]

    col_off = 0
    for block in range(4):
        for f in range(7):
            col = col_off + f
            for r in range(4):
                U[r, col] = U_s[f][r]
                j = r // 2
                k = 2 * block + r % 2
                V[j * 8 + k, col] = V_s[f][r]
                i = r // 2
                W[k * 2 + i, col] = W_s[r][f]
        col_off += 7

    for block in range(4):
        for f in range(7):
            col = col_off + f
            for r in range(4):
                U[2 + r, col] = U_s[f][r]
                j = 2 + r // 2
                k = 2 * block + r % 2
                V[j * 8 + k, col] = V_s[f][r]
                i = r // 2
                W[k * 2 + i, col] = W_s[r][f]
        col_off += 7

    return U.tolist(), V.tolist(), W.tolist()

def _compute_unfolding_rank(U, V, W, n, m, p):
    # Unfold the tensor T (n x m x p) along the first mode?
    # T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
    # This is not a simple outer product sum.
    # The tensor is T = sum_c U_c \otimes V_c \otimes W_c? No.
    # The rank-1 tensor is U[:,:,c] * V[:,:,c] * W[:,:,c]?
    # No, the decomposition is C = sum_c (A U_c) (B V_c) W_c?
    # The tensor T is defined by T[i,j,k] such that C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]?
    # No, the standard tensor for matrix multiplication is T[i,j,k] = delta_{i,j} delta_{j,k}? No.
    # The bilinear form is <A, U_c> <B, V_c> W_c.
    # This corresponds to a tensor T such that T[i,j,k] = U[i,j,c] V[j,k,c] W[k,i,c]?
    # This is not a rank-1 tensor in the standard sense.
    # The standard tensor rank is for T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c].
    # This is the "matrix multiplication tensor".
    # Its rank is the minimum c.
    # To find a lower bound, we can look at the matrix unfolding.
    # Flatten T to a matrix of size (n*m) x (m*p) x (p*n)?
    # The tensor T is in R^{n x m x p}.
    # Wait, the indices are i (n), j (m), k (p).
    # T[i,j,k] is the coefficient of A[i,j] B[j,k] in C[i,k]?
    # No, C[i,k] = sum_j A[i,j] B[j,k].
    # So T[i,j,k] = 1 if we are summing over j?
    # The tensor is T[i,j,k] = delta_{i,i} delta_{j,j} delta_{k,k}?
    # The tensor is T[i,j,k] = 1 for all i,j,k? No.
    # The tensor is T[i,j,k] such that C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k]?
    # This implies T[i,j,k] = 1 if the indices match the multiplication rule.
    # For C[i,k], we sum over j. A[i,j] and B[j,k].
    # So the term is A[i,j] B[j,k].
    # The tensor T has indices (i,j,k).
    # T[i,j,k] = 1.
    # Then C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k].
    # This is the standard matrix multiplication tensor.
    # Its rank is the rank of the matrix multiplication problem.
    
    # Unfolding:
    # Mode-1 unfolding: T_(1) is n x (m*p).
    # T_(1)[i, j*p+k] = T[i,j,k] = 1.
    # This matrix has rank n? No, it's all 1s in blocks.
    # Rank of T_(1) is n.
    # Mode-2 unfolding: T_(2) is m x (n*p).
    # T_(2)[j, i*p+k] = T[i,j,k] = 1.
    # Rank is m.
    # Mode-3 unfolding: T_(3) is p x (n*m).
    # Rank is p.
    # These are lower bounds.
    # n=2, m=4, p=8.
    # Lower bounds: 2, 4, 8.
    # This doesn't help distinguish 56 from 55.
    
    # We need a stronger lower bound.
    # The "border rank" or "approximate rank" might be useful.
    # But for exact rank, we rely on known results or search.
    
    return -1 # Placeholder

def construct():
    start_time = time.time()
    deadline = start_time + 135

    n, m, p = DIMENSIONS

    U_list, V_list, W_list = _build_block_strassen()
    U = np.array(U_list, dtype=float)
    V = np.array(V_list, dtype=float)
    W = np.array(W_list, dtype=float)

    if not _verify_fast(U_list, V_list, W_list, n, m, p, trials=30, tol=1e-9):
        schoolbook_rank = n * m * p
        U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
        V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
        W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}

    return {"U": U_list, "V": V_list, "W": W_list}
