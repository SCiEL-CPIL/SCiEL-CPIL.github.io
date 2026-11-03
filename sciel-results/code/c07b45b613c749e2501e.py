import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140  # seconds, safety margin before 150s kill

    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n=2, m=4, p=8

    # Strassen decomposition for 2x2 matrix multiplication (rank 7)
    U_local = [[0]*7 for _ in range(4)]
    V_local = [[0]*7 for _ in range(4)]
    W_local = [[0]*7 for _ in range(4)]

    # r=0 (M1): (a00+a11)(b00+b11)
    U_local[0][0] = 1; U_local[3][0] = 1
    V_local[0][0] = 1; V_local[3][0] = 1
    W_local[0][0] = 1  # C00
    W_local[3][0] = 1  # C11

    # r=1 (M2): (a10+a11)b00
    U_local[2][1] = 1; U_local[3][1] = 1
    V_local[0][1] = 1
    W_local[2][1] = 1   # C10
    W_local[3][1] = -1  # C11

    # r=2 (M3): a00(b01-b11)
    U_local[0][2] = 1
    V_local[1][2] = 1; V_local[3][2] = -1
    W_local[1][2] = 1   # C01
    W_local[3][2] = 1   # C11

    # r=3 (M4): a11(b10-b00)
    U_local[3][3] = 1
    V_local[2][3] = 1; V_local[0][3] = -1
    W_local[0][3] = 1   # C00
    W_local[2][3] = 1   # C10

    # r=4 (M5): (a00+a01)b11
    U_local[0][4] = 1; U_local[1][4] = 1
    V_local[3][4] = 1
    W_local[0][4] = -1  # C00
    W_local[1][4] = 1   # C01

    # r=5 (M6): (a10-a00)(b00+b01)
    U_local[2][5] = 1; U_local[0][5] = -1
    V_local[0][5] = 1; V_local[1][5] = 1
    W_local[3][5] = 1   # C11

    # r=6 (M7): (a01-a11)(b10+b11)
    U_local[1][6] = 1; U_local[3][6] = -1
    V_local[2][6] = 1; V_local[3][6] = 1
    W_local[0][6] = 1   # C00

    total_rank = 56

    U = [[0] * total_rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * total_rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * total_rank for _ in range(p * n)]  # 16 x 56

    # Decompose 2x4 x 4x8 into 8 blocks of 2x2 x 2x2
    for j in range(4):
        for s in range(2):
            base_r = (j * 2 + s) * 7

            a_map = [
                0 * 4 + (2 * s + 0),
                0 * 4 + (2 * s + 1),
                1 * 4 + (2 * s + 0),
                1 * 4 + (2 * s + 1),
            ]

            b_map = [
                (2 * s + 0) * 8 + (2 * j + 0),
                (2 * s + 0) * 8 + (2 * j + 1),
                (2 * s + 1) * 8 + (2 * j + 0),
                (2 * s + 1) * 8 + (2 * j + 1),
            ]

            c_map = [
                (2 * j + 0) * 2 + 0,
                (2 * j + 1) * 2 + 0,
                (2 * j + 0) * 2 + 1,
                (2 * j + 1) * 2 + 1,
            ]

            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la][l]
                    if val != 0:
                        U[a_map[la]][r] += val
                for lb in range(4):
                    val = V_local[lb][l]
                    if val != 0:
                        V[b_map[lb]][r] += val
                for lc in range(4):
                    val = W_local[lc][l]
                    if val != 0:
                        W[c_map[lc]][r] += val

    # Check for linear dependencies among the 56 rank-1 terms
    # Construct the tensor matrix M of size (n*m*p) x 56
    # Vectorized construction of M
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # M[idx, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # idx = (i*m + j)*p + k
    # i in 0..n-1, j in 0..m-1, k in 0..p-1
    
    # We can reshape U, V, W to facilitate broadcasting?
    # U is (n*m, R). We want U[i, j, r] where index is i*m+j.
    # V is (m*p, R). We want V[j, k, r] where index is j*p+k.
    # W is (p*n, R). We want W[k, i, r] where index is k*n+i.
    
    # Reshape U to (n, m, R)
    U_3d = U_np.reshape(n, m, total_rank)
    # Reshape V to (m, p, R)
    V_3d = V_np.reshape(m, p, total_rank)
    # Reshape W to (p, n, R)
    W_3d = W_np.reshape(p, n, total_rank)
    
    # We want M[i, j, k, r] = U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
    # This is a 4D tensor. We can compute this using einsum or broadcasting.
    # U_3d shape: (n, m, R)
    # V_3d shape: (m, p, R)
    # W_3d shape: (p, n, R)
    
    # Product P[i, j, k, r]
    # U term depends on i, j, r
    # V term depends on j, k, r
    # W term depends on k, i, r
    
    # Let's expand dimensions for broadcasting
    # U_3d: (n, m, 1, R) -> (n, m, 1, R)
    # V_3d: (1, m, p, R) -> (1, m, p, R)
    # W_3d: (p, n, 1, R) -> (p, n, 1, R) -- wait, W_3d is (p, n, R).
    # We need W[k, i, r].
    # W_3d indices: k (0..p-1), i (0..n-1), r.
    
    # U_3d[i, j, r]
    # V_3d[j, k, r]
    # W_3d[k, i, r]
    
    # Result M_flat: (n*m*p, R)
    
    # Using einsum is efficient
    # M_4d = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d) ? No, R is a dimension.
    # We want to keep R.
    # M_4d[i, j, k, r] = U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
    
    # np.einsum('ijr,jkr,kir->ijk', ...) sums over r? No, we want to keep r.
    # The indices i, j, k are free. r is a free index.
    # But U_3d has r as last dim.
    # So: np.einsum('ijr,jkr,kir->ijk', ...) is wrong because r is repeated in inputs but not in output?
    # If r is in output, it must be in all inputs or some?
    # Actually, we want M[i, j, k, r].
    # M[i, j, k, r] = U[i, j, r] * V[j, k, r] * W[k, i, r]
    # This is a diagonal-like operation over r? No, r is the same for all three.
    # So we can just broadcast.
    
    U_exp = U_3d[:, :, np.newaxis, :] # (n, m, 1, R)
    V_exp = V_3d[np.newaxis, :, :, :] # (1, m, p, R)
    W_exp = W_3d[:, np.newaxis, :, :] # (p, 1, n, R) -- Wait, W_3d is (p, n, R).
    # We need W[k, i, r].
    # W_3d[k, i, r]
    # So W_exp should be (p, n, 1, R) ? No, we need to align with i, j, k.
    # i is dim 0 of result. j is dim 1. k is dim 2.
    # U_exp: (n, m, 1, R) -> i, j, k, r
    # V_exp: (1, m, p, R) -> i, j, k, r
    # W_exp: (p, n, 1, R) -> k, i, r. We need to permute W_exp to (n, p, 1, R) or similar?
    # W_3d[k, i, r].
    # We want to broadcast W_3d[k, i, r] to (n, m, p, R) where the k index is the 3rd dim and i is the 1st dim.
    # W_3d shape (p, n, R).
    # W_3d.T shape (n, p, R).
    # W_3d.T[i, k, r].
    # So W_exp = W_3d.T[:, np.newaxis, :, :] # (n, 1, p, R) -> i, j, k, r
    
    M_4d = U_exp * V_exp * W_exp
    # M_4d shape (n, m, p, R)
    M = M_4d.reshape(n * m * p, total_rank)

    # Compute the rank of M using SVD for numerical stability
    try:
        # Use SVD to determine rank
        U_svd, S, Vt = np.linalg.svd(M, full_matrices=False)
        # Rank is number of singular values above tolerance
        tol = np.finfo(float).eps * max(M.shape) * S[0] if len(S) > 0 else 0
        rank_M = np.sum(S > tol)
        
        # If rank_M < total_rank, we can reduce the number of terms
        if rank_M < total_rank and rank_M > 0:
            # Find a basis of columns using QR with column pivoting
            Q, R, P = np.linalg.qr(M, mode='complete', pivoting=True)
            # The first rank_M columns of M[:, P[:rank_M]] form a basis
            basis_cols = P[:rank_M]
            
            # Construct new U, V, W with only the basis columns
            U_new = [[U[i][r] for r in basis_cols] for i in range(n*m)]
            V_new = [[V[i][r] for r in basis_cols] for i in range(m*p)]
            W_new = [[W[i][r] for r in basis_cols] for i in range(p*n)]
            
            U = U_new
            V = V_new
            W = W_new
            total_rank = rank_M
    except Exception:
        # If SVD or QR fails, keep the original 56 terms
        pass

    if time.time() - start_time > deadline:
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
