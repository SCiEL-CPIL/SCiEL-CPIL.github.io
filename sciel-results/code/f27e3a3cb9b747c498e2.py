"""Matrix multiplication 4x4x6 via hybrid Strassen decomposition."""
DIMENSIONS = [4, 4, 6]

import time
import numpy as np

def _get_strassen_2x2_factors():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices: U[i*2+j], V[j*2+k], W[k*2+i].
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    prods = [
        ([(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)]),
        ([(1,0,1.0), (1,1,1.0)], [(0,0,1.0)], [(1,0,1.0), (1,1,-1.0)]),
        ([(0,0,1.0)], [(0,1,1.0), (1,1,-1.0)], [(0,1,1.0), (1,1,1.0)]),
        ([(1,1,1.0)], [(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (1,0,1.0)]),
        ([(0,0,1.0), (0,1,1.0)], [(1,1,1.0)], [(0,0,-1.0), (0,1,1.0)]),
        ([(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (0,1,1.0)], [(1,1,1.0)]),
        ([(0,1,1.0), (1,1,-1.0)], [(1,0,1.0), (1,1,1.0)], [(0,0,1.0)]),
    ]

    for m_idx, (Ub, Vb, Wb) in enumerate(prods):
        for (i, j, c) in Ub:
            U[i * m + j][m_idx] += c
        for (j, k, c) in Vb:
            V[j * p + k][m_idx] += c
        for (i, k, c) in Wb:
            W[k * n + i][m_idx] += c

    return U, V, W

def _verify(U, V, W, n, m, p, rank):
    """Check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k)."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    err = np.max(np.abs(T - 1.0))
    return err

def _rank_reduce(U, V, W, n, m, p, rank, tol=1e-9):
    """
    Try to reduce the rank by identifying linearly dependent rank-1 terms.
    A rank-1 term c is redundant if its tensor contribution is a linear
    combination of other terms. We check this via the matricized tensor.
    
    Returns reduced U, V, W with potentially lower rank.
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # Build matrix where each column c is the flattened tensor T_c
    flat_size = n * m * p
    Tmat = np.zeros((flat_size, rank))
    
    for c in range(rank):
        Tc = np.einsum('ij,jk,ki->ijk', 
                       Ua[:, c].reshape(n, m),
                       Va[:, c].reshape(m, p),
                       Wa[:, c].reshape(p, n))
        Tmat[:, c] = Tc.flatten()
    
    # Greedy: keep columns that are not in the span of previously kept columns
    U_reduced = []
    V_reduced = []
    W_reduced = []
    Q = None  # Orthonormal basis for the span
    num_kept = 0
    
    for c in range(rank):
        if time.time() > 100:  # safety margin
            break
            
        col = Tmat[:, c:c+1]
        col_norm = np.linalg.norm(col)
        
        if col_norm < tol:
            # Zero column, skip
            continue
        
        if num_kept == 0:
            Q = col / col_norm
            U_reduced.append(Ua[:, c])
            V_reduced.append(Va[:, c])
            W_reduced.append(Wa[:, c])
            num_kept += 1
        else:
            # Project onto current Q
            coeff = Q.T @ col  # (num_kept, 1)
            residual = col - Q @ coeff
            resid_norm = np.linalg.norm(residual)
            
            if resid_norm <= tol * max(1.0, col_norm):
                # In span - skip
                continue
            else:
                q_new = residual / resid_norm
                Q = np.hstack([Q, q_new])
                U_reduced.append(Ua[:, c])
                V_reduced.append(Va[:, c])
                W_reduced.append(Wa[:, c])
                num_kept += 1
    
    if num_kept < rank and num_kept > 0:
        U_new = np.column_stack(U_reduced)
        V_new = np.column_stack(V_reduced)
        W_new = np.column_stack(W_reduced)
        
        # Verify
        Ur = U_new.reshape(n, m, num_kept)
        Vr = V_new.reshape(m, p, num_kept)
        Wr = W_new.reshape(p, n, num_kept)
        T_check = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
        err = np.max(np.abs(T_check - 1.0))
        
        if err < 1e-6:
            return U_new.tolist(), V_new.tolist(), W_new.tolist(), num_kept
    
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    t0 = time.time()
    deadline = t0 + 110.0

    U_s, V_s, W_s = _get_strassen_2x2_factors()
    
    # Total rank: 49 (for 4x4x4) + 28 (for 4x4x2) = 77
    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442
    
    U = [[0.0] * total_rank for _ in range(n * m)]  # 16 x 77
    V = [[0.0] * total_rank for _ in range(m * p)]  # 24 x 77
    W = [[0.0] * total_rank for _ in range(p * n)]  # 24 x 77
    
    # 1. 4x4x4 Strassen (columns 0-48)
    # Kronecker product of 2x2 Strassen with itself
    for i in range(4):
        I = i // 2
        i0 = i % 2
        for j in range(4):
            J = j // 2
            j0 = j % 2
            for t in range(7):
                for s in range(7):
                    col = 7 * t + s
                    val = U_s[2 * I + J][t] * U_s[2 * i0 + j0][s]
                    if val != 0:
                        U[i * 4 + j][col] += val
                        
    for j in range(4):
        J = j // 2
        j0 = j % 2
        for k in range(4):
            K = k // 2
            k0 = k % 2
            for t in range(7):
                for s in range(7):
                    col = 7 * t + s
                    val = V_s[2 * J + K][t] * V_s[2 * j0 + k0][s]
                    if val != 0:
                        V[j * 6 + k][col] += val
                        
    for k in range(4):
        K = k // 2
        k0 = k % 2
        for i in range(4):
            I = i // 2
            i0 = i % 2
            for t in range(7):
                for s in range(7):
                    col = 7 * t + s
                    val = W_s[2 * K + I][t] * W_s[2 * k0 + i0][s]
                    if val != 0:
                        W[k * 4 + i][col] += val

    # 2. 4x4x2 Strassen-block (columns 49-76)
    # Split B into B_top (rows 0-1, cols 4-5) and B_bot (rows 2-3, cols 4-5)
    # C_top = A_00 * B_top + A_01 * B_bot
    # C_bot = A_10 * B_top + A_11 * B_bot
    # Each product is 2x2 * 2x2, using Strassen (rank 7)
    
    products = [
        (0, 0, 0),  # I=0, J=0, is_bot=0 (B_top)
        (0, 1, 1),  # I=0, J=1, is_bot=1 (B_bot)
        (1, 0, 0),  # I=1, J=0, is_bot=0 (B_top)
        (1, 1, 1),  # I=1, J=1, is_bot=1 (B_bot)
    ]
    
    for p_idx, (I, J, is_bot) in enumerate(products):
        col_offset = rank_444 + p_idx * 7
        
        # A block: A_IJ, rows 2I..2I+1, cols 2J..2J+1
        for i0 in range(2):
            for j0 in range(2):
                i = 2 * I + i0
                j = 2 * J + j0
                for s in range(7):
                    val = U_s[2 * i0 + j0][s]
                    if val != 0:
                        U[i * 4 + j][col_offset + s] += val
                        
        # B block: rows 2*is_bot..2*is_bot+1, cols 4..5
        for j0 in range(2):
            for k0 in range(2):
                j_global = 2 * is_bot + j0
                k_global = 4 + k0
                for s in range(7):
                    val = V_s[2 * j0 + k0][s]
                    if val != 0:
                        V[j_global * 6 + k_global][col_offset + s] += val
                        
        # C block: rows 2I..2I+1, cols 4..5
        for i0 in range(2):
            for k0 in range(2):
                i_global = 2 * I + i0
                k_global = 4 + k0
                for s in range(7):
                    val = W_s[2 * k0 + i0][s]
                    if val != 0:
                        W[k_global * 4 + i_global][col_offset + s] += val

    err = _verify(U, V, W, n, m, p, total_rank)
    
    if err > 1e-6:
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

    # Try rank reduction by removing linearly dependent rank-1 terms
    if time.time() < deadline - 5.0:
        U_red, V_red, W_red, new_rank = _rank_reduce(U, V, W, n, m, p, total_rank)
        if new_rank < total_rank:
            return {"U": U_red, "V": V_red, "W": W_red}

    return {"U": U, "V": V, "W": W}
