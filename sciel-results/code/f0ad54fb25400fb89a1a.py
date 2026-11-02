# EVOLVE-BLOCK-START

import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def _verify_fast(U, V, W, n, m, p):
    """Faster verification using numpy."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    expected = np.ones((n, m, p))
    return np.allclose(recon, expected, atol=1e-9)

def _prune_rank(U, V, W):
    """Remove zero columns and try to find linearly dependent columns."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    rank = U.shape[1]
    if rank == 0:
        return U.tolist(), V.tolist(), W.tolist()
    
    # Remove zero columns
    keep = []
    for c in range(rank):
        if np.any(U[:, c] != 0) or np.any(V[:, c] != 0) or np.any(W[:, c] != 0):
            keep.append(c)
    
    if len(keep) < rank:
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
        rank = len(keep)
    
    # Try to remove linearly dependent columns using SVD-based approach
    # For each column, check if it's in the span of previous kept columns
    # This is approximate; we'll use a simple approach
    if rank > 0:
        keep2 = [0]
        U_kept = U[:, keep2]
        V_kept = V[:, keep2]
        W_kept = W[:, keep2]
        
        for c in range(1, rank):
            # Check if column c is linearly dependent with kept columns
            # Simple check: see if the "combined" vector is in span
            # We'll skip this for now as it's complex; just return what we have
            keep2.append(c)
            U_kept = np.column_stack([U_kept, U[:, c]])
            V_kept = np.column_stack([V_kept, V[:, c]])
            W_kept = np.column_stack([W_kept, W[:, c]])
        
        U, V, W = U_kept, V_kept, W_kept
    
    return U.tolist(), V.tolist(), W.tolist()

def schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def strassen_2x2():
    """
    Return Strassen decomposition for 2x2 * 2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)*(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3: A00*(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4: A11*(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5: (A00+A10)*B11
    U[0][4] = 1; U[2][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (A01-A10)*(B00+B10)
    U[1][5] = 1; U[2][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1
    
    # M7: (A01-A11)*(B01+B11)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def schoolbook_sub(na, mb, pb):
    """Schoolbook decomposition for na x mb * mb x pb."""
    rank = na * mb * pb
    U = [[0] * rank for _ in range(na * mb)]
    V = [[0] * rank for _ in range(mb * pb)]
    W = [[0] * rank for _ in range(pb * na)]
    col = 0
    for i in range(na):
        for j in range(mb):
            for k in range(pb):
                U[i*mb+j][col] = 1
                V[j*pb+k][col] = 1
                W[k*na+i][col] = 1
                col += 1
    return U, V, W

def strassen_2x4_x_4x2():
    """
    Decomposition for 2x4 * 4x2 using Strassen on two 2x2x2 blocks.
    A is 2x4 = [A1 A2] where A1, A2 are 2x2
    B is 4x2 = [B1; B2] where B1, B2 are 2x2
    C = A1*B1 + A2*B2, each 2x2x2 uses Strassen (rank 7)
    Total rank: 14
    U: (8, 14), V: (8, 14), W: (4, 14)
    U indices: [i*4+j] for i in 0..1, j in 0..3
    V indices: [j*2+k] for j in 0..3, k in 0..1
    W indices: [k*2+i] for k in 0..1, i in 0..1
    """
    U_str, V_str, W_str = strassen_2x2()
    r = 7
    
    U = [[0] * (2*r) for _ in range(8)]
    V = [[0] * (2*r) for _ in range(8)]
    W = [[0] * (2*r) for _ in range(4)]
    
    # First block: A1 (rows 0-1, cols 0-1) * B1 (rows 0-1, cols 0-1)
    for c in range(r):
        for i in range(2):
            for j in range(2):
                U[i*4+j][c] += U_str[i*2+j][c]
        for j in range(2):
            for k in range(2):
                V[j*2+k][c] += V_str[j*2+k][c]
        for k in range(2):
            for i in range(2):
                W[k*2+i][c] += W_str[k*2+i][c]
    
    # Second block: A2 (rows 0-1, cols 2-3) * B2 (rows 2-3, cols 0-1)
    for c in range(r):
        for i in range(2):
            for j in range(2):
                U[i*4+(j+2)][r+c] += U_str[i*2+j][c]
        for j in range(2):
            for k in range(2):
                V[(j+2)*2+k][r+c] += V_str[j*2+k][c]
        for k in range(2):
            for i in range(2):
                W[k*2+i][r+c] += W_str[k*2+i][c]
    
    return U, V, W

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication.
    Uses Strassen for 2x2*2x2 sub-blocks to reduce rank.
    """
    n, m, p = DIMENSIONS  # 3, 4, 4
    deadline = time.time() + 140  # Safety margin
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Build decomposition using Strassen for 2x4*4x2 blocks
    # C[0:2, 0:2] = A[0:2,:] * B[:,0:2] -> 2x4 * 4x2, rank 14
    # C[0:2, 2:4] = A[0:2,:] * B[:,2:4] -> 2x4 * 4x2, rank 14
    # C[2:3, 0:2] = A[2:3,:] * B[:,0:2] -> 1x4 * 4x2, rank 4
    # C[2:3, 2:4] = A[2:3,:] * B[:,2:4] -> 1x4 * 4x2, rank 4
    # Total: 14 + 14 + 4 + 4 = 36
    
    rank_total = 36
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    col = 0
    
    def add_sub(U_sub, V_sub, W_sub, i_off, j_off, k_off, na, mb, pb, col_start):
        nonlocal col
        r = len(U_sub[0])
        for c in range(r):
            for i_prime in range(na):
                for j_prime in range(mb):
                    u_val = U_sub[i_prime*mb + j_prime][c]
                    if u_val != 0:
                        U[(i_off + i_prime)*m + (j_off + j_prime)][col_start + c] += u_val
            for j_prime in range(mb):
                for k_prime in range(pb):
                    v_val = V_sub[j_prime*pb + k_prime][c]
                    if v_val != 0:
                        V[(j_off + j_prime)*p + (k_off + k_prime)][col_start + c] += v_val
            for k_prime in range(pb):
                for i_prime in range(na):
                    w_val = W_sub[k_prime*na + i_prime][c]
                    if w_val != 0:
                        W[(k_off + k_prime)*n + (i_off + i_prime)][col_start + c] += w_val
        col = col_start + r
    
    # C[0:2, 0:2]: A[0:2, 0:4] * B[0:4, 0:2]
    U_2442, V_2442, W_2442 = strassen_2x4_x_4x2()
    add_sub(U_2442, V_2442, W_2442, 0, 0, 0, 2, 4, 2, col)
    
    # C[0:2, 2:4]: A[0:2, 0:4] * B[0:4, 2:4]
    U_2442, V_2442, W_2442 = strassen_2x4_x_4x2()
    add_sub(U_2442, V_2442, W_2442, 0, 0, 2, 2, 4, 2, col)
    
    # C[2:3, 0:2]: A[2:3, 0:4] * B[0:4, 0:2]
    U_1442, V_1442, W_1442 = schoolbook_sub(1, 4, 2)
    add_sub(U_1442, V_1442, W_1442, 2, 0, 0, 1, 4, 2, col)
    
    # C[2:3, 2:4]: A[2:3, 0:4] * B[0:4, 2:4]
    U_1442, V_1442, W_1442 = schoolbook_sub(1, 4, 2)
    add_sub(U_1442, V_1442, W_1442, 2, 0, 2, 1, 4, 2, col)
    
    # Prune and verify
    U_pruned, V_pruned, W_pruned = _prune_rank(U, V, W)
    
    if _verify_fast(U_pruned, V_pruned, W_pruned, n, m, p):
        return {"U": U_pruned, "V": V_pruned, "W": W_pruned}
    
    # Fallback to schoolbook
    return best_result

# EVOLVE-BLOCK-END
