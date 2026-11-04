"""Matrix multiplication 3x4x8 rank-minimal decomposition search."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def _verify_safe(U, V, W, n=3, m=4, p=8, tol=1e-6):
    """
    Safely verify the decomposition using vectorized operations.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    Target: T[i,j,k] = 1 for all i,j,k.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    if U_arr.shape[1] != V_arr.shape[1] or U_arr.shape[1] != W_arr.shape[1]:
        return False
    rank = U_arr.shape[1]
    
    # Reshape for vectorized computation
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    T_recon = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    if not np.allclose(T_recon, 1.0, atol=tol):
        return False
    return True

def _construct_rank_88():
    """
    Construct a rank-88 decomposition.
    Split A (3x4) into A0 (1x4) and A12 (2x4).
    A0 x B (4x8): Schoolbook, rank 32.
    A12 x B (4x8): Split A12 into two 2x2 blocks (left/right).
                   Split B into four 2x2 blocks (top/bottom, left/right).
                   8 sub-problems of 2x2x2, each Strassen (rank 7).
                   Total rank 56.
    Grand total: 32 + 56 = 88.
    """
    n, m, p = 3, 4, 8
    rank = 88
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: i=0, schoolbook 1x4x8
    for j in range(4):
        for k in range(8):
            U[0 * m + j, col] = 1.0
            V[j * p + k, col] = 1.0
            W[k * n + 0, col] = 1.0
            col += 1
            
    # Part 2: i=1,2 using Strassen on 2x2 blocks
    # Strassen matrices for 2x2x2 (rank 7)
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[C00, C01], [C10, C11]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    strassen_U = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]).T  # shape (4, 7)
    
    strassen_V = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]).T  # shape (4, 7)
    
    strassen_W = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]).T  # shape (4, 7)
    
    for k_block in range(4):
        k_start = k_block * 2
        
        # Sub-problem 1: A_L (cols 0-1) * B_TL (rows 0-1, cols k_start..k_start+1)
        for t in range(7):
            U[4, col] += strassen_U[0, t]
            U[5, col] += strassen_U[1, t]
            U[8, col] += strassen_U[2, t]
            U[9, col] += strassen_U[3, t]
            
            V[k_start, col] += strassen_V[0, t]
            V[k_start+1, col] += strassen_V[1, t]
            V[p+k_start, col] += strassen_V[2, t]
            V[p+k_start+1, col] += strassen_V[3, t]
            
            W[k_start*n + 1, col] += strassen_W[0, t]
            W[(k_start+1)*n + 1, col] += strassen_W[1, t]
            W[k_start*n + 2, col] += strassen_W[2, t]
            W[(k_start+1)*n + 2, col] += strassen_W[3, t]
            
            col += 1
            
        # Sub-problem 2: A_R (cols 2-3) * B_BL (rows 2-3, cols k_start..k_start+1)
        for t in range(7):
            U[6, col] += strassen_U[0, t]
            U[7, col] += strassen_U[1, t]
            U[10, col] += strassen_U[2, t]
            U[11, col] += strassen_U[3, t]
            
            V[2*p+k_start, col] += strassen_V[0, t]
            V[2*p+k_start+1, col] += strassen_V[1, t]
            V[3*p+k_start, col] += strassen_V[2, t]
            V[3*p+k_start+1, col] += strassen_V[3, t]
            
            W[k_start*n + 1, col] += strassen_W[0, t]
            W[(k_start+1)*n + 1, col] += strassen_W[1, t]
            W[k_start*n + 2, col] += strassen_W[2, t]
            W[(k_start+1)*n + 2, col] += strassen_W[3, t]
            
            col += 1

    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Fallback: Schoolbook
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_schoolbook
    
    # Try Rank 88
    if time.time() < deadline:
        try:
            U_88, V_88, W_88 = _construct_rank_88()
            if _verify_safe(U_88, V_88, W_88):
                if U_88.shape[1] < best_rank:
                    best_result = {"U": U_88.tolist(), "V": V_88.tolist(), "W": W_88.tolist()}
                    best_rank = U_88.shape[1]
        except Exception:
            pass
            
    return best_result
