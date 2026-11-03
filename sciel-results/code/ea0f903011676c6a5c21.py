"""Program to find rank-minimal exact tensor decomposition for 4x5 * 5x5 matrix multiplication."""

import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return U, V, W for 2x2 * 2x2 -> 2x2 with rank 7."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11 + A22)(B11 + B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[2, 0] = 1; W[1, 0] = -1; W[3, 0] = -1
    
    # M2 = (A21 + A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A11(B21 - B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[0, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1
    
    # M4 = A22(B12 - B21)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[1, 3] = 1
    
    # M5 = (A11 + A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[3, 4] = -1
    
    # M6 = (A21 - A11)(B11 + B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12 - A22)(B21 + B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[2, 6] = 1
    
    return U, V, W


def _validate(U, V, W, tol=1e-9):
    n, m, p = 4, 5, 5
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True


def construct():
    n, m, p = 4, 5, 5
    t0 = time.time()
    
    # Strategy: Split inner dimension m=5 into 2+2+1
    # Split output n=4 into 2+2, output p=5 into 2+2+1
    
    # For each combination of (i_block, j_block, k_block):
    # i_block in {0,1} (size 2 each), j_block in {0,1,2} (sizes 2,2,1), k_block in {0,1,2} (sizes 2,2,1)
    
    # If i_size=2, j_size=2, k_size=2: use Strassen (rank 7)
    # Otherwise: schoolbook (rank = i_size * j_size * k_size)
    
    i_splits = [(0, 2), (2, 2)]  # (offset, size)
    j_splits = [(0, 2), (2, 2), (4, 1)]
    k_splits = [(0, 2), (2, 2), (4, 1)]
    
    U_list = []
    V_list = []
    W_list = []
    
    for i_off, i_sz in i_splits:
        for j_off, j_sz in j_splits:
            for k_off, k_sz in k_splits:
                if i_sz == 2 and j_sz == 2 and k_sz == 2:
                    # Use Strassen
                    Us, Vs, Ws = _strassen_2x2x2()
                    # Map to full dimensions
                    U_block = np.zeros((n*m, 7))
                    V_block = np.zeros((m*p, 7))
                    W_block = np.zeros((p*n, 7))
                    for r in range(7):
                        for ii in range(2):
                            for jj in range(2):
                                U_block[(i_off+ii)*m + (j_off+jj), r] = Us[ii*2+jj, r]
                        for jj in range(2):
                            for kk in range(2):
                                V_block[(j_off+jj)*p + (k_off+kk), r] = Vs[jj*2+kk, r]
                        for kk in range(2):
                            for ii in range(2):
                                W_block[(k_off+kk)*n + (i_off+ii), r] = Ws[kk*2+ii, r]
                    U_list.append(U_block)
                    V_list.append(V_block)
                    W_list.append(W_block)
                else:
                    # Schoolbook
                    rank = i_sz * j_sz * k_sz
                    U_block = np.zeros((n*m, rank))
                    V_block = np.zeros((m*p, rank))
                    W_block = np.zeros((p*n, rank))
                    c = 0
                    for ii in range(i_sz):
                        for jj in range(j_sz):
                            for kk in range(k_sz):
                                U_block[(i_off+ii)*m + (j_off+jj), c] = 1.0
                                V_block[(j_off+jj)*p + (k_off+kk), c] = 1.0
                                W_block[(k_off+kk)*n + (i_off+ii), c] = 1.0
                                c += 1
                    U_list.append(U_block)
                    V_list.append(V_block)
                    W_list.append(W_block)
    
    # Combine
    total_rank = sum(U.shape[1] for U in U_list)
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    c = 0
    for U_b, V_b, W_b in zip(U_list, V_list, W_list):
        r = U_b.shape[1]
        U[:, c:c+r] = U_b
        V[:, c:c+r] = V_b
        W[:, c:c+r] = W_b
        c += r
    
    # Validate
    if not _validate(U, V, W):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
