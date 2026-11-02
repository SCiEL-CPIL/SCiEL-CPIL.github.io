"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    
    for i in range(n):
        for k in range(p):
            M = np.zeros((m, m))
            for r in range(R):
                w_val = W[k * n + i, r]
                if w_val == 0: continue
                u_vec = U[i * m : (i + 1) * m, r]
                v_col = V[k::p, r]
                M += w_val * np.outer(u_vec, v_col)
            
            if not np.allclose(M, np.eye(m), atol=1e-9):
                return False
    return True

def _construct_rank28():
    """
    Construct a rank-28 decomposition for 2x4x4 matrix multiplication using
    Strassen's algorithm on four 2x2x2 sub-blocks.
    
    A is 2x4, B is 4x4, C is 2x4.
    Split: A = [A0 | A1] (each 2x2), B = [[B00,B01],[B10,B11]] (each 2x2)
    C[:,0:2] = A0*B00 + A1*B10
    C[:,2:4] = A0*B01 + A1*B11
    Each 2x2*2x2 uses Strassen (rank 7), total rank = 28.
    """
    n, m, p = 2, 4, 4
    R = 28
    U = np.zeros((n * m, R))  # 8 x 28: U[i*m+j, r]
    V = np.zeros((m * p, R))  # 16 x 28: V[j*p+k, r]
    W = np.zeros((p * n, R))  # 8 x 28: W[k*n+i, r]
    
    def add_strassen(u_idx, v_idx, w_idx, col_offset):
        """
        Add 7 Strassen rank-1 terms.
        u_idx: [u00, u01, u10, u11] - row indices in U
        v_idx: [v00, v01, v10, v11] - row indices in V  
        w_idx: [w00, w01, w10, w11] - row indices in W
        col_offset: starting column in U, V, W
        """
        u00, u01, u10, u11 = u_idx
        v00, v01, v10, v11 = v_idx
        w00, w01, w10, w11 = w_idx
        
        # M1 = (u00+u11)*(v00+v11)
        c = col_offset
        U[u00, c] += 1; U[u11, c] += 1
        V[v00, c] += 1; V[v11, c] += 1
        W[w00, c] += 1; W[w11, c] += 1
        
        # M2 = (u10+u11)*v00
        c = col_offset + 1
        U[u10, c] += 1; U[u11, c] += 1
        V[v00, c] += 1
        W[w10, c] += 1
        
        # M3 = u00*(v01-v11)
        c = col_offset + 2
        U[u00, c] += 1
        V[v01, c] += 1; V[v11, c] -= 1
        W[w01, c] += 1
        
        # M4 = u11*(v10-v00)
        c = col_offset + 3
        U[u11, c] += 1
        V[v10, c] += 1; V[v00, c] -= 1
        W[w00, c] += 1; W[w10, c] += 1
        
        # M5 = (u00+u01)*v11
        c = col_offset + 4
        U[u00, c] += 1; U[u01, c] += 1
        V[v11, c] += 1
        W[w00, c] -= 1; W[w01, c] += 1
        
        # M6 = (u10-u00)*(v00+v01)
        c = col_offset + 5
        U[u10, c] += 1; U[u00, c] -= 1
        V[v00, c] += 1; V[v01, c] += 1
        W[w11, c] += 1
        
        # M7 = (u01-u11)*(v10+v11)
        c = col_offset + 6
        U[u01, c] += 1; U[u11, c] -= 1
        V[v10, c] += 1; V[v11, c] += 1
        W[w00, c] += 1
    
    # U indices for A (2x4): A[i,j] -> U[i*4+j]
    # A0 = A[:,0:2]: A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[4], A[1,1]=U[5]
    # A1 = A[:,2:4]: A[0,2]=U[2], A[0,3]=U[3], A[1,2]=U[6], A[1,3]=U[7]
    
    # V indices for B (4x4): B[j,k] -> V[j*4+k]
    # B00 = B[0:2,0:2]: B[0,0]=V[0], B[0,1]=V[1], B[1,0]=V[4], B[1,1]=V[5]
    # B01 = B[0:2,2:4]: B[0,2]=V[2], B[0,3]=V[3], B[1,2]=V[6], B[1,3]=V[7]
    # B10 = B[2:4,0:2]: B[2,0]=V[8], B[2,1]=V[9], B[3,0]=V[12], B[3,1]=V[13]
    # B11 = B[2:4,2:4]: B[2,2]=V[10], B[2,3]=V[11], B[3,2]=V[14], B[3,3]=V[15]
    
    # W indices for C (2x4): C[i,k] -> W[k*2+i]
    # C[:,0:2]: C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[2], C[1,1]=W[3]
    # C[:,2:4]: C[0,2]=W[4], C[0,3]=W[5], C[1,2]=W[6], C[1,3]=W[7]
    
    # Block P = A0 * B00 -> contributes to C[:,0:2]
    u_P = [0, 1, 4, 5]
    v_P = [0, 1, 4, 5]
    w_P = [0, 1, 2, 3]
    add_strassen(u_P, v_P, w_P, 0)
    
    # Block Q = A1 * B10 -> contributes to C[:,0:2]
    u_Q = [2, 3, 6, 7]
    v_Q = [8, 9, 12, 13]
    w_Q = [0, 1, 2, 3]
    add_strassen(u_Q, v_Q, w_Q, 7)
    
    # Block R = A0 * B01 -> contributes to C[:,2:4]
    u_R = [0, 1, 4, 5]
    v_R = [2, 3, 6, 7]
    w_R = [4, 5, 6, 7]
    add_strassen(u_R, v_R, w_R, 14)
    
    # Block S = A1 * B11 -> contributes to C[:,2:4]
    u_S = [2, 3, 6, 7]
    v_S = [10, 11, 14, 15]
    w_S = [4, 5, 6, 7]
    add_strassen(u_S, v_S, w_S, 21)
    
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

def _construct_schoolbook(n, m, p):
    """Return the schoolbook (trivial) decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    result = _construct_rank28()
    if result is not None:
        return result
    
    return _construct_schoolbook(n, m, p)
# EVOLVE-BLOCK-END
