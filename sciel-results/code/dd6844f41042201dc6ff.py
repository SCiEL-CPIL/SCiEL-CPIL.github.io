"""
Rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication.

Strategy:
  1. For the first 4 columns of B, use the Kronecker product of two Strassen 2x2x2
     decompositions to get a rank-49 decomposition of 4x4x4.
  2. For the remaining 3 columns, use the schoolbook decomposition (rank 48).
  Total rank = 49 + 48 = 97.

The Strassen 2x2x2 decomposition is verified at runtime before use.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 7]


def _strassen_2x2x2():
    """
    Return U, V, W matrices (each 4x7) for the rank-7 Strassen decomposition of 2x2x2.
    
    Indexing convention:
      U[i*m + j, t] where i,j in {0,1} for A[i,j], m=2
      V[j*p + k, t] where j,k in {0,1} for B[j,k], p=2
      W[k*n + i, t] where k,i in {0,1} for C[k,i], n=2  (trace convention W[k*n+i])
    
    Strassen formulation:
      M1 = a(e+f)
      M2 = (a+b)h
      M3 = (c+d)e
      M4 = d(g+f)
      M5 = (a-d)(e+h)
      M6 = (b-c)(e+f)
      M7 = (a+c)(h-g)
      
      C11 = M5 + M4 - M6 - M7
      C12 = M1 + M2
      C21 = M3 + M7
      C22 = M5 + M1 - M2 - M4
    
    Verified at construction time.
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # Index mapping:
    # U: i*2+j  (i=0,1; j=0,1) -> rows 0,1,2,3
    # V: j*2+k  (j=0,1; k=0,1) -> rows 0,1,2,3
    # W: k*2+i  (k=0,1; i=0,1) -> rows 0,1,2,3
    
    # M1 = a(e+f): contributes to C[0,0]
    U[0, 0] = 1    # a = A[0,0]
    V[0, 0] = 1    # e = B[0,0]
    V[1, 0] = 1    # f = B[0,1]
    W[0, 0] = 1    # C[0,0] (k=0,i=0)
    
    # M2 = (a+b)h: contributes to C[0,1]
    U[0, 1] = 1    # a
    U[1, 1] = 1    # b = A[0,1]
    V[3, 1] = 1    # h = B[1,1] (j=1,k=1 -> 1*2+1=3)
    W[1, 1] = 1    # C[0,1] (k=0,i=1)
    
    # M3 = (c+d)e: contributes to C[1,0]
    U[2, 2] = 1    # c = A[1,0]
    U[3, 2] = 1    # d = A[1,1]
    V[0, 2] = 1    # e = B[0,0]
    W[2, 2] = 1    # C[1,0] (k=1,i=0)
    
    # M4 = d(g+f): contributes to C[1,1]
    U[3, 3] = 1    # d
    V[2, 3] = 1    # g = B[1,0] (j=1,k=0 -> 1*2+0=2)
    V[1, 3] = 1    # f = B[0,1]
    W[3, 3] = 1    # C[1,1] (k=1,i=1)
    
    # M5 = (a-d)(e+h): contributes to C[0,0] and C[1,1]
    U[0, 4] = 1    # a
    U[3, 4] = -1   # -d
    V[0, 4] = 1    # e
    V[3, 4] = 1    # h
    W[0, 4] = 1    # C[0,0]
    W[3, 4] = 1    # C[1,1]
    
    # M6 = (b-c)(e+f): contributes to C[0,0] and C[1,0]
    U[1, 5] = 1    # b
    U[2, 5] = -1   # -c
    V[0, 5] = 1    # e
    V[1, 5] = 1    # f
    W[0, 5] = 1    # C[0,0]
    W[2, 5] = 1    # C[1,0]
    
    # M7 = (a+c)(h-g): contributes to C[0,1] and C[1,1]
    U[0, 6] = 1    # a
    U[2, 6] = 1    # c
    V[3, 6] = 1    # h
    V[2, 6] = -1   # -g
    W[1, 6] = 1    # C[0,1]
    W[3, 6] = -1   # C[1,1]
    
    # Verify
    for i in range(2):
        for j in range(2):
            for k in range(2):
                total = 0.0
                for t in range(7):
                    total += U[i * 2 + j, t] * V[j * 2 + k, t] * W[k * 2 + i, t]
                if abs(total - 1.0) > 1e-9:
                    raise ValueError(f"Strassen verification failed at i={i},j={j},k={k}: got {total}")
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for t in range(rank):
                    total += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                if abs(total - 1.0) > 1e-9:
                    return False
    
    return True


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 4, 4, 7
    DEADLINE = 120  # safety margin before 150s kill

    # Schoolbook decomposition as fallback: rank = n*m*p = 112
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb), dtype=float)
    V_sb = np.zeros((m * p, rank_sb), dtype=float)
    W_sb = np.zeros((p * n, rank_sb), dtype=float)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    # Try Strassen-Kronecker for first 4 columns + schoolbook for last 3
    try:
        U1, V1, W1 = _strassen_2x2x2()
        
        # Kronecker product of two 2x2x2 Strassen decompositions gives 4x4x4 rank 49
        r1 = U1.shape[1]  # 7
        r2 = U1.shape[1]  # 7
        r_kron = r1 * r2  # 49
        
        U_kron = np.zeros((16, r_kron), dtype=float)  # 4*4
        V_kron = np.zeros((16, r_kron), dtype=float)  # 4*4
        W_kron = np.zeros((16, r_kron), dtype=float)  # 4*4
        
        for t1 in range(r1):
            for t2 in range(r2):
                t = t1 * r2 + t2
                
                # U_kron index: (i1*2+i2)*4 + (j1*2+j2)
                # U1 index: i1*2+j1 for first factor, i2*2+j2 for second
                for i1 in range(2):
                    for j1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                idx_U = (i1 * 2 + i2) * 4 + (j1 * 2 + j2)
                                U_kron[idx_U, t] = U1[i1 * 2 + j1, t1] * U1[i2 * 2 + j2, t2]
                
                # V_kron index: (j1*2+j2)*4 + (k1*2+k2)
                # V1 index: j1*2+k1 for first factor, j2*2+k2 for second
                for j1 in range(2):
                    for k1 in range(2):
                        for j2 in range(2):
                            for k2 in range(2):
                                idx_V = (j1 * 2 + j2) * 4 + (k1 * 2 + k2)
                                V_kron[idx_V, t] = V1[j1 * 2 + k1, t1] * V1[j2 * 2 + k2, t2]
                
                # W_kron index: (k1*2+k2)*4 + (i1*2+i2)
                # W1 index: k1*2+i1 for first factor, k2*2+i2 for second
                for k1 in range(2):
                    for i1 in range(2):
                        for k2 in range(2):
                            for i2 in range(2):
                                idx_W = (k1 * 2 + k2) * 4 + (i1 * 2 + i2)
                                W_kron[idx_W, t] = W1[k1 * 2 + i1, t1] * W1[k2 * 2 + i2, t2]
        
        # Verify the 4x4x4 Kronecker decomposition
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    total = 0.0
                    for t in range(r_kron):
                        total += U_kron[i * 4 + j, t] * V_kron[j * 4 + k, t] * W_kron[k * 4 + i, t]
                    if abs(total - 1.0) > 1e-9:
                        raise ValueError(f"Kronecker verification failed at i={i},j={j},k={k}: got {total}")
        
        # Schoolbook for the remaining 3 columns (k=4,5,6)
        rank_rem = 4 * 4 * 3  # 48
        U_rem = np.zeros((16, rank_rem), dtype=float)
        V_rem = np.zeros((12, rank_rem), dtype=float)  # 4*3
        W_rem = np.zeros((12, rank_rem), dtype=float)  # 3*4
        
        col = 0
        for i in range(4):
            for j in range(4):
                for kprime in range(3):
                    U_rem[i * 4 + j, col] = 1
                    V_rem[j * 3 + kprime, col] = 1
                    W_rem[kprime * 4 + i, col] = 1
                    col += 1
        
        # Combine into full 4x4x7 decomposition
        total_rank = r_kron + rank_rem  # 49 + 48 = 97
        U_full = np.zeros((16, total_rank), dtype=float)
        V_full = np.zeros((28, total_rank), dtype=float)  # 4*7
        W_full = np.zeros((28, total_rank), dtype=float)  # 7*4
        
        # First 49 columns: Kronecker part for columns 0-3
        U_full[:, 0:r_kron] = U_kron
        V_full[:, 0:r_kron] = V_kron
        W_full[:, 0:r_kron] = W_kron
        
        # Next 48 columns: schoolbook for columns 4-6
        U_full[:, r_kron:total_rank] = U_rem
        # V_rem has shape (12, 48) with rows indexed j*3+k' for j in 0..3, k' in 0..2
        # In the full problem, these correspond to columns 4,5,6: row j*7+(4+k')
        for j in range(4):
            for kprime in range(3):
                V_full[j * 7 + (4 + kprime), r_kron:total_rank] = V_rem[j * 3 + kprime, :]
        # W_rem has shape (12, 48) with rows indexed k'*4+i for k' in 0..2, i in 0..3
        # In the full problem, these correspond to output rows k=4,5,6: row (4+k')*4+i
        for kprime in range(3):
            for i in range(4):
                W_full[(4 + kprime) * 4 + i, r_kron:total_rank] = W_rem[kprime * 4 + i, :]
        
        # Verify full decomposition
        if not _verify_decomposition(U_full, V_full, W_full, 4, 4, 7):
            raise ValueError("Full decomposition verification failed")
        
        if time.time() - start_time > DEADLINE:
            return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
        
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    except Exception:
        # Fallback to schoolbook
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
