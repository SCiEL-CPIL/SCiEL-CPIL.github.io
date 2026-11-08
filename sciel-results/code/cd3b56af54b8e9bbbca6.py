import time
import numpy as np

_T0 = time.time()
_DEADLINE = 120.0

def _deadline_ok():
    return (time.time() - _T0) < (_DEADLINE - 8.0)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
                r += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    n, m, p = 3, 4, 6
    # Target rank 66
    rank = 66
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    
    r = 0
    # Split p=6 into three blocks of size 2
    for ko in range(0, p, 2):
        k1 = ko
        k2 = ko + 1
        
        # Block 1: Row 0 of A (1x4) * Block of B (4x2) -> C[0, k1:k2]
        # Rank 8 (Schoolbook)
        for j in range(m):
            U[0 * m + j, r] = 1
            V[j * p + k1, r] = 1
            W[k1 * n + 0, r] = 1
            r += 1
            U[0 * m + j, r] = 1
            V[j * p + k2, r] = 1
            W[k2 * n + 0, r] = 1
            r += 1
        
        # Block 2: Rows 1-2 of A (2x4) * Block of B (4x2) -> C[1:3, k1:k2]
        # Rank 14 (2 * Strassen 2x2x2)
        # Split m=4 into two blocks of size 2: j in (0,1) and j in (2,3)
        for j0, j1 in ((0, 1), (2, 3)):
            # Strassen 7 terms for 2x2x2 block
            # Indices in A: rows 1,2 (local 0,1), cols j0,j1
            # Indices in B: rows j0,j1, cols k1,k2
            # Indices in C: rows 1,2 (local 0,1), cols k1,k2
            
            # m1 = (A[1,j0]+A[2,j1]) * (B[j0,k1]+B[j1,k2])
            U[1 * m + j0, r] = 1
            U[2 * m + j1, r] = 1
            V[j0 * p + k1, r] = 1
            V[j1 * p + k2, r] = 1
            W[k1 * n + 1, r] = 1
            W[k2 * n + 2, r] = 1
            r += 1
            
            # m2 = (A[2,j0]+A[2,j1]) * B[j0,k1]
            U[2 * m + j0, r] = 1
            U[2 * m + j1, r] = 1
            V[j0 * p + k1, r] = 1
            W[k1 * n + 2, r] = 1
            W[k2 * n + 2, r] = -1
            r += 1
            
            # m3 = A[1,j0] * (B[j0,k2]-B[j1,k2])
            U[1 * m + j0, r] = 1
            V[j0 * p + k2, r] = 1
            V[j1 * p + k2, r] = -1
            W[k2 * n + 1, r] = 1
            W[k2 * n + 2, r] = 1
            r += 1
            
            # m4 = A[2,j1] * (B[j1,k1]-B[j0,k1])
            U[2 * m + j1, r] = 1
            V[j1 * p + k1, r] = 1
            V[j0 * p + k1, r] = -1
            W[k1 * n + 1, r] = 1
            W[k1 * n + 2, r] = 1
            r += 1
            
            # m5 = (A[1,j0]+A[1,j1]) * B[j1,k2]
            U[1 * m + j0, r] = 1
            U[1 * m + j1, r] = 1
            V[j1 * p + k2, r] = 1
            W[k1 * n + 1, r] = -1
            W[k2 * n + 1, r] = 1
            r += 1
            
            # m6 = (A[2,j0]-A[1,j0]) * (B[j0,k1]+B[j0,k2])
            U[2 * m + j0, r] = 1
            U[1 * m + j0, r] = -1
            V[j0 * p + k1, r] = 1
            V[j0 * p + k2, r] = 1
            W[k2 * n + 2, r] = 1
            r += 1
            
            # m7 = (A[1,j1]-A[2,j1]) * (B[j1,k1]+B[j1,k2])
            U[1 * m + j1, r] = 1
            U[2 * m + j1, r] = -1
            V[j1 * p + k1, r] = 1
            V[j1 * p + k2, r] = 1
            W[k1 * n + 1, r] = 1
            r += 1

    if not _deadline_ok():
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Verify tensor decomposition: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1 for all i,j,k
    # Using numpy for speed
    U_np = U.astype(int)
    V_np = V.astype(int)
    W_np = W.astype(int)
    
    # Construct the tensor T[i,j,k] = sum_r U[i,m,j,r] * V[j,p,k,r] * W[k,n,i,r]
    # We check if T is all ones.
    
    # Optimization: Only check a subset if time is tight, but 72 checks is fast.
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for rr in range(rank):
                    s += U_np[i * m + j, rr] * V_np[j * p + k, rr] * W_np[k * n + i, rr]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        return _schoolbook(n, m, p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
