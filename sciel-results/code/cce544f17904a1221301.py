import numpy as np
import time

DIMENSIONS = [5, 5, 5]
DEADLINE_SECONDS = 140.0

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][col] = 1
                V[k * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, tol=1e-10):
    rng = np.random.RandomState(42)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    C_expected = A @ B
    
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            U_slice = U_arr[i*m : i*m+m, :]
            V_indices = np.arange(m) * p + j
            V_slice = V_arr[V_indices, :]
            term = np.sum(U_slice * V_slice, axis=0)
            W_val = W_arr[j * n + i, :]
            C_computed[i, j] = np.dot(term, W_val)
    
    if np.max(np.abs(C_computed - C_expected)) > tol:
        return False
    return True

def _strassen_2x2x2():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A11(B21-B22)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A22(B12-B11)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[3, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[1, 6] = 1; W[3, 6] = -1
    
    return U.tolist(), V.tolist(), W.tolist()

def _build_strassen_5x5x5():
    n, m, p = 5, 5, 5
    block_sizes = [2, 2, 1]
    i_offsets = [0, 2, 4]
    k_offsets = [0, 2, 4]
    j_offsets = [0, 2, 4]
    
    sU, sV, sW = _strassen_2x2x2()
    
    total_rank = 0
    for a in range(3):
        for b in range(3):
            for c in range(3):
                sa, sb, sc = block_sizes[a], block_sizes[b], block_sizes[c]
                if sa == 2 and sb == 2 and sc == 2:
                    total_rank += 7
                else:
                    total_rank += sa * sb * sc
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for a in range(3):
        for b in range(3):
            for c in range(3):
                sa, sb, sc = block_sizes[a], block_sizes[b], block_sizes[c]
                io, ko, jo = i_offsets[a], k_offsets[b], j_offsets[c]
                
                if sa == 2 and sb == 2 and sc == 2:
                    for rr in range(7):
                        for li in range(2):
                            for lk in range(2):
                                val = sU[li * 2 + lk][rr]
                                if val != 0:
                                    U[(io + li) * m + (ko + lk)][col] += val
                        for lk in range(2):
                            for lj in range(2):
                                val = sV[lk * 2 + lj][rr]
                                if val != 0:
                                    V[(ko + lk) * p + (jo + lj)][col] += val
                        for lj in range(2):
                            for li in range(2):
                                val = sW[lj * 2 + li][rr]
                                if val != 0:
                                    W[(jo + lj) * n + (io + li)][col] += val
                        col += 1
                else:
                    for li in range(sa):
                        for lk in range(sb):
                            for lj in range(sc):
                                U[(io + li) * m + (ko + lk)][col] = 1
                                V[(ko + lk) * p + (jo + lj)][col] = 1
                                W[(jo + lj) * n + (io + li)][col] = 1
                                col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    result = _build_strassen_5x5x5()
    
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    result = _schoolbook(n, m, p)
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    return result
