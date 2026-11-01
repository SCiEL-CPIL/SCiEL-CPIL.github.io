import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    A_dim = n * m  # 8
    B_dim = m * p  # 32
    C_dim = p * n  # 16
    
    rank = 56
    U = np.zeros((A_dim, rank), dtype=int)
    V = np.zeros((B_dim, rank), dtype=int)
    W = np.zeros((C_dim, rank), dtype=int)
    
    col = 0
    
    # Strassen terms for 2x2 * 2x2 -> 2x2
    # Local indices: (r,c) for 2x2 matrix
    strassen_terms = [
        # M1 = (a00+a11)(b00+b11) -> c00 += M1, c11 += M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (a10+a11)b00 -> c10 += M2, c11 -= M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = a00(b01-b11) -> c01 += M3, c11 += M3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = a11(b10-b00) -> c00 += M4, c10 += M4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (a00+a01)b11 -> c00 -= M5, c01 += M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6 = (a10-a00)(b00+b01) -> c11 += M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (a01-a11)(b10+b11) -> c00 += M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # Split j into 2 groups of 2: {0,1} and {2,3}
    # Split k into 4 groups of 2: {0,1}, {2,3}, {4,5}, {6,7}
    for j_group in range(2):
        for k_group in range(4):
            for term in strassen_terms:
                A_coeffs, B_coeffs, C_coeffs = term
                
                for (li, lj), coeff in A_coeffs.items():
                    global_i = li
                    global_j = 2 * j_group + lj
                    U[global_i * m + global_j, col] += coeff
                
                for (lj, lk), coeff in B_coeffs.items():
                    global_j = 2 * j_group + lj
                    global_k = 2 * k_group + lk
                    V[global_j * p + global_k, col] += coeff
                
                for (li, lk), coeff in C_coeffs.items():
                    global_i = li
                    global_k = 2 * k_group + lk
                    W[global_k * n + global_i, col] += coeff
                
                col += 1
    
    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify
    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return result
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)

def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
