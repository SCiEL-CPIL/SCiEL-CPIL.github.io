import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def _verify(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    Texp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                Texp[a, b, c] = 1.0
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    err = np.max(np.abs(T - Texp))
    return err < tol

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Use Strassen-based decomposition:
    # Group j into pairs: (0,1), (2,3)
    # Group k into pairs: (0,1), (2,3), (4,5), and singleton (6)
    # For each j-pair and k-pair: use Strassen (7 mults)
    # For each j-pair and k-singleton: use schoolbook 2x2x1 (4 mults)
    # Total: 2*3*7 + 2*1*4 = 42 + 8 = 50
    
    rank = 50
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        if time.time() > deadline:
            return
        
        # Indices for A (n x m): a = A[i0,j0], b = A[i0,j0+1], c = A[i0+1,j0], d = A[i0+1,j0+1]
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        # Indices for B (m x p): e = B[j0,k0], f = B[j0,k0+1], g = B[j0+1,k0], h = B[j0+1,k0+1]
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        # Indices for C (n x p): c00 = C[i0,k0], c01 = C[i0,k0+1], c10 = C[i0+1,k0], c11 = C[i0+1,k0+1]
        # W is indexed by c = k*n + i
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a+d)(e+h) -> contributes to c00 and c11
        U[a_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1; W[c11_idx, col] += 1
        col += 1
        
        # M2 = (c+d)e -> contributes to c10 and c11
        U[c_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1; W[c11_idx, col] -= 1
        col += 1
        
        # M3 = a(f-h) -> contributes to c01 and c11
        U[a_idx, col] += 1
        V[f_idx, col] += 1; V[h_idx, col] -= 1
        W[c01_idx, col] += 1; W[c11_idx, col] += 1
        col += 1
        
        # M4 = d(g-e) -> contributes to c00 and c10
        U[d_idx, col] += 1
        V[g_idx, col] += 1; V[e_idx, col] -= 1
        W[c00_idx, col] += 1; W[c10_idx, col] += 1
        col += 1
        
        # M5 = (a+b)h -> contributes to c00 and c01
        U[a_idx, col] += 1; U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1; W[c01_idx, col] += 1
        col += 1
        
        # M6 = (c-a)(e+f) -> contributes to c11
        U[c_idx, col] += 1; U[a_idx, col] -= 1
        V[e_idx, col] += 1; V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1
        
        # M7 = (b-d)(g+h) -> contributes to c00
        U[b_idx, col] += 1; U[d_idx, col] -= 1
        V[g_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1
    
    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        if time.time() > deadline:
            return
        
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)
        
        # P1 = a*e -> c00
        U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00_idx, col] += 1
        col += 1
        # P2 = b*g -> c00
        U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00_idx, col] += 1
        col += 1
        # P3 = c*e -> c10
        U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10_idx, col] += 1
        col += 1
        # P4 = d*g -> c10
        U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10_idx, col] += 1
        col += 1
    
    # j pairs: (0,1), (2,3)
    # k pairs: (0,1), (2,3), (4,5), and singleton (6)
    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
        add_schoolbook_221_block(0, j_start, 6)
    
    if col != rank:
        return _schoolbook(n, m, p)
    
    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify correctness
    if _verify(U, V, W, n, m, p):
        return result
    
    # Fallback to schoolbook if verification fails
    return _schoolbook(n, m, p)
