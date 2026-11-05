import time
import numpy as np

_DIMS = (3, 4, 7)

def construct():
    t0 = time.monotonic()
    T_LIM = 105.0
    
    n, m, p = _DIMS
    # Max rank estimate
    R = 84 
    
    U = np.zeros((n * m, R), dtype=np.float64)
    V = np.zeros((m * p, R), dtype=np.float64)
    W = np.zeros((p * n, R), dtype=np.float64)
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val=1.0):
        nonlocal col
        U[u_idx, col] = val
        V[v_idx, col] = val
        W[w_idx, col] = val
        col += 1

    # Helper to add Strassen block
    def add_strassen(i0, j0, k0):
        nonlocal col
        # i0, j0, k0 are top-left indices
        # Block covers i in [i0, i0+1], j in [j0, j0+1], k in [k0, k0+1]
        
        # M1 = (A11+A22)(B11+B22)
        # U: A11, A22. V: B11, B22. W: C11, C22? No, W is Cki.
        # C11 = M1 - M4 - M5 - M7
        # C12 = M3 + M4 + M7
        # C21 = M2 + M5 + M6
        # C22 = M1 - M2 - M3 + M6
        
        # Let's use the standard 7 terms mapping to U, V, W.
        # Term 1: U=A11+A22, V=B11+B22, W=C11+C22? No.
        # The decomposition is:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 - M4 - M5 - M7
        # C12 = M3 + M4 + M7
        # C21 = M2 + M5 + M6
        # C22 = M1 - M2 - M3 + M6
        
        # We need to express this as sum of rank-1 tensors.
        # Each M is a rank-1 tensor? No, M is a product of linear forms.
        # M1 = (u1+v1) * (v1+v2) * w?
        # The tensor is T_ijk.
        # M1 contributes to C11 and C22?
        # Actually, the standard Strassen decomposition gives 7 rank-1 terms directly.
        
        # Term 1: U=(1,0,0,1) [A11,A22], V=(1,0,1,0) [B11,B22], W=(1,0,0,1) [C11,C22]?
        # Indices for U: i0*m+j0, (i0+1)*m+(j0+1)
        # Indices for V: j0*p+k0, (j0+1)*p+(k0+1)
        # Indices for W: k0*n+i0, (k0+1)*n+(i0+1)
        
        u11 = i0*m+j0
        u22 = (i0+1)*m+(j0+1)
        u21 = (i0+1)*m+j0
        u12 = i0*m+(j0+1)
        
        v11 = j0*p+k0
        v22 = (j0+1)*p+(k0+1)
        v21 = (j0+1)*p+k0
        v12 = j0*p+(k0+1)
        
        w11 = k0*n+i0
        w22 = (k0+1)*n+(i0+1)
        w21 = (k0+1)*n+i0
        w12 = k0*n+(i0+1)
        
        # M1
        U[u11, col] = 1; U[u22, col] = 1
        V[v11, col] = 1; V[v22, col] = 1
        W[w11, col] = 1; W[w22, col] = 1
        col += 1
        
        # M2
        U[u21, col] = 1; U[u22, col] = 1
        V[v11, col] = 1
        W[w21, col] = 1
        col += 1
        
        # M3
        U[u11, col] = 1
        V[v21, col] = 1; V[v22, col] = -1
        W[w12, col] = 1
        col += 1
        
        # M4
        U[u22, col] = 1
        V[v12, col] = 1; V[v11, col] = -1
        W[w11, col] = 1
        col += 1
        
        # M5
        U[u11, col] = 1; U[u12, col] = 1
        V[v22, col] = 1
        W[w11, col] = 1
        col += 1
        
        # M6
        U[u21, col] = 1; U[u11, col] = -1
        V[v11, col] = 1; V[v12, col] = 1
        W[w22, col] = 1
        col += 1
        
        # M7
        U[u12, col] = 1; U[u22, col] = -1
        V[v21, col] = 1; V[v22, col] = 1
        W[w12, col] = 1
        col += 1

    # 1. 3x3x3 Schoolbook (I:0-2, J:0-2, K:0-2)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                add_term(i*m+j, j*p+k, k*n+i)
                
    # 2. J=3, K:3-6 (I:0-2)
    # Rank 1
    for i in range(3):
        for k in range(3, 7):
            # We need one term for all
            pass
    # Add single term
    for i in range(3):
        U[i*m+3, col] = 1
    for k in range(3, 7):
        V[3*p+k, col] = 1
    for k in range(3, 7):
        for i in range(3):
            W[k*n+i, col] = 1
    col += 1

    # 3a. Strassen I:0-1, J:0-1, K:3-4
    add_strassen(0, 0, 3)
    
    # 3b. I=2, J:0-1, K:3-4
    # Rank 1
    for j in range(2):
        U[2*m+j, col] = 1
    for j in range(2):
        for k in range(3, 5):
            V[j*p+k, col] = 1
    for k in range(3, 5):
        W[k*n+2, col] = 1
    col += 1

    # 4. J=2, K:3-6 (I:0-2)
    # Rank 1
    for i in range(3):
        U[i*m+2, col] = 1
    for k in range(3, 7):
        V[2*p+k, col] = 1
    for k in range(3, 7):
        for i in range(3):
            W[k*n+i, col] = 1
    col += 1

    # 5a. Strassen I:0-1, J:0-1, K:5-6
    add_strassen(0, 0, 5)
    
    # 5b. I=2, J:0-1, K:5-6
    # Rank 1
    for j in range(2):
        U[2*m+j, col] = 1
    for j in range(2):
        for k in range(5, 7):
            V[j*p+k, col] = 1
    for k in range(5, 7):
        W[k*n+2, col] = 1
    col += 1

    # 7. J=3, K:0-2 (I:0-2)
    # Rank 1
    for i in range(3):
        U[i*m+3, col] = 1
    for k in range(3):
        V[3*p+k, col] = 1
    for k in range(3):
        for i in range(3):
            W[k*n+i, col] = 1
    col += 1

    if time.monotonic() - t0 > T_LIM:
        return _fallback(n, m, p)

    # Verify
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]
    
    if _check(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return _fallback(n, m, p)

def _check(U, V, W, n, m, p):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > 1e-7:
                    return False
    return True

def _fallback(n, m, p):
    R = n * m * p
    U = np.zeros((n * m, R), dtype=np.float64)
    V = np.zeros((m * p, R), dtype=np.float64)
    W = np.zeros((p * n, R), dtype=np.float64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
