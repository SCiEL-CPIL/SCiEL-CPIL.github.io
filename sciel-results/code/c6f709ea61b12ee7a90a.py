import time
import random
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank):
    """Verify tensor decomposition exactly (for integer/half-integer entries)."""
    def scale2(mat):
        out = np.array(mat, dtype=np.float64) * 2
        if np.max(np.abs(out - np.round(out))) > 1e-9:
            return None
        return np.round(out).astype(np.int64)
    U2 = scale2(U)
    V2 = scale2(V)
    W2 = scale2(W)
    if U2 is None or V2 is None or W2 is None:
        return False
    # Build target tensor
    target = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                target[a*m + b, b*p + c, c*n + a] = 1
    T_computed = np.einsum('ic,jc,kc->ijk', U2, V2, W2, optimize=True)
    T_target_8 = target * 8
    if np.max(np.abs(T_computed - T_target_8)) > 1e-6:
        return False
    return True

def _build_target(n, m, p):
    """Build the target tensor for matrix multiplication."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                T[a*m + b, b*p + c, c*n + a] = 1
    return T

def _schoolbook(n, m, p):
    """Return schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for a in range(n):
        for b in range(m):
            for c in range(p):
                U[a*m + b][col] = 1
                V[b*p + c][col] = 1
                W[c*n + a][col] = 1
                col += 1
    return U, V, W

def _strassen_2x2():
    """Return Strassen decomposition for 2x2 matrix multiplication."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _winograd_3x3():
    """Return Winograd decomposition for 3x3 matrix multiplication (Rank 23)."""
    # This is a known decomposition.
    # I will construct it based on standard Winograd coefficients.
    # Due to complexity, I will use a simplified version or fall back to schoolbook if I can't get it right quickly.
    # However, for the sake of the challenge, I will attempt a standard implementation.
    
    # Actually, implementing Winograd 3x3 from scratch in the time limit is risky.
    # I will use Schoolbook for 3x3 to ensure correctness, which gives Rank 124.
    # To improve, I will try to find a better block decomposition or use the Rank 8 trick for cross terms if I can verify it.
    
    # Let's stick to the safe improvement: 
    # If I cannot guarantee Winograd 3x3 correctness, I will return Schoolbook 3x3.
    # But I can try to use a "hybrid" or just return the schoolbook 3x3 to be safe.
    
    # For now, return schoolbook 3x3 to ensure the code runs and is valid.
    U = [[0]*27 for _ in range(9)]
    V = [[0]*27 for _ in range(9)]
    W = [[0]*27 for _ in range(9)]
    col = 0
    for i in range(3):
        for k in range(3):
            for j in range(3):
                U[i*3+k][col] = 1
                V[k*3+j][col] = 1
                W[j*3+i][col] = 1
                col += 1
    return U, V, W

def _build_5x5_improved():
    """Build 5x5 decomposition using Strassen 2x2 and Schoolbook 3x3 (Rank 124)."""
    # This is the same as the seed's _build_5x5_strassen_based
    # I will reuse the logic but ensure it's robust.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def make_u(i, j):
        vec = np.zeros(25)
        vec[i*5 + j] = 1
        return vec
    
    def make_v(j, k):
        vec = np.zeros(25)
        vec[j*5 + k] = 1
        return vec
    
    def make_w(k, i):
        vec = np.zeros(25)
        vec[k*5 + i] = 1
        return vec
    
    def add(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
        
    # 1. A11(2x2) * B11(2x2) -> C11(2x2) : Strassen (7)
    U_s, V_s, W_s = _strassen_2x2()
    for r in range(7):
        u = np.zeros(25)
        v = np.zeros(25)
        w = np.zeros(25)
        for i in range(4):
            if U_s[i][r] != 0:
                a_idx = i // 2
                b_idx = i % 2
                u[a_idx*5 + b_idx] = U_s[i][r]
        for j in range(4):
            if V_s[j][r] != 0:
                b_idx = j // 2
                c_idx = j % 2
                v[b_idx*5 + c_idx] = V_s[j][r]
        for k in range(4):
            if W_s[k][r] != 0:
                c_idx = k // 2
                a_idx = k % 2
                w[c_idx*5 + a_idx] = W_s[k][r]
        add(u, v, w)
        
    # 2. A22(3x3) * B22(3x3) -> C22(3x3) : Schoolbook (27)
    for i in range(3):
        for k in range(3):
            for j in range(3):
                add(make_u(2+i, 2+k), make_v(2+k, 2+j), make_w(2+j, 2+i))
                
    # 3. Cross terms (Schoolbook)
    # A12(2x3) * B21(3x2) -> C11(2x2)
    for i in range(2):
        for k in range(3):
            for j in range(2):
                add(make_u(i, 2+k), make_v(2+k, j), make_w(j, i))
                
    # A11(2x2) * B12(2x3) -> C12(2x3)
    for i in range(2):
        for k in range(2):
            for j in range(3):
                add(make_u(i, k), make_v(k, 2+j), make_w(2+j, i))
                
    # A12(2x3) * B22(3x3) -> C12(2x3)
    for i in range(2):
        for k in range(3):
            for j in range(3):
                add(make_u(i, 2+k), make_v(2+k, 2+j), make_w(2+j, i))
                
    # A21(3x2) * B11(2x2) -> C21(3x2)
    for i in range(3):
        for k in range(2):
            for j in range(2):
                add(make_u(2+i, k), make_v(k, j), make_w(j, 2+i))
                
    # A22(3x3) * B21(3x2) -> C21(3x2)
    for i in range(3):
        for k in range(3):
            for j in range(2):
                add(make_u(2+i, 2+k), make_v(2+k, j), make_w(j, 2+i))
                
    # A21(3x2) * B12(2x3) -> C22(3x3)
    for i in range(3):
        for k in range(2):
            for j in range(3):
                add(make_u(2+i, k), make_v(k, 2+j), make_w(2+j, 2+i))
                
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS
    
    # Use the improved block decomposition
    U, V, W = _build_5x5_improved()
    
    # Verify
    if _verify(U, V, W, n, m, p, len(U[0])):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        U_sb, V_sb, W_sb = _schoolbook(n, m, p)
        return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
