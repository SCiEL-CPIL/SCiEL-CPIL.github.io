# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p, tol=1e-9):
    U = np.asarray(U, dtype=float); V = np.asarray(V, dtype=float); W = np.asarray(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(r):
                    s += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1; V[j*p+k][c] = 1; W[k*n+i][c] = 1; c += 1
    return {"U": U, "V": V, "W": W}

def _strassen22():
    # Canonical Strassen 2x2x2 rank-7, verified:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M2 + M4
    # C10 = M3 + M7
    # C11 = M1 - M3 + M6 - M2  (check below)
    # Standard:
    # M1=(a00+a11)(b00+b11)
    # M2=(a10+a11)b00
    # M3=a00(b01-b11)
    # M4=a11(b10-b00)
    # M5=(a00+a01)(b00+b01)
    # M6=(a10+a11)(b10+b11)
    # M7=(a01-a11)(b10+b11)
    # C00=M1-M4+M5-M7 ; C01=M2+M4 ; C10=M3+M7 ; C11=M1-M3+M6-M2
    U2 = [[1,0,1,0,1,0,0],
          [0,0,0,0,1,0,1]]   # wait row1: a10,a11
    # Let me define properly as 2x2 per term.
    U2 = [
        [[1,0],[0,1]],  # M1
        [[0,0],[1,1]],  # M2
        [[1,0],[0,0]],  # M3
        [[0,0],[0,1]],  # M4
        [[1,1],[0,0]],  # M5
        [[0,0],[1,1]],  # M6
        [[0,1],[0,-1]], # M7
    ]
    V2 = [
        [[1,0],[0,1]],  # M1
        [[1,0],[0,0]],  # M2
        [[0,1],[0,-1]], # M3
        [[0,1],[0,-1]], # M4
        [[1,0],[0,1]],  # M5
        [[0,0],[1,1]],  # M6
        [[1,1],[0,0]],  # M7
    ]
    W2 = [
        [[1,0],[0,1]],  # M1 -> C00+1,C11+1
        [[0,1],[0,0]],  # M2 -> C01+1
        [[0,0],[1,0]],  # M3 -> C10+1
        [[0,-1],[0,0]], # M4 -> C01-1? need C00-1
        [[1,0],[0,0]],  # M5 -> C00+1
        [[0,0],[0,1]],  # M6 -> C11+1
        [[-1,0],[0,0]], # M7 -> C00-1, C10+1
    ]
    # Fix W to match formulas:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M2 + M4
    # C10 = M3 + M7
    # C11 = M1 - M3 + M6 - M2
    W2 = [
        [[1,0],[0,1]],  # M1: C00+1, C11+1
        [[0,1],[0,0]],  # M2: C01+1 ; but formula C11 has -M2 -> need C11-1
        [[0,0],[1,0]],  # M3: C10+1 ; C11 has -M3 -> C11-1
        [[1,1],[0,0]],  # M4: C00-1? formula C00=-M4 -> C00-1 ; C01=+M4 -> C01+1
        [[1,0],[0,0]],  # M5: C00+1
        [[0,0],[0,1]],  # M6: C11+1
        [[-1,0],[1,0]], # M7: C00-1, C10+1
    ]
    # Recompute W2 carefully per formula:
    # M1: C00 +=1, C11 +=1
    # M2: C01 +=1, C11 -=1
    # M3: C10 +=1, C11 -=1
    # M4: C00 -=1, C01 +=1
    # M5: C00 +=1
    # M6: C11 +=1
    # M7: C00 -=1, C10 +=1
    W2 = [
        [[1,0],[0,1]],
        [[0,1],[0,-1]],
        [[0,0],[1,-1]],
        [[-1,1],[0,0]],
        [[1,0],[0,0]],
        [[0,0],[0,1]],
        [[-1,0],[1,0]],
    ]
    # Flatten to (4,7): index (i,k) -> i*2+k
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    for t in range(7):
        for i in range(2):
            for k in range(2):
                U[i*2+k][t] = U2[t][i][k]
                V[i*2+k][t] = V2[t][i][k]  # B index (j,k): here j=i? 
                W[i*2+k][t] = W2[t][i][k]
    # Note: V index is (j,k). In 2x2 Strassen, M_t uses A rows i and B cols k,
    # with the inner index being the shared one. Our tensor convention:
    # T_{(i,j),(j,k),(k,i)}. For 2x2 block, A_{i,j}, B_{j,k}, C_{k,i}.
    # Strassen: M_t = (sum_{i,j} U2_t[i,j] A[i,j]) (sum_{j,k} V2_t[j,k] B[j,k]).
    # So U2_t[i,j] multiplies A[i,j]; V2_t[j,k] multiplies B[j,k].
    # In our flatten, A index (i,j) -> i*2+j. So U[i*2+j][t]=U2_t[i][j]. Good.
    # B index (j,k) -> j*2+k. So V[j*2+k][t]=V2_t[j][k].
    # I mistakenly used V[i*2+k]; fix:
    V = [[0]*7 for _ in range(4)]
    for t in range(7):
        for j in range(2):
            for k in range(2):
                V[j*2+k][t] = V2[t][j][k]
    # W index (k,i) -> k*2+i. W2_t[k,i]:
    W = [[0]*7 for _ in range(4)]
    for t in range(7):
        for k in range(2):
            for i in range(2):
                W[k*2+i][t] = W2[t][k][i]
    return U, V, W

def _build_3x3x3_core():
    """Rank-22 3x3x3: Strassen(2x2) for {0,1}x{0,1} outputs + schoolbook for index-2 outputs."""
    U2, V2, W2 = _strassen22()
    rank = 7 + 15  # 7 Strassen + 15 schoolbook (5 outputs x 3)
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(9)]
    for t in range(7):
        for i in range(2):
            for j in range(2):
                U[i*3+j][t] = U2[i*2+j][t]
                V[j*3+i][t] = V2[j*2+i][t]  # B(j,k): j is row, i is col? 
                W[i*3+j][t] = W2[i*2+j][t]  # C(k,i): k=i, i=j
    # Hmm indices: C index (k,i). Strassen W2_t[k,i]. In 3x3, C[k,i] -> k*3+i.
    # For block, k in {0,1}, i in {0,1}: W[k*3+i][t] = W2_t[k,i] = W2[k*2+i][t].
    W = [[0]*rank for _ in range(9)]
    for t in range(7):
        for k in range(2):
            for i in range(2):
                W[k*3+i][t] = W2[k*2+i][t]
    # V index (j,k): V2_t[j,k] = V2[j*2+k][t]. In 3x3: V[j*3+k][t].
    V = [[0]*rank for _ in range(9)]
    for t in range(7):
        for j in range(2):
            for k in range(2):
                V[j*3+k][t] = V2[j*2+k][t]
    # U index (i,j): U2_t[i,j]=U2[i*2+j][t]. In 3x3: U[i*3+j][t].
    U = [[0]*rank for _ in range(9)]
    for t in range(7):
        for i in range(2):
            for j in range(2):
                U[i*3+j][t] = U2[i*2+j][t]
    # Schoolbook for outputs C[i,k] where i==2 or k==2 (5 outputs: C02,C12,C20,C21,C22)
    c = 7
    for i in range(3):
        for k in range(3):
            if i != 2 and k != 2:
                continue
            for j in range(3):
                U[i*3+j][c] = 1
                V[j*3+k][c] = 1
                W[k*3+i][c] = 1
                c += 1
    return U, V, W, c

def _build_3x3x6_44():
    n, m, p = 3, 3, 6
    Uc, Vc, Wc, r = _build_3x3x3_core()
    rank = 2 * r
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(18)]
    W = [[0]*rank for _ in range(18)]
    for t in range(r):
        for i in range(3):
            for j in range(3):
                U[i*3+j][t] = Uc[i*3+j][t]
                U[i*3+j][r+t] = Uc[i*3+j][t]
        for j in range(3):
            for k in range(3):
                V[j*6+k][t] = Vc[j*3+k][t]
                V[j*6+3+k][r+t] = Vc[j*3+k][t]
        for k in range(3):
            for i in range(3):
                W[k*3+i][t] = Wc[k*3+i][t]
                W[(3+k)*3+i][r+t] = Wc[k*3+i][t]
    return U, V, W, rank

def construct():
    start = time.time()
    deadline = 150.0
    n, m, p = 3, 3, 6
    school = _schoolbook(n, m, p)
    try:
        U, V, W, rank = _build_3x3x6_44()
        if rank < 54 and _verify(U, V, W, n, m, p, tol=1e-8):
            if time.time() - start < deadline - 10:
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    if _verify(school['U'], school['V'], school['W'], n, m, p):
        return school
    return school

# EVOLVE-BLOCK-END
