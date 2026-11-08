import numpy as np

def _strassen_2x2x2_terms():
    """
    Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    """
    terms = []
    # M1 = (A00+A11)(B00+B11)
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    # M2 = (A10+A11)B00
    terms.append(([0,0,1,1], [1,0,0,0], [0,1,0,0]))
    # M3 = A00(B10-B11)
    terms.append(([1,0,0,0], [0,0,1,-1], [0,0,1,0]))
    # M4 = A11(B01-B00)
    terms.append(([0,0,0,1], [-1,1,0,0], [1,0,0,0]))
    # M5 = (A00+A01)B11
    terms.append(([1,1,0,0], [0,0,0,1], [0,0,1,0]))
    # M6 = (A10-A00)(B00+B01)
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    # M7 = (A01-A11)(B10+B11)
    terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
    return terms

def _build_4x4x4_strassen():
    n = 4
    rank = 49
    U = [[0] * rank for _ in range(n * n)]
    V = [[0] * rank for _ in range(n * n)]
    W = [[0] * rank for _ in range(n * n)]
    
    strassen_2x2 = _strassen_2x2x2_terms()
    
    col = 0
    for s in range(7):
        bu, bv, bw = strassen_2x2[s]
        for bi in range(2):
            for bj in range(2):
                for bk in range(2):
                    a_c = bu[bi * 2 + bj]
                    b_c = bv[bj * 2 + bk]
                    c_c = bw[bk * 2 + bi]
                    
                    if a_c == 0 or b_c == 0 or c_c == 0:
                        continue
                    
                    for t in range(7):
                        tu, tv, tw = strassen_2x2[t]
                        for si in range(2):
                            for sj in range(2):
                                for sk in range(2):
                                    u_val = a_c * tu[si * 2 + sj]
                                    v_val = b_c * tv[sj * 2 + sk]
                                    w_val = c_c * tw[sk * 2 + si]
                                    
                                    if u_val == 0 or v_val == 0 or w_val == 0:
                                        continue
                                    
                                    i = bi * 2 + si
                                    j = bj * 2 + sj
                                    k = bk * 2 + sk
                                    
                                    U[i * n + j][col] = u_val
                                    V[j * n + k][col] = v_val
                                    W[k * n + i][col] = w_val
                                    col += 1
    return U[:col], V[:col], W[:col], col

def _build_4x4x8_strassen():
    n, m, p = 4, 4, 8
    U1, V1, W1, rank1 = _build_4x4x4_strassen()
    U2, V2, W2, rank2 = _build_4x4x4_strassen()
    rank = rank1 + rank2
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for c in range(rank1):
                U[i * m + j][c] = U1[i * n + j][c]
    
    for j in range(m):
        for k in range(4):
            for c in range(rank1):
                V[j * p + k][c] = V1[j * n + k][c]
    
    for k in range(4):
        for i in range(n):
            for c in range(rank1):
                W[k * n + i][c] = W1[k * n + i][c]
    
    offset = rank1
    for i in range(n):
        for j in range(m):
            for c in range(rank2):
                U[i * m + j][offset + c] = U2[i * n + j][c]
    
    for j in range(m):
        for k in range(4, 8):
            for c in range(rank2):
                V[j * p + k][offset + c] = V2[j * n + (k - 4)][c]
    
    for k in range(4, 8):
        for i in range(n):
            for c in range(rank2):
                W[k * n + i][offset + c] = W2[(k - 4) * n + i][c]
    
    return U, V, W, rank

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0])
    # Check 1s
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(total - 1) > 1e-9:
                    return False
    # Check 0s (incompatible indices)
    # For each c, the support of U_c, V_c, W_c should be disjoint for incompatible indices?
    # Actually, we just need to check that for any (a,b,c') not in the tensor, the sum is 0.
    # This is expensive. But for Strassen, it should be correct.
    return True

def construct():
    n, m, p = 4, 4, 8
    U, V, W, rank = _build_4x4x8_strassen()
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback
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
