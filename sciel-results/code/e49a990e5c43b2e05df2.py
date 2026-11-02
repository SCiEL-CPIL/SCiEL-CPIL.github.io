import numpy as np
import time

DIMENSIONS = [3, 4, 6]

_start_time = time.time()
_DEADLINE = 145.0

def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = len(U[0]) if U and len(U) > 0 else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def _strassen_2x2x2():
    n = m = p = 2
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = -1; W[3][1] = -1
    
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = -1; W[3][4] = -1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0][5] = 1; W[2][5] = 1
    
    U[3][6] = 1; U[1][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[1][6] = -1; W[3][6] = -1
    
    return U, V, W

def _combine_split_m(U1, V1, W1, U2, V2, W2, n, m1, m2, p):
    rank1 = len(U1[0])
    rank2 = len(U2[0])
    rank_full = rank1 + rank2
    m_full = m1 + m2
    
    U_full = [[0] * rank_full for _ in range(n * m_full)]
    V_full = [[0] * rank_full for _ in range(m_full * p)]
    W_full = [[0] * rank_full for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m1):
            U_full[i * m_full + j] = U1[i * m1 + j] + [0] * rank2
        for j in range(m2):
            U_full[i * m_full + (m1 + j)] = [0] * rank1 + U2[i * m2 + j]
            
    for j in range(m1):
        for k in range(p):
            V_full[j * p + k] = V1[j * p + k] + [0] * rank2
    for j in range(m2):
        for k in range(p):
            V_full[(m1 + j) * p + k] = [0] * rank1 + V2[j * p + k]
            
    for k in range(p):
        for i in range(n):
            idx = k * n + i
            W_full[idx] = [W1[idx][r] for r in range(rank1)] + [W2[idx][r] for r in range(rank2)]
            
    return U_full, V_full, W_full

def _combine_split_p(U1, V1, W1, U2, V2, W2, n, m, p1, p2):
    rank1 = len(U1[0])
    rank2 = len(U2[0])
    rank_full = rank1 + rank2
    p_full = p1 + p2
    
    U_full = [[0] * rank_full for _ in range(n * m)]
    V_full = [[0] * rank_full for _ in range(m * p_full)]
    W_full = [[0] * rank_full for _ in range(p_full * n)]
    
    for i in range(n):
        for j in range(m):
            idx = i * m + j
            U_full[idx] = [U1[idx][r] for r in range(rank1)] + [U2[idx][r] for r in range(rank2)]
            
    for j in range(m):
        for k in range(p1):
            V_full[j * p_full + k] = V1[j * p1 + k] + [0] * rank2
    for j in range(m):
        for k in range(p2):
            V_full[j * p_full + (p1 + k)] = [0] * rank1 + V2[j * p2 + k]
            
    for k in range(p1):
        for i in range(n):
            idx = k * n + i
            W_full[idx] = W1[idx] + [0] * rank2
    for k in range(p2):
        for i in range(n):
            idx = (p1 + k) * n + i
            W_full[idx] = [0] * rank1 + W2[idx]
            
    return U_full, V_full, W_full

def _combine_split_n(U1, V1, W1, U2, V2, W2, n1, n2, m, p):
    rank1 = len(U1[0])
    rank2 = len(U2[0])
    rank_full = rank1 + rank2
    n_full = n1 + n2
    
    U_full = [[0] * rank_full for _ in range(n_full * m)]
    V_full = [[0] * rank_full for _ in range(m * p)]
    W_full = [[0] * rank_full for _ in range(p * n_full)]
    
    for i in range(n1):
        for j in range(m):
            U_full[i * m + j] = U1[i * m + j] + [0] * rank2
    for i in range(n2):
        for j in range(m):
            U_full[(n1 + i) * m + j] = [0] * rank1 + U2[i * m + j]
            
    for j in range(m):
        for k in range(p):
            idx = j * p + k
            V_full[idx] = [V1[idx][r] for r in range(rank1)] + [V2[idx][r] for r in range(rank2)]
            
    for k in range(p):
        for i in range(n1):
            idx = k * n_full + i
            W_full[idx] = W1[k * n1 + i] + [0] * rank2
    for k in range(p):
        for i in range(n2):
            idx = k * n_full + (n1 + i)
            W_full[idx] = [0] * rank1 + W2[k * n2 + i]
            
    return U_full, V_full, W_full

def _try_strassen_recursive(n, m, p):
    if n == 1 and m == 1 and p == 1:
        return [[1]], [[1]], [[1]]
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2x2()
        
    best = None
    best_rank = n * m * p
    
    if n % 2 == 0:
        n1, n2 = n // 2, n - n // 2
        U1, V1, W1 = _try_strassen_recursive(n1, m, p)
        U2, V2, W2 = _try_strassen_recursive(n2, m, p)
        U, V, W = _combine_split_n(U1, V1, W1, U2, V2, W2, n1, n2, m, p)
        if len(U[0]) < best_rank:
            best_rank = len(U[0])
            best = (U, V, W)

    if m % 2 == 0:
        m1, m2 = m // 2, m - m // 2
        U1, V1, W1 = _try_strassen_recursive(n, m1, p)
        U2, V2, W2 = _try_strassen_recursive(n, m2, p)
        U, V, W = _combine_split_m(U1, V1, W1, U2, V2, W2, n, m1, m2, p)
        if len(U[0]) < best_rank:
            best_rank = len(U[0])
            best = (U, V, W)

    if p % 2 == 0:
        p1, p2 = p // 2, p - p // 2
        U1, V1, W1 = _try_strassen_recursive(n, m, p1)
        U2, V2, W2 = _try_strassen_recursive(n, m, p2)
        U, V, W = _combine_split_p(U1, V1, W1, U2, V2, W2, n, m, p1, p2)
        if len(U[0]) < best_rank:
            best_rank = len(U[0])
            best = (U, V, W)

    if best:
        return best
    return _schoolbook(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    if time.time() > _DEADLINE - 10:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    U, V, W = _try_strassen_recursive(n, m, p)
    return {"U": U, "V": V, "W": W}
