import numpy as np
from functools import lru_cache

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    U, V, W = get_decomposition(n, m, p)
    return {"U": U, "V": V, "W": W}

@lru_cache(maxsize=None)
def get_decomposition(n, m, p):
    # Base cases
    if n == 1:
        return get_1xmxp(m, p)
    if m == 1:
        return get_nx1xp(n, p)
    if p == 1:
        return get_nmxp1(n, m)
    if n == 2 and m == 2 and p == 2:
        return get_strassen()
    
    best_rank = float('inf')
    best_decomp = None
    
    # Try splitting n
    for n1 in range(1, n):
        n2 = n - n1
        U1, V1, W1 = get_decomposition(n1, m, p)
        U2, V2, W2 = get_decomposition(n2, m, p)
        rank = U1.shape[1] + U2.shape[1]
        if rank < best_rank:
            best_rank = rank
            # Combine
            U = np.hstack([U1, U2])
            V = np.hstack([V1, V2])
            # W needs careful indexing
            W = np.zeros((p*n, rank))
            for k in range(p):
                for i in range(n):
                    if i < n1:
                        W[k*n + i, :U1.shape[1]] = W1[k*n1 + i, :]
                    else:
                        W[k*n + i, U1.shape[1]:] = W2[k*n2 + (i-n1), :]
            best_decomp = (U, V, W)
            
    # Try splitting m
    for m1 in range(1, m):
        m2 = m - m1
        U1, V1, W1 = get_decomposition(n, m1, p)
        U2, V2, W2 = get_decomposition(n, m2, p)
        rank = U1.shape[1] + U2.shape[1]
        if rank < best_rank:
            best_rank = rank
            U = np.hstack([U1, U2])
            V = np.zeros((m*p, rank))
            for j in range(m):
                for k in range(p):
                    if j < m1:
                        V[j*p + k, :U1.shape[1]] = V1[j*p + k, :]
                    else:
                        V[j*p + k, U1.shape[1]:] = V2[(j-m1)*p + k, :]
            W = np.hstack([W1, W2])
            best_decomp = (U, V, W)
            
    # Try splitting p
    for p1 in range(1, p):
        p2 = p - p1
        U1, V1, W1 = get_decomposition(n, m, p1)
        U2, V2, W2 = get_decomposition(n, m, p2)
        rank = U1.shape[1] + U2.shape[1]
        if rank < best_rank:
            best_rank = rank
            U = np.hstack([U1, U2])
            V = np.zeros((m*p, rank))
            for j in range(m):
                for k in range(p):
                    if k < p1:
                        V[j*p + k, :U1.shape[1]] = V1[j*p1 + k, :]
                    else:
                        V[j*p + k, U1.shape[1]:] = V2[j*p2 + (k-p1), :]
            W = np.zeros((p*n, rank))
            for k in range(p):
                for i in range(n):
                    if k < p1:
                        W[k*n + i, :U1.shape[1]] = W1[k*n + i, :]
                    else:
                        W[k*n + i, U1.shape[1]:] = W2[(k-p1)*n + i, :]
            best_decomp = (U, V, W)
            
    return best_decomp

def get_1xmxp(m, p):
    rank = m * p
    U = np.zeros((1*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*1, rank))
    r = 0
    for j in range(m):
        for k in range(p):
            U[j, r] = 1
            V[j*p + k, r] = 1
            W[k, r] = 1
            r += 1
    return U, V, W

def get_nx1xp(n, p):
    rank = n * p
    U = np.zeros((n*1, rank))
    V = np.zeros((1*p, rank))
    W = np.zeros((p*n, rank))
    r = 0
    for i in range(n):
        for k in range(p):
            U[i, r] = 1
            V[k, r] = 1
            W[k*n + i, r] = 1
            r += 1
    return U, V, W

def get_nmxp1(n, m):
    rank = n * m
    U = np.zeros((n*m, rank))
    V = np.zeros((m*1, rank))
    W = np.zeros((1*n, rank))
    r = 0
    for i in range(n):
        for j in range(m):
            U[i*m + j, r] = 1
            V[j, r] = 1
            W[i, r] = 1
            r += 1
    return U, V, W

def get_strassen():
    # 2x2x2
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1
    
    return U, V, W
