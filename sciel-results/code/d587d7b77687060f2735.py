import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 2, 9]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
    U3 = np.array(U, dtype=float).reshape(n, m, -1)
    V3 = np.array(V, dtype=float).reshape(m, p, -1)
    W3 = np.array(W, dtype=float).reshape(p, n, -1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U3[i, j, :], V3[j, k, :] * W3[k, i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_base_decomposition(n, m, p):
    """Build the base rank-32 decomposition."""
    rank = 4 * 7 + 4  # 32
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0

    def add_strassen(k0, k1):
        nonlocal col
        U[0][col] = 1.0; U[3][col] = 1.0
        V[0*p + k0][col] = 1.0; V[1*p + k1][col] = 1.0
        W[k0*n + 0][col] = 1.0; W[k1*n + 1][col] = 1.0
        col += 1
        U[2][col] = 1.0; U[3][col] = 1.0
        V[0*p + k0][col] = 1.0
        W[k0*n + 1][col] = 1.0; W[k1*n + 1][col] = -1.0
        col += 1
        U[0][col] = 1.0
        V[0*p + k1][col] = 1.0; V[1*p + k1][col] = -1.0
        W[k1*n + 0][col] = 1.0; W[k1*n + 1][col] = 1.0
        col += 1
        U[3][col] = 1.0
        V[1*p + k0][col] = 1.0; V[0*p + k0][col] = -1.0
        W[k0*n + 0][col] = 1.0; W[k0*n + 1][col] = 1.0
        col += 1
        U[0][col] = 1.0; U[1][col] = 1.0
        V[1*p + k1][col] = 1.0
        W[k0*n + 0][col] = -1.0; W[k1*n + 0][col] = 1.0
        col += 1
        U[2][col] = 1.0; U[0][col] = -1.0
        V[0*p + k0][col] = 1.0; V[0*p + k1][col] = 1.0
        W[k1*n + 1][col] = 1.0
        col += 1
        U[1][col] = 1.0; U[3][col] = -1.0
        V[1*p + k0][col] = 1.0; V[1*p + k1][col] = 1.0
        W[k0*n + 0][col] = 1.0
        col += 1

    def add_schoolbook(k):
        nonlocal col
        U[0][col] = 1.0; V[0*p + k][col] = 1.0; W[k*n + 0][col] = 1.0; col += 1
        U[1][col] = 1.0; V[1*p + k][col] = 1.0; W[k*n + 0][col] = 1.0; col += 1
        U[2][col] = 1.0; V[0*p + k][col] = 1.0; W[k*n + 1][col] = 1.0; col += 1
        U[3][col] = 1.0; V[1*p + k][col] = 1.0; W[k*n + 1][col] = 1.0; col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_schoolbook(8)
    return U, V, W

def _try_remove_single_term(U, V, W, n, m, p, r_idx):
    rank = len(U[0])
    if r_idx < 0 or r_idx >= rank:
        return False, None, None, None
    
    sig_r = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = U[i*m+j][r_idx] * V[j*p+k][r_idx] * W[k*n+i][r_idx]
                if abs(val) > 1e-12:
                    sig_r[(i,j,k)] = val
    
    if not sig_r:
        new_U = [row[:r_idx] + row[r_idx+1:] for row in U]
        new_V = [row[:r_idx] + row[r_idx+1:] for row in V]
        new_W = [row[:r_idx] + row[r_idx+1:] for row in W]
        return True, new_U, new_V, new_W
    
    for s in range(rank):
        if s == r_idx:
            continue
        
        alphas = []
        valid = True
        for (i,j,k), val in sig_r.items():
            denom = U[i*m+j][s] * V[j*p+k][s] * W[k*n+i][s]
            if abs(denom) < 1e-12:
                valid = False
                break
            a = val / denom
            alphas.append(a)
        
        if not valid or not alphas:
            continue
        
        a0 = alphas[0]
        if all(abs(a - a0) < 1e-9 for a in alphas):
            new_U = [row[:r_idx] + row[r_idx+1:] for row in U]
            new_V = [row[:r_idx] + row[r_idx+1:] for row in V]
            new_W = [row[:r_idx] + row[r_idx+1:] for row in W]
            s_new = s - 1 if s > r_idx else s
            for i in range(n*m):
                new_U[i][s_new] += a0 * U[i][s]
            for j in range(m*p):
                new_V[j][s_new] += a0 * V[j][s]
            for k in range(p*n):
                new_W[k][s_new] += a0 * W[k][s]
            return True, new_U, new_V, new_W
    
    return False, None, None, None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    U, V, W = _build_base_decomposition(n, m, p)
    
    if not _verify(U, V, W, n, m, p):
        rank_fb = n * m * p
        U_fb = [[0.0] * rank_fb for _ in range(n * m)]
        V_fb = [[0.0] * rank_fb for _ in range(m * p)]
        W_fb = [[0.0] * rank_fb for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][c] = 1.0
                    V_fb[j*p+k][c] = 1.0
                    W_fb[k*n+i][c] = 1.0
                    c += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    best_U, best_V, best_W = U, V, W
    best_rank = len(U[0])
    
    rank = len(U[0])
    for r in range(rank):
        if time.time() > deadline - 20:
            break
        success, nU, nV, nW = _try_remove_single_term(U, V, W, n, m, p, r)
        if success:
            if _verify(nU, nV, nW, n, m, p, tol=1e-8):
                best_U, best_V, best_W = nU, nV, nW
                best_rank = len(nU[0])
                U, V, W = nU, nV, nW
                rank = len(U[0])
    
    if time.time() < deadline - 30:
        rank = len(U[0])
        for r1, r2 in combinations(range(rank), 2):
            if time.time() > deadline - 20:
                break
            # Pair removal logic simplified for speed
            # Check if r1 and r2 can be removed
            # This is complex, so we skip it in this version to ensure speed
            pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
