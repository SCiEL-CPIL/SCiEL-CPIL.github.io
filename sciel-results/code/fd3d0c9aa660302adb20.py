import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def _strassen_2x2():
    """Rank-7 decomposition for 2x2 * 2x2 -> 2x2."""
    triples = []
    def add(u, v, w):
        triples.append((np.array(u, dtype=float),
                        np.array(v, dtype=float),
                        np.array(w, dtype=float)))
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    add([1, 1], [1, 1], [1, 1])
    # m2 = (c+d)e
    add([0, 1], [1, 1], [1, 0])
    # m3 = a(f-h)
    add([1, 0], [1, 0], [0, 1])
    add([1, 0], [0, 1], [0, -1])
    # m4 = d(g-e)
    add([0, 1], [0, 1], [1, 0])
    add([0, 1], [1, 0], [-1, 0])
    # m5 = (a+b)h
    add([1, 0], [1, 1], [0, 1])
    # m6 = (c-a)(e+f)
    add([-1, 1], [1, 0], [1, 1])
    # m7 = (b-d)(g+h)
    add([1, -1], [0, 1], [1, 1])
    
    return triples

def _schoolbook_rect(n, m, p):
    """Schoolbook rank-1 triples for n x m * m x p -> n x p."""
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0.0]*n; u[i] = 1.0
                v = [0.0]*m; v[j] = 1.0
                w = [0.0]*p; w[k] = 1.0
                triples.append((np.array(u), np.array(v), np.array(w)))
    return triples

def _verify_triples(triples, n, m, p, tol=1e-9):
    T = np.zeros((n, m, p))
    for u, v, w in triples:
        T += np.einsum('i,j,k->ijk', u, v, w)
    target = np.ones((n, m, p))
    return np.allclose(T, target, atol=tol)

def _build_global():
    n, m, p = 4, 5, 5
    global_triples = []
    
    # Index maps
    I_TOP = [0, 1]
    I_BOT = [2, 3]
    J_LEFT = [0, 1]
    J_RIGHT = [2, 3, 4]
    K_LEFT = [0, 1]
    K_RIGHT = [2, 3, 4]
    
    def add_block(dec, i_map, j_map, k_map):
        for (u, v, w) in dec:
            u_g = np.zeros(n)
            v_g = np.zeros(m)
            w_g = np.zeros(p)
            for li, gi in enumerate(i_map):
                u_g[gi] = u[li]
            for lj, gj in enumerate(j_map):
                v_g[gj] = v[lj]
            for lk, gk in enumerate(k_map):
                w_g[gk] = w[lk]
            global_triples.append((u_g, v_g, w_g))

    # Get decompositions
    strassen = _strassen_2x2()
    if not _verify_triples(strassen, 2, 2, 2):
        strassen = _schoolbook_rect(2, 2, 2)
        
    # 2x2x2
    dec_222 = strassen
    # 2x2x3
    dec_223 = _schoolbook_rect(2, 2, 3)
    # 2x3x2
    dec_232 = _schoolbook_rect(2, 3, 2)
    
    # 2x3x3 (Optimized)
    dec_233 = []
    # A1 B11: i in 0,1; j in 0,1; k in 0,1
    for u, v, w in dec_222:
        u_g = np.zeros(2); u_g[:] = u
        v_g = np.zeros(3); v_g[0:2] = v
        w_g = np.zeros(3); w_g[0:2] = w
        dec_233.append((u_g, v_g, w_g))
        
    # A1 B12: i in 0,1; j in 0,1; k in 2
    dec_221 = _schoolbook_rect(2, 2, 1)
    for u, v, w in dec_221:
        u_g = np.zeros(2); u_g[:] = u
        v_g = np.zeros(3); v_g[0:2] = v
        w_g = np.zeros(3); w_g[2] = w[0]
        dec_233.append((u_g, v_g, w_g))
        
    # A2 B21: i in 0,1; j in 2; k in 0,1
    dec_212 = _schoolbook_rect(2, 1, 2)
    for u, v, w in dec_212:
        u_g = np.zeros(2); u_g[:] = u
        v_g = np.zeros(3); v_g[2] = v[0]
        w_g = np.zeros(3); w_g[0:2] = w
        dec_233.append((u_g, v_g, w_g))
        
    # A2 B22: i in 0,1; j in 2; k in 2
    dec_211 = _schoolbook_rect(2, 1, 1)
    for u, v, w in dec_211:
        u_g = np.zeros(2); u_g[:] = u
        v_g = np.zeros(3); v_g[2] = v[0]
        w_g = np.zeros(3); w_g[2] = w[0]
        dec_233.append((u_g, v_g, w_g))
        
    if not _verify_triples(dec_233, 2, 3, 3):
        dec_233 = _schoolbook_rect(2, 3, 3)

    # 2x3x2 (Optimized)
    dec_232_opt = []
    # A1 B1: i 0,1; j 0,1; k 0,1
    for u, v, w in dec_222:
        u_g = np.zeros(2); u_g[:] = u
        v_g = np.zeros(3); v_g[0:2] = v
        w_g = np.zeros(2); w_g[:] = w
        dec_232_opt.append((u_g, v_g, w_g))
        
    # A2 B2: i 0,1; j 2; k 0,1
    for u, v, w in dec_212:
        u_g = np.zeros(2); u_g[:] = u
        v_g = np.zeros(3); v_g[2] = v[0]
        w_g = np.zeros(2); w_g[:] = w
        dec_232_opt.append((u_g, v_g, w_g))
        
    if not _verify_triples(dec_232_opt, 2, 3, 2):
        dec_232_opt = _schoolbook_rect(2, 3, 2)

    # Assemble
    # C11 = A11 B11
    add_block(dec_222, I_TOP, J_LEFT, K_LEFT)
    
    # C12 = A11 B12 + A12 B22
    add_block(dec_223, I_TOP, J_LEFT, K_RIGHT)
    add_block(dec_233, I_TOP, J_RIGHT, K_RIGHT)
    
    # C21 = A21 B11 + A22 B21
    add_block(dec_222, I_BOT, J_LEFT, K_LEFT)
    add_block(dec_232_opt, I_BOT, J_RIGHT, K_LEFT)
    
    # C22 = A21 B12 + A22 B22
    add_block(dec_223, I_BOT, J_LEFT, K_RIGHT)
    add_block(dec_233, I_BOT, J_RIGHT, K_RIGHT)
    
    rank = len(global_triples)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for r, (u_g, v_g, w_g) in enumerate(global_triples):
        for i in range(n):
            for j in range(m):
                U[i * m + j, r] = u_g[i] * v_g[j]
        for j in range(m):
            for k in range(p):
                V[j * p + k, r] = v_g[j] * w_g[k]
        for k in range(p):
            for i in range(n):
                W[k * n + i, r] = w_g[k] * u_g[i]
                
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    target = np.ones((n, m, p))
    return np.allclose(T, target, atol=tol)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    U, V, W = _build_global()
    
    if not _verify(U, V, W, n, m, p, tol=1e-8):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U = np.zeros((n * m, rank_sb))
        V = np.zeros((m * p, rank_sb))
        W = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1

    U = np.round(U * 2) / 2.0
    V = np.round(V * 2) / 2.0
    W = np.round(W * 2) / 2.0

    if not _verify(U, V, W, n, m, p, tol=1e-8):
        rank_sb = n * m * p
        U = np.zeros((n * m, rank_sb))
        V = np.zeros((m * p, rank_sb))
        W = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
