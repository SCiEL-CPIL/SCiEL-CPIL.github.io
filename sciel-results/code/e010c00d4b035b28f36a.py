"""Block-structured low-rank decomposition for 4x5x6 matrix multiplication."""
import time
import random

DIMENSIONS = [4, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 145.0

def remaining():
    return TIME_LIMIT - (time.time() - START_TIME)

def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W, "rank": rank}

def _verify(U, V, W, rank):
    """Verify exact decomposition using random integer matrices."""
    n, m, p = DIMENSIONS
    for _ in range(5):
        if remaining() <= 0:
            return False
        A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
        C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        for i in range(n):
            for k in range(p):
                acc = 0
                for j in range(m):
                    for c in range(rank):
                        u_val = U[i * m + j][c]
                        if u_val == 0:
                            continue
                        acc += u_val * W[k * n + i][c] * V[j * p + k][c]
                if acc != C[i][k]:
                    return False
    return True

def _winograd_2x2x2():
    """Returns 7 rank-1 terms for 2x2x2 matmul."""
    terms = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
        ([0,0,0,1], [-1,0,1,0], [-1,1,1,0]),
        ([1,1,0,0], [0,0,1,0], [1,1,0,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,-1,1]),
        ([0,1,0,-1], [0,1,0,1], [-1,1,1,0]),
    ]
    return terms

def _embed_block(terms, i_base, j_base, k_base, i_size, j_size, k_size, n, m, p):
    """Embed a block's terms into full-sized vectors."""
    full_terms = []
    for (u, v, w) in terms:
        ufull = [0] * (n * m)
        vfull = [0] * (m * p)
        wfull = [0] * (p * n)
        
        for a in range(i_size):
            for b in range(j_size):
                idx = (i_base + a) * m + (j_base + b)
                ufull[idx] = u[a * j_size + b]
        
        for b in range(j_size):
            for c in range(k_size):
                idx = (j_base + b) * p + (k_base + c)
                vfull[idx] = v[b * k_size + c]
        
        for c in range(k_size):
            for a in range(i_size):
                idx = (k_base + c) * n + (i_base + a)
                wfull[idx] = w[c * i_size + a]
        
        full_terms.append((ufull, vfull, wfull))
    return full_terms

def _build_from_terms(terms_U, terms_V, terms_W, n, m, p):
    """Build U, V, W matrices from lists of term vectors."""
    rank = len(terms_U)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(zip(terms_U, terms_V, terms_W)):
        for i in range(n * m):
            U[i][c] = u[i]
        for j in range(m * p):
            V[j][c] = v[j]
        for k in range(p * n):
            W[k][c] = w[k]
    
    return {"U": U, "V": V, "W": W, "rank": rank}

def _strategy_hybrid():
    """
    Use 2x2x2 Winograd blocks for i:0-3, j:0-3, k:0-5,
    then schoolbook for j=4 (all i,k).
    Total: 12 blocks * 7 + 24 = 84 + 24 = 108
    """
    n, m, p = DIMENSIONS
    winograd_222 = _winograd_2x2x2()
    
    terms_U = []
    terms_V = []
    terms_W = []
    
    for ib in range(2):
        for jb in range(2):
            for kb in range(3):
                i_base = 2 * ib
                j_base = 2 * jb
                k_base = 2 * kb
                
                embedded = _embed_block(winograd_222, i_base, j_base, k_base, 2, 2, 2, n, m, p)
                terms_U.extend([t[0] for t in embedded])
                terms_V.extend([t[1] for t in embedded])
                terms_W.extend([t[2] for t in embedded])
    
    for i in range(n):
        for k in range(p):
            ufull = [0] * (n * m)
            vfull = [0] * (m * p)
            wfull = [0] * (p * n)
            ufull[i * m + 4] = 1
            vfull[4 * p + k] = 1
            wfull[k * n + i] = 1
            terms_U.append(ufull)
            terms_V.append(vfull)
            terms_W.append(wfull)
    
    return _build_from_terms(terms_U, terms_V, terms_W, n, m, p)

def construct():
    n, m, p = DIMENSIONS
    
    candidates = []
    
    if remaining() > 10:
        try:
            candidates.append(_strategy_hybrid())
        except Exception as e:
            pass
    
    candidates.append(_schoolbook())
    
    best = None
    for cand in candidates:
        if remaining() <= 0:
            break
        rank = cand["rank"]
        U, V, W = cand["U"], cand["V"], cand["W"]
        if _verify(U, V, W, rank):
            if best is None or rank < best["rank"]:
                best = cand
    
    if best is None:
        best = candidates[-1]
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
