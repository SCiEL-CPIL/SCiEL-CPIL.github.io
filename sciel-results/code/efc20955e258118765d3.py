"""Block-structured low-rank decomposition for 4x5x6 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 145.0  # 5s safety margin before 150s kill

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
    import random
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
    # Standard Winograd F^T(2,2,2) with 7 multiplications
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C00 = p1 - p4 + p5 - p7
    # C01 = p3 + p4 + p5 + p7
    # C10 = p2 + p4 - p6 + p7
    # C11 = p1 - p2 + p6 - p7
    terms = [
        # (U_vec, V_vec, W_vec) where U_vec is flat [a,b,c,d], V_vec is [e,f,g,h], W_vec is [C00,C01,C10,C11]
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),    # p1 = (a+d)(e+h)
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),   # p2 = (c+d)e
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),   # p3 = a(f-h)
        ([0,0,0,1], [-1,0,1,0], [-1,1,1,0]),  # p4 = d(g-e)
        ([1,1,0,0], [0,0,1,0], [1,1,0,0]),    # p5 = (a+b)g
        ([-1,0,1,0], [1,1,0,0], [0,0,-1,1]),  # p6 = (c-a)(e+f)
        ([0,1,0,-1], [0,1,0,1], [-1,1,1,0]),  # p7 = (b-d)(h+f)
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

def _strategy_optimal_222():
    """
    Use 2x2x2 Winograd blocks to cover i:0-3, j:0-3, k:0-5,
    then schoolbook for j=4 (all i,k).
    
    Blocks: 2 (i) x 2 (j) x 3 (k) = 12 blocks, each rank 7 = 84 terms
    Schoolbook for j=4: 4 * 6 = 24 terms
    Total: 84 + 24 = 108
    """
    n, m, p = DIMENSIONS
    winograd_222 = _winograd_2x2x2()
    
    terms_U = []
    terms_V = []
    terms_W = []
    
    # 2x2x2 Winograd blocks for i:0-3, j:0-3, k:0-5
    for ib in range(2):  # i blocks: 0-1, 2-3
        for jb in range(2):  # j blocks: 0-1, 2-3
            for kb in range(3):  # k blocks: 0-1, 2-3, 4-5
                i_base = 2 * ib
                j_base = 2 * jb
                k_base = 2 * kb
                
                embedded = _embed_block(winograd_222, i_base, j_base, k_base, 2, 2, 2, n, m, p)
                terms_U.extend([t[0] for t in embedded])
                terms_V.extend([t[1] for t in embedded])
                terms_W.extend([t[2] for t in embedded])
    
    # Schoolbook for j=4 (all i, k)
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

def _strategy_strassen_4x4x4():
    """
    Use 2x2x2 Winograd blocks for i:0-3, j:0-3, k:0-3,
    then schoolbook for remainder (j=4 all i,k and k=4,5 all i,j).
    
    Blocks: 2 (i) x 2 (j) x 2 (k) = 8 blocks, each rank 7 = 56 terms
    Schoolbook remainder: 
      j=4: 4*6 = 24
      k=4,5 with j=0-3: 4*4*2 = 32
      Total remainder: 56
    Total: 56 + 56 = 112
    """
    n, m, p = DIMENSIONS
    winograd_222 = _winograd_2x2x2()
    
    terms_U = []
    terms_V = []
    terms_W = []
    covered = set()
    
    # 2x2x2 Winograd blocks for i:0-3, j:0-3, k:0-3
    for ib in range(2):
        for jb in range(2):
            for kb in range(2):
                i_base = 2 * ib
                j_base = 2 * jb
                k_base = 2 * kb
                
                embedded = _embed_block(winograd_222, i_base, j_base, k_base, 2, 2, 2, n, m, p)
                terms_U.extend([t[0] for t in embedded])
                terms_V.extend([t[1] for t in embedded])
                terms_W.extend([t[2] for t in embedded])
                
                for a in range(2):
                    for b in range(2):
                        for c in range(2):
                            covered.add((i_base + a, j_base + b, k_base + c))
    
    # Schoolbook for uncovered (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                ufull = [0] * (n * m)
                vfull = [0] * (m * p)
                wfull = [0] * (p * n)
                ufull[i * m + j] = 1
                vfull[j * p + k] = 1
                wfull[k * n + i] = 1
                terms_U.append(ufull)
                terms_V.append(vfull)
                terms_W.append(wfull)
    
    return _build_from_terms(terms_U, terms_V, terms_W, n, m, p)

def construct():
    n, m, p = DIMENSIONS
    
    candidates = []
    
    # Strategy 1: 2x2x2 Winograd for i:0-3, j:0-3, k:0-5 + schoolbook j=4 (rank 108)
    if remaining() > 10:
        try:
            candidates.append(_strategy_optimal_222())
        except:
            pass
    
    # Strategy 2: 2x2x2 Winograd for i:0-3, j:0-3, k:0-3 + schoolbook remainder (rank 112)
    if remaining() > 5:
        try:
            candidates.append(_strategy_strassen_4x4x4())
        except:
            pass
    
    # Strategy 3: Schoolbook (rank 120)
    candidates.append(_schoolbook())
    
    # Verify and pick best
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
        best = candidates[-1]  # schoolbook
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
