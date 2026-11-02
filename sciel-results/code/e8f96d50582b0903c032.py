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

def _winograd_2x2x3():
    """Returns rank-1 terms for 2x2x3 matmul using Winograd-like approach."""
    # For 2x2x3: A is 2x2, B is 2x3, C is 2x3
    # Schoolbook rank = 2*2*3 = 12
    # We can use a combination: treat as two 2x2x2 blocks sharing some structure
    # Or use a direct Winograd-like construction
    # 
    # Let A = [[a,b],[c,d]], B = [[e,f,g],[h,i,j]]
    # C = [[ae+bh, af+bi, ag+bj], [ce+dh, cf+di, cg+dj]]
    #
    # Using 2x2x2 Winograd on columns (e,g) and (h,j) separately doesn't help directly.
    # Instead, let's use a known low-rank construction.
    # 
    # Actually, let's use a simpler approach: partition the 2x2x3 into two 1x2x3 blocks
    # or use the fact that we can share computations.
    #
    # For now, let's use a construction that gives rank 11 (saving 1 over schoolbook):
    # We can write:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e  
    # p3 = a(f-h)
    # p4 = d(g-e)  -- wait, g is in position (0,2), h is (1,0)
    # 
    # Let me be more careful. B = [[e,f,g],[h,i,j]]
    # C[0,0] = ae+bh, C[0,1] = af+bi, C[0,2] = ag+bj
    # C[1,0] = ce+dh, C[1,1] = cf+di, C[1,2] = cg+dj
    #
    # A simple rank-11 construction:
    # Use 7 terms for the "2x2x2" part (columns 0,1 of B) and 4 terms for column 2,
    # but share some. Actually, let's just use schoolbook for now and try 2x2x2 blocks.
    
    # For simplicity and correctness, let's return schoolbook terms for 2x2x3
    # This gives rank 12, same as schoolbook. Not ideal but correct.
    # TODO: Find a better construction
    
    terms = []
    # Schoolbook: 12 terms
    for i in range(2):
        for j in range(2):
            for k in range(3):
                u = [0]*4; v = [0]*6; w = [0]*6
                u[i*2+j] = 1
                v[j*3+k] = 1
                w[k*2+i] = 1
                terms.append((u, v, w))
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

def _block_decomposition():
    """Build 4x5x6 as sum of Winograd blocks plus schoolbook remainders."""
    n, m, p = DIMENSIONS
    
    terms_U = []
    terms_V = []
    terms_W = []
    
    # 2x2x2 Winograd blocks
    winograd_222 = _winograd_2x2x2()
    
    covered = set()
    
    # Try to cover as much as possible with 2x2x2 blocks
    # i: 0-3 (4 rows), j: 0-4 (5 cols), k: 0-5 (6 cols)
    # 2x2x2 blocks: i in {0,1},{2,3}; j in {0,1},{2,3}; k in {0,1},{2,3}
    # That covers i:0-3, j:0-3, k:0-3 with 2*2*2 = 8 blocks of rank 7 = 56 terms
    # Remaining: j=4 (all i,k), k=4,5 (all i,j)
    
    for ib in range(2):
        for kb in range(2):
            for jb in range(2):
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
    
    # Add schoolbook terms for uncovered (i,j,k)
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

def _try_2x2x3_blocks():
    """Try using 2x2x3 blocks for better coverage of k dimension."""
    n, m, p = DIMENSIONS
    
    terms_U = []
    terms_V = []
    terms_W = []
    
    covered = set()
    
    # 2x2x2 Winograd blocks for k in {0,1},{2,3}
    winograd_222 = _winograd_2x2x2()
    
    for ib in range(2):
        for kb in range(2):
            for jb in range(2):
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
    
    # For k in {4,5}: use 2x2x2 blocks with j in {0,1},{2,3}
    for ib in range(2):
        for jb in range(2):
            i_base = 2 * ib
            j_base = 2 * jb
            k_base = 4
            
            embedded = _embed_block(winograd_222, i_base, j_base, k_base, 2, 2, 2, n, m, p)
            terms_U.extend([t[0] for t in embedded])
            terms_V.extend([t[1] for t in embedded])
            terms_W.extend([t[2] for t in embedded])
            
            for a in range(2):
                for b in range(2):
                    for c in range(2):
                        covered.add((i_base + a, j_base + b, k_base + c))
    
    # Add schoolbook for j=4 (all i,k)
    for i in range(n):
        for k in range(p):
            if (i, 4, k) in covered:
                continue
            ufull = [0] * (n * m)
            vfull = [0] * (m * p)
            wfull = [0] * (p * n)
            ufull[i * m + 4] = 1
            vfull[4 * p + k] = 1
            wfull[k * n + i] = 1
            terms_U.append(ufull)
            terms_V.append(vfull)
            terms_W.append(wfull)
    
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

def construct():
    n, m, p = DIMENSIONS
    
    # Try multiple block strategies and pick the best verified one
    candidates = []
    
    # Strategy 1: Basic 2x2x2 blocks + schoolbook remainder
    if remaining() > 10:
        try:
            result1 = _block_decomposition()
            candidates.append(result1)
        except:
            pass
    
    # Strategy 2: 2x2x2 blocks covering k=0..5 + schoolbook for j=4
    if remaining() > 5:
        try:
            result2 = _try_2x2x3_blocks()
            candidates.append(result2)
        except:
            pass
    
    # Strategy 3: Schoolbook as fallback
    result3 = _schoolbook()
    candidates.append(result3)
    
    # Verify each candidate and pick the best valid one
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
        best = result3
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
