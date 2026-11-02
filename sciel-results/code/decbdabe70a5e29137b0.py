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

def _strassen_4x4x4_terms():
    """
    Rank-27 decomposition for 4x4x4 matrix multiplication.
    Uses Strassen's method: 4x4 = 2x2 blocks, 7 multiplications of 2x2x2.
    Each 2x2x2 uses Winograd rank 7, total 7*7 = 49? No.
    
    Actually, Strassen for 4x4x4: partition into 2x2 blocks in each dimension.
    M1 = (A11+A22)*(B11+B22)
    M2 = (A21+A22)*B11
    M3 = A11*(B12-B22)
    M4 = A22*(B21-B11)
    M5 = (A11+A12)*B22
    M6 = (A21-A11)*(B11+B12)
    M7 = (A12-A22)*(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Each M is a 2x2x2 product, rank 7. Total rank = 7*7 = 49.
    But we can do better by sharing. Let's just use 49 for now and see.
    """
    # Winograd 2x2x2 terms
    winograd_222 = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
        ([0,0,0,1], [-1,0,1,0], [-1,1,1,0]),
        ([1,1,0,0], [0,0,1,0], [1,1,0,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,-1,1]),
        ([0,1,0,-1], [0,1,0,1], [-1,1,1,0]),
    ]
    
    # Strassen combinations for 4x4x4
    # A is 4x4: A11=A[0:2,0:2], A12=A[0:2,2:4], A21=A[2:4,0:2], A22=A[2:4,2:4]
    # B is 4x4: B11=B[0:2,0:2], B12=B[0:2,2:4], B21=B[2:4,0:2], B22=B[2:4,2:4]
    # C is 4x4: C11=C[0:2,0:2], etc.
    
    # For each Strassen multiplication, we need to:
    # 1. Form the linear combinations of A and B blocks
    # 2. Multiply (2x2x2, rank 7)
    # 3. Combine results into C blocks
    
    # U vector: 16 entries (4x4 A), V vector: 16 entries (4x4 B), W vector: 16 entries (4x4 C)
    # Index: A[i*4+j], B[j*4+k], C[k*4+i]
    
    terms = []
    
    # Strassen M1 = (A11+A22)*(B11+B22)
    # A11+A22: indices (0,0),(0,1),(1,0),(1,1) and (2,2),(2,3),(3,2),(3,3)
    # B11+B22: indices (0,0),(0,1),(1,0),(1,1) and (2,2),(2,3),(3,2),(3,3)
    # Result goes to C11 and C22
    for (u, v, w) in winograd_222:
        ufull = [0]*16
        vfull = [0]*16
        wfull = [0]*16
        # u: A11+A22
        ufull[0*4+0] = u[0]  # A[0,0]
        ufull[0*4+1] = u[1]  # A[0,1]
        ufull[1*4+0] = u[2]  # A[1,0]
        ufull[1*4+1] = u[3]  # A[1,1]
        ufull[2*4+2] = u[0]  # A[2,2]
        ufull[2*4+3] = u[1]  # A[2,3]
        ufull[3*4+2] = u[2]  # A[3,2]
        ufull[3*4+3] = u[3]  # A[3,3]
        # v: B11+B22
        vfull[0*4+0] = v[0]
        vfull[0*4+1] = v[1]
        vfull[1*4+0] = v[2]
        vfull[1*4+1] = v[3]
        vfull[2*4+2] = v[0]
        vfull[2*4+3] = v[1]
        vfull[3*4+2] = v[2]
        vfull[3*4+3] = v[3]
        # w: C11 and C22
        # C11 = w[0:4] mapped to C[0:2,0:2], C22 = w[4:8]... no, w is 4 entries for 2x2
        # w[0]=C11[0,0], w[1]=C11[0,1], w[2]=C11[1,0], w[3]=C11[1,1]
        wfull[0*4+0] = w[0]  # C[0,0]
        wfull[0*4+1] = w[1]  # C[0,1]
        wfull[1*4+0] = w[2]  # C[1,0]
        wfull[1*4+1] = w[3]  # C[1,1]
        wfull[2*4+2] = w[0]  # C[2,2]
        wfull[2*4+3] = w[1]  # C[2,3]
        wfull[3*4+2] = w[2]  # C[3,2]
        wfull[3*4+3] = w[3]  # C[3,3]
        terms.append((ufull, vfull, wfull))
    
    # M2 = (A21+A22)*B11
    # A21+A22: A[2,0],A[2,1],A[3,0],A[3,1] + A[2,2],A[2,3],A[3,2],A[3,3]
    # B11: B[0,0],B[0,1],B[1,0],B[1,1]
    # Result to C21
    for (u, v, w) in winograd_222:
        ufull = [0]*16
        vfull = [0]*16
        wfull = [0]*16
        ufull[2*4+0] = u[0]
        ufull[2*4+1] = u[1]
        ufull[3*4+0] = u[2]
        ufull[3*4+1] = u[3]
        ufull[2*4+2] = u[0]
        ufull[2*4+3] = u[1]
        ufull[3*4+2] = u[2]
        ufull[3*4+3] = u[3]
        vfull[0*4+0] = v[0]
        vfull[0*4+1] = v[1]
        vfull[1*4+0] = v[2]
        vfull[1*4+1] = v[3]
        wfull[2*4+0] = w[0]
        wfull[2*4+1] = w[1]
        wfull[3*4+0] = w[2]
        wfull[3*4+1] = w[3]
        terms.append((ufull, vfull, wfull))
    
    # M3 = A11*(B12-B22)
    for (u, v, w) in winograd_222:
        ufull = [0]*16
        vfull = [0]*16
        wfull = [0]*16
        ufull[0*4+0] = u[0]
        ufull[0*4+1] = u[1]
        ufull[1*4+0] = u[2]
        ufull[1*4+1] = u[3]
        vfull[0*4+2] = v[0]
        vfull[0*4+3] = v[1]
        vfull[1*4+2] = v[2]
        vfull[1*4+3] = v[3]
        vfull[2*4+2] = -v[0]
        vfull[2*4+3] = -v[1]
        vfull[3*4+2] = -v[2]
        vfull[3*4+3] = -v[3]
        wfull[0*4+2] = w[0]
        wfull[0*4+3] = w[1]
        wfull[1*4+2] = w[2]
        wfull[1*4+3] = w[3]
        terms.append((ufull, vfull, wfull))
    
    # M4 = A22*(B21-B11)
    for (u, v, w) in winograd_222:
        ufull = [0]*16
        vfull = [0]*16
        wfull = [0]*16
        ufull[2*4+2] = u[0]
        ufull[2*4+3] = u[1]
        ufull[3*4+2] = u[2]
        ufull[3*4+3] = u[3]
        vfull[2*4+0] = v[0]
        vfull[2*4+1] = v[1]
        vfull[3*4+0] = v[2]
        vfull[3*4+1] = v[3]
        vfull[0*4+0] = -v[0]
        vfull[0*4+1] = -v[1]
        vfull[1*4+0] = -v[2]
        vfull[1*4+1] = -v[3]
        wfull[2*4+0] = w[0]
        wfull[2*4+1] = w[1]
        wfull[3*4+0] = w[2]
        wfull[3*4+1] = w[3]
        terms.append((ufull, vfull, wfull))
    
    # M5 = (A11+A12)*B22
    for (u, v, w) in winograd_222:
        ufull = [0]*16
        vfull = [0]*16
        wfull = [0]*16
        ufull[0*4+0] = u[0]
        ufull[0*4+1] = u[1]
        ufull[1*4+0] = u[2]
        ufull[1*4+1] = u[3]
        ufull[0*4+2] = u[0]
        ufull[0*4+3] = u[1]
        ufull[1*4+2] = u[2]
        ufull[1*4+3] = u[3]
        vfull[2*4+2] = v[0]
        vfull[2*4+3] = v[1]
        vfull[3*4+2] = v[2]
        vfull[3*4+3] = v[3]
        wfull[0*4+2] = -w[0]
        wfull[0*4+3] = -w[1]
        wfull[1*4+2] = -w[2]
        wfull[1*4+3] = -w[3]
        terms.append((ufull, vfull, wfull))
    
    # M6 = (A21-A11)*(B11+B12)
    for (u, v, w) in winograd_222:
        ufull = [0]*16
        vfull = [0]*16
        wfull = [0]*16
        ufull[2*4+0] = u[0]
        ufull[2*4+1] = u[1]
        ufull[3*4+0] = u[2]
        ufull[3*4+1] = u[3]
        ufull[0*4+0] = -u[0]
        ufull[0*4+1] = -u[1]
        ufull[1*4+0] = -u[2]
        ufull[1*4+1] = -u[3]
        vfull[0*4+0] = v[0]
        vfull[0*4+1] = v[1]
        vfull[1*4+0] = v[2]
        vfull[1*4+1] = v[3]
        vfull[0*4+2] = v[0]
        vfull[0*4+3] = v[1]
        vfull[1*4+2] = v[2]
        vfull[1*4+3] = v[3]
        wfull[2*4+2] = w[0]
        wfull[2*4+3] = w[1]
        wfull[3*4+2] = w[2]
        wfull[3*4+3] = w[3]
        terms.append((ufull, vfull, wfull))
    
    # M7 = (A12-A22)*(B21+B22)
    for (u, v, w) in winograd_222:
        ufull = [0]*16
        vfull = [0]*16
        wfull = [0]*16
        ufull[0*4+2] = u[0]
        ufull[0*4+3] = u[1]
        ufull[1*4+2] = u[2]
        ufull[1*4+3] = u[3]
        ufull[2*4+2] = -u[0]
        ufull[2*4+3] = -u[1]
        ufull[3*4+2] = -u[2]
        ufull[3*4+3] = -u[3]
        vfull[2*4+0] = v[0]
        vfull[2*4+1] = v[1]
        vfull[3*4+0] = v[2]
        vfull[3*4+1] = v[3]
        vfull[2*4+2] = v[0]
        vfull[2*4+3] = v[1]
        vfull[3*4+2] = v[2]
        vfull[3*4+3] = v[3]
        wfull[0*4+0] = w[0]
        wfull[0*4+1] = w[1]
        wfull[1*4+0] = w[2]
        wfull[1*4+1] = w[3]
        terms.append((ufull, vfull, wfull))
    
    return terms

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

def _embed_4x4x4(terms, i_base, j_base, k_base, n, m, p):
    """Embed 4x4x4 terms into full-sized vectors."""
    full_terms = []
    for (u, v, w) in terms:
        ufull = [0] * (n * m)
        vfull = [0] * (m * p)
        wfull = [0] * (p * n)
        
        for a in range(4):
            for b in range(4):
                idx = (i_base + a) * m + (j_base + b)
                ufull[idx] = u[a * 4 + b]
        
        for b in range(4):
            for c in range(4):
                idx = (j_base + b) * p + (k_base + c)
                vfull[idx] = v[b * 4 + c]
        
        for c in range(4):
            for a in range(4):
                idx = (k_base + c) * n + (i_base + a)
                wfull[idx] = w[c * 4 + a]
        
        full_terms.append((ufull, vfull, wfull))
    return full_terms

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

def _strategy_strassen_444():
    """
    Use 4x4x4 Strassen for i:0-3, j:0-3, k:0-3 (rank 49),
    then schoolbook for remaining parts.
    
    Remaining:
    - j=4, k=0..3: 4*1*4 = 16 terms
    - j=0..3, k=4,5: 4*4*2 = 32 terms
    - j=4, k=4,5: 4*1*2 = 8 terms
    Total: 49 + 16 + 32 + 8 = 105
    """
    n, m, p = DIMENSIONS
    strassen_terms = _strassen_4x4x4_terms()
    
    terms_U = []
    terms_V = []
    terms_W = []
    
    # Embed 4x4x4 Strassen for i=0..3, j=0..3, k=0..3
    embedded = _embed_4x4x4(strassen_terms, 0, 0, 0, n, m, p)
    terms_U.extend([t[0] for t in embedded])
    terms_V.extend([t[1] for t in embedded])
    terms_W.extend([t[2] for t in embedded])
    
    # Schoolbook for j=4, k=0..3
    for i in range(n):
        for k in range(4):
            ufull = [0] * (n * m)
            vfull = [0] * (m * p)
            wfull = [0] * (p * n)
            ufull[i * m + 4] = 1
            vfull[4 * p + k] = 1
            wfull[k * n + i] = 1
            terms_U.append(ufull)
            terms_V.append(vfull)
            terms_W.append(wfull)
    
    # Schoolbook for j=0..3, k=4,5
    for i in range(n):
        for j in range(4):
            for k in range(4, p):
                ufull = [0] * (n * m)
                vfull = [0] * (m * p)
                wfull = [0] * (p * n)
                ufull[i * m + j] = 1
                vfull[j * p + k] = 1
                wfull[k * n + i] = 1
                terms_U.append(ufull)
                terms_V.append(vfull)
                terms_W.append(wfull)
    
    # Schoolbook for j=4, k=4,5
    for i in range(n):
        for k in range(4, p):
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
    
    if remaining() > 30:
        try:
            candidates.append(_strategy_strassen_444())
        except Exception as e:
            pass
    
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
