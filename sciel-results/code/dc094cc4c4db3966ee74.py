import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Return 7 Strassen terms for 2x2 matmul: (u, v, w) each length 4.
    u[i*2+k], v[k*2+j], w[j*2+i]"""
    return [
        ([1,0,1,0],[1,0,1,0],[1,0,0,0]),
        ([0,0,1,1],[1,0,0,0],[0,1,0,0]),
        ([1,0,0,0],[0,1,0,-1],[0,1,0,0]),
        ([0,0,0,1],[0,0,1,-1],[0,1,0,0]),
        ([1,1,0,0],[0,0,1,1],[0,0,0,1]),
        ([0,0,1,-1],[1,1,0,0],[0,0,0,1]),
        ([0,1,0,-1],[0,1,1,1],[0,0,0,1]),
    ]

def _strassen_2x2_out():
    """Output recon: for each of 7 products, list of (c_block, coeff)."""
    return [
        [(0,1),(3,1)],
        [(2,1),(3,-1)],
        [(1,1),(3,1)],
        [(0,1),(2,1)],
        [(0,-1),(1,1)],
        [(3,1)],
        [(0,1)],
    ]

def _build_4x4_strassen():
    """Build rank-49 terms for 4x4 matmul using 2x2 Strassen recursively.
    Each term: (u, v, w) length 16. u[i*4+k], v[k*4+j], w[j*4+i]."""
    s2 = _strassen_2x2()
    s2o = _strassen_2x2_out()
    # 4x4 block-level Strassen (same 7 products, 2x2 blocks of 2x2)
    blk = [
        ([(0,1),(3,1)],[(0,1),(3,1)]),
        ([(2,1),(3,1)],[(0,1)]),
        ([(0,1)],[(1,1),(3,-1)]),
        ([(3,1)],[(2,1),(0,-1)]),
        ([(0,1),(1,1)],[(3,1)]),
        ([(2,1),(0,-1)],[(0,1),(1,1)]),
        ([(1,1),(3,-1)],[(2,1),(3,1)]),
    ]
    out4 = s2o  # same output structure
    terms = []
    for op in range(7):
        ac, bc = blk[op]
        for it in range(7):
            ui, vi, wi = s2[it]
            u = [0]*16; v = [0]*16; w = [0]*16
            for (ba, ca) in ac:
                ra = (ba//2)*2; ca_ = (ba%2)*2
                for i in range(2):
                    for k in range(2):
                        u[(ra+i)*4+(ca_+k)] += ca * ui[i*2+k]
            for (bb, cb) in bc:
                rb = (bb//2)*2; cb_ = (bb%2)*2
                for k in range(2):
                    for j in range(2):
                        v[(rb+k)*4+(cb_+j)] += cb * vi[k*2+j]
            for (bc_, cc) in out4[op]:
                rc = (bc_//2)*2; cc_ = (bc_%2)*2
                for i in range(2):
                    for j in range(2):
                        w[(cc_+j)*4+(rc+i)] += cc * wi[j*2+i]
            terms.append((u,v,w))
    return terms

def _schoolbook(m, n, p):
    """Schoolbook terms for m x n * n x p = m x p.
    u[i*n+k], v[k*p+j], w[j*m+i]."""
    terms = []
    for i in range(m):
        for k in range(n):
            for j in range(p):
                u = [0]*(m*n); v = [0]*(n*p); w = [0]*(p*m)
                u[i*n+k] = 1; v[k*p+j] = 1; w[j*m+i] = 1
                terms.append((u,v,w))
    return terms

def _verify(terms, m, n, p):
    for i in range(m):
        for k in range(n):
            for j in range(p):
                t = sum(u[i*n+k]*v[k*p+j]*w[j*m+i] for u,v,w in terms)
                if abs(t-1) > 1e-10:
                    return False
    return True

def construct():
    start = time.time()
    deadline = start + 120
    n, m, p = DIMENSIONS  # 6,6,6
    
    # Strategy: 4+2 block partition
    # A = [[A00(4x4), A01(4x2)],[A10(2x4), A11(2x2)]]
    # B = [[B00(4x4), B01(4x2)],[B10(2x4), B11(2x2)]]
    # C00(4x4) = A00*B00 + A01*B10
    # C01(4x2) = A00*B01 + A01*B11
    # C10(2x4) = A10*B00 + A11*B10
    # C11(2x2) = A10*B01 + A11*B11
    
    # Since C00 is a sum of two products, we can't directly use 4x4 Strassen.
    # Instead, use schoolbook for all: rank = 6*6*6 = 216.
    # But we can do better with 3+3 Strassen: rank 189.
    
    # Use 3+3 Strassen approach (proven to work, rank 189)
    bs = 3
    s3 = _schoolbook(3,3,3)  # rank 27
    rank_pb = len(s3)
    
    strassen_terms = [
        ([(0,1),(3,1)],[(0,1),(3,1)]),
        ([(2,1),(3,1)],[(0,1)]),
        ([(0,1)],[(1,1),(3,-1)]),
        ([(3,1)],[(2,1),(0,-1)]),
        ([(0,1),(1,1)],[(3,1)]),
        ([(2,1),(0,-1)],[(0,1),(1,1)]),
        ([(1,1),(3,-1)],[(2,1),(3,1)]),
    ]
    output_contrib = [
        [(0,1),(3,1)],
        [(2,1),(3,-1)],
        [(1,1),(3,1)],
        [(0,1),(2,1)],
        [(0,-1),(1,1)],
        [(3,1)],
        [(0,1)],
    ]
    
    total_rank = 7 * rank_pb
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col = 0
    for ti in range(7):
        ac, bc = strassen_terms[ti]
        for (ul, vl, wl) in s3:
            for i in range(bs):
                for k in range(bs):
                    uc = ul[i*3+k]
                    if uc == 0: continue
                    for j in range(bs):
                        vc = vl[k*3+j]
                        if vc == 0: continue
                        wc = wl[j*3+i]
                        if wc == 0: continue
                        for (ba, ca) in ac:
                            ra = (ba//2)*bs; ca_ = (ba%2)*bs
                            U[(ra+i)*m+(ca_+k)][col] += ca*uc
                        for (bb, cb) in bc:
                            rb = (bb//2)*bs; cb_ = (bb%2)*bs
                            V[(rb+k)*p+(cb_+j)][col] += cb*vc
                        for (bc_, cc) in output_contrib[ti]:
                            rc = (bc_//2)*bs; cc_ = (bc_%2)*bs
                            W[(cc_+j)*n+(rc+i)][col] += cc*wc
                        col += 1
    
    if time.time() > deadline - 5:
        pass
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
