import numpy as np

DIMENSIONS = [3, 4, 7]

def _verify(U, V, W):
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Tensor T[i*m+j, j*p+k, k*n+i] = 1
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # Build T from U,V,W: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # But indices must be consistent: a=i*m+j, b=j*p+k, c=k*n+i
    # We need T to equal the multiplication tensor: for each (i,j,k), the
    # entry (a=i*m+j, b=j*p+k, c=k*n+i) should be 1, and all "inconsistent"
    # triples 0.
    # Reconstruct full 3-tensor.
    T = np.einsum('ar,br,cr->abc', U, V, W)
    # Build target
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    return np.allclose(T, target, atol=1e-9)

def construct():
    n, m, p = DIMENSIONS
    # Build a candidate decomposition via block Strassen scheme.
    terms = []  # each: (u_vec, v_vec, w_vec) as index lists or dense

    A_dim = n*m
    B_dim = m*p
    C_dim = p*n

    def Aidx(i, j):
        return i*m + j
    def Bidx(j, k):
        return j*p + k
    def Cidx(k, i):
        return k*n + i

    # We'll accumulate dense vectors for each term.
    def add_term(au, bv, cw):
        u = np.zeros(A_dim); v = np.zeros(B_dim); w = np.zeros(C_dim)
        for (i,j) in au:
            u[Aidx(i,j)] += 1
        for (j,k) in bv:
            v[Bidx(j,k)] += 1
        for (k,i) in cw:
            w[Cidx(k,i)] += 1
        terms.append((u, v, w))

    # Naive helper: multiply A-block (rows i0..i1, cols j0..j1) by B-block
    # (rows j0..j1, cols k0..k1) contributing to C-block (rows i0..i1, cols k0..k1)
    def naive_block(i0,i1,j0,j1,k0,k1, sign=1):
        for i in range(i0,i1):
            for k in range(k0,k1):
                for j in range(j0,j1):
                    add_term([(i,j)], [(j,k)], [(k,i)])
        # sign handled by scaling later; here sign always +1

    # We need signed linear forms for Strassen. Represent terms as
    # (list of (idx, coeff)) for u, v, w.
    terms2 = []
    def add2(ul, vl, wl):
        terms2.append((ul, vl, wl))

    # Strassen on 2x2 blocks: A block rows (i0,i0+1) cols (j0,j0+1),
    # B block rows (j0,j0+1) cols (k0,k0+1) -> C block rows (i0,i0+1) cols (k0,k0+1)
    def strassen(i0, j0, k0):
        # A11=(i0,j0), A12=(i0,j0+1), A21=(i0+1,j0), A22=(i0+1,j0+1)
        # B11=(j0,k0), B12=(j0,k0+1), B21=(j0+1,k0), B22=(j0+1,k0+1)
        # C11=(k0,i0), C12=(k0+1,i0), C21=(k0,i0+1), C22=(k0+1,i0+1)
        def A(i,j): return [(i,j,1)]
        def B(j,k): return [(j,k,1)]
        def C(k,i): return [(k,i,1)]
        def addcoef(lst, idx, c):
            lst.append((idx, c))
        # M1=(A11+A22)(B11+B22)
        m1u = A(i0,j0)+A(i0+1,j0+1)
        m1v = B(j0,k0)+B(j0,k0+1)
        # contributes to Z11,Z22
        # We'll just emit 7 products and then linear combos are absorbed by
        # adding to output with coefficients.
        # Represent each Strassen product as a term (u,v, base_w) but w must be
        # a single C index for rank-1. Instead expand: each product M is a 2x2
        # matrix; we distribute.
        # Simpler: do naive for correctness but that loses Strassen savings.
        # To keep verifiability and still save, use Strassen with w as sum:
        # but rank-1 requires w a single basis vector. So we expand each M's
        # 4 entries into 4 rank-1 terms -> 7*4=28, worse than naive 8.
        # Therefore Strassen at scalar level needs the M's to be scalars, i.e.
        # 1x1 blocks. Not applicable. Return naive.
        naive_block(i0,i0+2,j0,j0+2,k0,k0+2)

    # Given scalar-level Strassen doesn't reduce rank for 2x2 (needs 1x1),
    # the block approach above gives naive rank. So just do full naive = 84.
    # To get ANY saving we need a genuine bilinear algorithm; the safe,
    # verifiable improvement is the block naive which equals 84. So we fall
    # back to trivial to guarantee validity.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                add_term([(i,j)], [(j,k)], [(k,i)])

    U = np.zeros((A_dim, len(terms)))
    V = np.zeros((B_dim, len(terms)))
    W = np.zeros((C_dim, len(terms)))
    for r,(u,v,w) in enumerate(terms):
        U[:,r]=u; V[:,r]=v; W[:,r]=w

    if _verify(U,V,W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback trivial (guaranteed valid)
    rank = n*m*p
    U = np.zeros((A_dim, rank)); V=np.zeros((B_dim,rank)); W=np.zeros((C_dim,rank))
    col=0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j,col]=1.0; V[j*p+k,col]=1.0; W[k*n+i,col]=1.0
                col+=1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
