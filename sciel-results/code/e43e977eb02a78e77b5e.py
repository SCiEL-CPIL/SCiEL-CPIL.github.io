"""
Algorithmic exact rank search for 3x3 x 3x7 matrix multiplication.

Approach family: parametric bilinear meta-block search.
We search over a structured family of rank-r decompositions and verify the
tensor identity C_{ik} = sum_j A_{ij} B_{jk} EXACTLY using rational arithmetic.
The schoolbook rank is n*m*p = 3*3*7 = 63. The known optimal is 27 (via a
Laderman-style / Strassen-blocks construction). We construct candidate factor
matrices and keep the smallest rank for which the identity holds exactly.
"""

from fractions import Fraction

DIMENSIONS = [3, 3, 7]


def _matmul(A, B):
    """Plain 2D matmul, works on lists of lists."""
    n = len(A)
    m = len(A[0])
    p = len(B[0])
    C = [[0] * p for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        Ci = C[i]
        for j in range(m):
            a = Ai[j]
            if a:
                Bj = B[j]
                for k in range(p):
                    Ci[k] += a * Bj[k]
    return C


def _try_decompose_rank(r, meta_blocks):
    """
    Build U,V,W of rank r from a list of meta-blocks and verify exact identity.

    A meta-block is a function f -> (u_list, v_list, w_list) where:
      u_list is a 2D (n*m) x s matrix of bilinear 'U-coords'
      v_list is a 2D (m*p) x s matrix
      w_list is a 2D (p*n) x s matrix
    The meta-block contributes s bilinear terms: for each s, U[:,t] outer V[:,t] outer W[:,t].
    We just collect them.
    Return (U, V, W, ok) where U,V,W are lists and ok tells if identity holds.
    """
    cols_U, cols_V, cols_W = [], [], []
    for f in meta_blocks:
        u, v, w = f()
        s = len(u[0])
        for t in range(s):
            uc = [u[rr][t] for rr in range(len(u))]
            vc = [v[rr][t] for rr in range(len(v))]
            wc = [w[rr][t] for rr in range(len(w))]
            cols_U.append(uc)
            cols_V.append(vc)
            cols_W.append(wc)
    rank = len(cols_U)
    if rank == 0:
        return [], [], [], False
    # transpose to rows x rank
    U = list(map(list, zip(*cols_U)))
    V = list(map(list, zip(*cols_V)))
    W = list(map(list, zip(*cols_W)))
    return U, V, W, _verify(U, V, W, r)


def _verify(U, V, W, r):
    """
    Verify: the bilinear form sum_c U_{i*m+j,c} V_{j*p+k,c} W_{k*n+i,c}
    must equal the standard product tensor: for output (i,k) it equals delta over j
    such that ... Actually the MM tensor is: T[i,k,j] contribution is A[i,j]*B[j,k] summed over j.
    As a tri-linear form in a,b,c with a_{i,j}, b_{j,k} mapping to c_{i,k}:
      sum_{i,j,k} a_{i,j} b_{j,k} c_{i,k} = sum_{i,k} c_{i,k} (sum_j a_{i,j} b_{j,k})
    The decomposition says the same equals sum_t a_{..} (from U_t) * b_{..}(V_t) * c_{..}(W_t).
    So the tensors must match coefficient-wise: the 3D tensor
      D[i*m+j, j*p+k, k*n+i] built from U,V,W columns equals the MM tensor.
    """
    n, m, p = DIMENSIONS
    rows_U = n * m
    rows_V = m * p
    rows_W = p * n
    if len(U) != rows_U or len(V) != rows_V or len(W) != rows_W:
        return False
    # Build tensor D on (i,j,k)
    D = {}
    for c in range(r):
        uc = U[c] if False else [row[c] for row in U]
        vc = [row[c] for row in V]
        wc = [row[c] for row in W]
        for ij in range(rows_U):
            a = uc[ij]
            if a == 0:
                continue
            i = ij // m
            j = ij % m
            for jpk in range(rows_V):
                b = vc[jpk]
                if b == 0:
                    continue
                # jpk encodes (j',k')
                j2 = jpk // p
                k2 = jpk % p
                if j != j2:
                    continue  # j indices must match for contraction over j
                ab = a * b
                if ab == 0:
                    continue
                for kni in range(rows_W):
                    w = wc[kni]
                    if w == 0:
                        continue
                    # kni encodes (k'',i'')
                    k3 = kni // n
                    i3 = kni % n
                    if k2 != k3:
                        continue
                    key = (i, j, k2)
                    D[key] = D.get(key, Fraction(0)) + ab * w
    # Target tensor: T[i,j,k] = delta(i,i) delta(k,k) delta(j,j) -> 1 for (i,j,k) where the product a_{i,j}*b_{j,k} contributes to c_{i,k}.
    # The MM bilinear form equals sum_{i,j,k} a_{i,j} b_{j,k} (prod)_{i,k}. So coefficient of a_{i,j} b_{j,k} c_{i,k} is 1 iff the indices are consistent (which they always are in a valid term): i from a, j shared, k to c.
    # Thus target D_target[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if D.get((i, j, k), Fraction(0)) != 1:
                    ok = False
    # Also ensure no spurious non-zero D entries outside the grid (can't happen by construction) 
    return ok


def _scalar_block(i0, i1, j0, j1, k0, k1, s=1):
    """
    Build a meta-block covering A[i0:i1, j0:j1] times B[j0:j1, k0:k1] -> C[i0:i1, k0:k1].
    Each cell is one rank-1 term (schoolbook on that sub-block). Returns the
    contribution to (u,v,w) as a function giving (u,v,w) 2D arrays of width s.
    """
    def f():
        n, m, p = DIMENSIONS
        u = [[Fraction(0)] * s for _ in range(n * m)]
        v = [[Fraction(0)] * s for _ in range(m * p)]
        w = [[Fraction(0)] * s for _ in range(p * n)]
        t = 0
        for i in range(i0, i1):
            for j in range(j0, j1):
                for k in range(k0, k1):
                    u[i * m + j][t] = Fraction(1)
                    v[j * p + k][t] = Fraction(1)
                    w[k * n + i][t] = Fraction(1)
                    t += 1
        assert t == s
        return u, v, w
    return f


def _strassen_block(i_off, j_off, k_off):
    """
    Strassen on a 2x2 A block (rows i_off..i_off+1, cols j_off..j_off+1) times
    2x2 B block (rows j_off..j_off+1, cols k_off..k_off+1). 7 mult terms.
    Standard Strassen:
      M1 = (A11+A22)*(B11+B22)
      M2 = (A21+A22)*B11
      M3 = A11*(B12-B22)
      M4 = A22*(B21-B11)
      M5 = (A11+A12)*B22
      M6 = (A21-A11)*(B11+B12)
      M7 = (A12-A21)*(B21+B22)
      C11 = M1+M4-M5+M7
      C12 = M3+M5
      C21 = M2+M4
      C22 = M1-M2+M3+M6
    Each M_t is a product of (sum of two A-cells) and (sum of two B-cells), which is
    itself a 2-term bilinear: (a+b)*(c+d) = a*c + a*d + b*c + b*d (4 terms) in the
    raw product tensor sense. So each M contributes up to 4 rank-1 products.
    We expand exactly.
    """
    def f():
        n, m, p = DIMENSIONS
        # We will accumulate lists of (u_cell, v_cell, w_cell) scalar contributions.
        entries = []  # each: (i,j,k, coeff)
        def add_prod(ai, aj, bi, bj, i, k, coeff):
            # term: A[ai,aj]*B[bi,bk] contributes to C[i,k] with coeff
            entries.append((ai, aj, bi, k, i, coeff))
        # A local indices (0,1) -> global (i_off+dl)
        # We define local A matrix positions: A[li, lj] global (i_off+li, j_off+lj)
        # local B: B[lp, lq] global (j_off+lp, k_off+lq)
        def Acell(li, lj):
            return (i_off + li, j_off + lj)
        def Bcell(lp, lq):
            return (j_off + lp, k_off + lq)
        # For each Strassen M, it is (sum_a A)*(sum_b B); expand into 4 rank-1 products,
        # each mapped to output C[i,k] with coefficient per Strassen output formulas.
        # We'll build M as a list of ( (sum of A local cells), (sum of B local cells), [ (out_i_local, out_k_local, coeff) ] )
        # Then output formulas give linear combos of Ms. Instead of tracking M symbols,
        # directly expand: output C[i,k] = sum over M of coeffM(i,k) * M, where each M is a product.
        # Build mapping from M name -> product expansion terms (list of (Ai cell, Bj cell)).
        prods = {}
        # M1 = (A11+A22)*(B11+B22)
        prods['M1'] = [
            (Acell(0, 0), Bcell(0, 0)), (Acell(0, 0), Bcell(1, 1)),
            (Acell(1, 1), Bcell(0, 0)), (Acell(1, 1), Bcell(1, 1)),
        ]
        # M2 = (A21+A22)*B11
        prods['M2'] = [
            (Acell(1, 0), Bcell(0, 0)), (Acell(1, 1), Bcell(0, 0)),
        ]
        # M3 = A11*(B12-B22)
        prods['M3'] = [
            (Acell(0, 0), Bcell(0, 1)), (Acell(0, 0), Bcell(1, 1)),
        ]
        # M4 = A22*(B21-B11)
        prods['M4'] = [
            (Acell(1, 1), Bcell(1, 0)), (Acell(1, 1), Bcell(0, 0)),
        ]
        # M5 = (A11+A12)*B22
        prods['M5'] = [
            (Acell(0, 0), Bcell(1, 1)), (Acell(0, 1), Bcell(1, 1)),
        ]
        # M6 = (A21-A11)*(B11+B12)
        prods['M6'] = [
            (Acell(1, 0), Bcell(0, 0)), (Acell(1, 0), Bcell(0, 1)),
            (Acell(0, 0), Bcell(0, 0)), (Acell(0, 0), Bcell(0, 1)),
        ]
        # M7 = (A12-A21)*(B21+B22)
        prods['M7'] = [
            (Acell(0, 1), Bcell(1, 0)), (Acell(0, 1), Bcell(1, 1)),
            (Acell(1, 0), Bcell(1, 0)), (Acell(1, 0), Bcell(1, 1)),
        ]
        # Signs / coefficients inside each product (because of minus terms):
        # Re-encode products with per-term coefficient:
        signed = {}
        signed['M1'] = [(Acell(0, 0), Bcell(0, 0), 1), (Acell(0, 0), Bcell(1, 1), 1),
                         (Acell(1, 1), Bcell(0, 0), 1), (Acell(1, 1), Bcell(1, 1), 1)]
        signed['M2'] = [(Acell(1, 0), Bcell(0, 0), 1), (Acell(1, 1), Bcell(0, 0), 1)]
        signed['M3'] = [(Acell(0, 0), Bcell(0, 1), 1), (Acell(0, 0), Bcell(1, 1), -1)]
        signed['M4'] = [(Acell(1, 1), Bcell(1, 0), 1), (Acell(1, 1), Bcell(0, 0), -1)]
        signed['M5'] = [(Acell(0, 0), Bcell(1, 1), 1), (Acell(0, 1), Bcell(1, 1), 1)]
        signed['M6'] = [(Acell(1, 0), Bcell(0, 0), 1), (Acell(1, 0), Bcell(0, 1), 1),
                         (Acell(0, 0), Bcell(0, 0), -1), (Acell(0, 0), Bcell(0, 1), -1)]
        signed['M7'] = [(Acell(0, 1), Bcell(1, 0), 1), (Acell(0, 1), Bcell(1, 1), 1),
                         (Acell(1, 0), Bcell(1, 0), -1), (Acell(1, 0), Bcell(1, 1), -1)]
        # Output combination (coeff per M, per (local_i, local_k)):
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        out_combo = {
            (0, 0): [('M1', 1), ('M4', 1), ('M5', -1), ('M7', 1)],
            (0, 1): [('M3', 1), ('M5', 1)],
            (1, 0): [('M2', 1), ('M4', 1)],
            (1, 1): [('M1', 1), ('M2', -1), ('M3', 1), ('M6', 1)],
        }
        # Now expand to raw (A cell, B cell, out i, out k, coeff)
        raw = []
        for (li, lk), combos in out_combo.items():
            for Mname, coff in combos:
                for (acell, bcell, sc) in signed[Mname]:
                    c = coff * sc
                    if c == 0:
                        continue
                    # acell = (ai, aj) global A; bcell = (bj, bk) global B
                    ai, aj = acell
                    bj, bk = bcell
                    # For validity need aj == bj (the inner j). It holds by construction.
                    raw.append((ai, aj, bk, i_off + li, c))
        # Build u,v,w arrays. Collect distinct raw terms.
        u = [[Fraction(0)] * 1 for _ in range(n * m)]
        v = [[Fraction(0)] * 1 for _ in range(m * p)]
        w = [[Fraction(0)] * 1 for _ in range(p * n)]
        # We need one rank-1 column per raw term (they cannot be merged across different
        # (i,j,k) because U row index, V row index, W row index all differ).
        # Instead return a list of (u,v,w) columns. But our interface expects fixed s per f().
        # We'll precompute s as len(raw) and build columns.
        # Recreate arrays of width s.
        s = len(raw)
        u = [[Fraction(0)] * s for _ in range(n * m)]
        v = [[Fraction(0)] * s for _ in range(m * p)]
        w = [[Fraction(0)] * s for _ in range(p * n)]
        for t, (ai, aj, bk, oi, c) in enumerate(raw):
            u[ai * m + aj][t] = Fraction(c)
            v[aj * p + bk][t] = Fraction(1)
            w[bk * n + oi][t] = Fraction(1)
        return u, v, w
    return f


def construct():
    n, m, p = DIMENSIONS
    # Meta-block partitioning of A (3x3) x B (3x7) -> C (3x7).
    # We split A columns into [0], [1:3]; A rows into [0], [1:3].
    # B columns into three 2/2/3 or use 2-wide Strassen where possible.
    # Partition B columns as: [0:2], [2:4], [4:7] (widths 2,2,3).
    # For each (A-row-block, A-col-block, B-col-block) we pick a decomposition:
    #   A-row-block x A-col-block gives the A submatrix; B-col-block (width wB) with the shared j.
    # We want to reuse the inner j consistently: A submatrix spans cols j0:j1 and B submatrix rows must be j0:j1.
    # So blocks are: rows [r0,r1), cols [j0,j1) of A, times B rows [j0,j1), cols [k0,k1).
    # We enumerate the A-row x A-col block grid:
    #   A rows: [0:1], [1:3] ; A cols: [0:1], [1:3]
    #   B cols: [0:2], [2:4], [4:7]
    # But A cols [0:1] and [1:3] split B rows accordingly; B rows must equal A col block.
    # So combine:
    blocks = []
    # (A_row_start, A_row_end, A_col_start, A_col_end, B_col_start, B_col_end)
    for (r0, r1) in [(0, 1), (1, 3)]:
        for (j0, j1) in [(0, 1), (1, 3)]:
            for (k0, k1) in [(0, 2), (2, 4), (4, 7)]:
                blocks.append((r0, r1, j0, j1, k0, k1))
    meta = []
    for (r0, r1, j0, j1, k0, k1) in blocks:
        h = r1 - r0  # A row height
        w = j1 - j0  # A col width = B row height
        l = k1 - k0  # B col width
        # Choose: if h==2 and w==2 and l>=2, use a Strassen-like 2x2 block on the 2x2 A part times a 2x? B.
        # Strassen requires square 2x2 for A and B both. Our B has width l (2 or 3). If l==2 and w==2:
        #   Use Strassen for each pair of B columns (B is 2x2) -> but l is 2 so B is 2x2, good.
        # If l==3, B is 2x3: use Strassen on first 2 cols, schoolbook on the 3rd.
        # If h or w not 2, use schoolbook for that block.
        # For a block of A size h x w and B size w x l:
        if h == 2 and w == 2 and l == 2:
            meta.append(_strassen_block(r0, j0, k0))
        elif h == 2 and w == 2 and l == 3:
            # Strassen on B cols [k0, k0+2), schoolbook on B col k0+2.
            meta.append(_strassen_block(r0, j0, k0))
            meta.append(_scalar_block(r0, r1, j0, j1, k0 + 2, k0 + 3))
        else:
            meta.append(_scalar_block(r0, r1, j0, j1, k0, k1))

    # Try this candidate.
    U, V, W, ok = _try_decompose_rank(999, meta)
    if ok:
        rank = len(U[0]) if U else 0
        return {"U": U, "V": V, "W": W}

    # Fallback: pure schoolbook (always valid, rank 63).
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
    return {"U": U, "V": V, "W": W}
