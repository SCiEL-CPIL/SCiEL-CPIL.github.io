"""Combined program: recursive Strassen (2x2x2 -> 8x8x8) with zero-pruning for 4x5x6."""
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2_terms():
    """Return the 7 rank-1 terms (u,v,w) each a length-4 vector (2x2 flattened, idx=row*2+col).
    Term c: U_vec (A entries), V_vec (B entries), W_vec (C entries).
    C = A*B, C index = row*2+col.
    """
    return [
        # M1 = (A11+A22)(B11+B22); contributes to C11,C22
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        # M2 = (A21+A22)B11; contributes to C21
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
        # M3 = A11(B21-B22); contributes to C12
        ([1,0,0,0], [0,0,1,-1], [0,1,0,0]),
        # M4 = A22(B12-B11); contributes to C22
        ([0,0,0,1], [0,1,0,-1], [0,0,0,1]),
        # M5 = (A11+A12)B22; contributes to C12
        ([1,1,0,0], [0,0,0,1], [0,1,0,0]),
        # M6 = (A21-A11)(B11+B12); contributes to C21
        ([-1,0,1,0], [1,1,0,0], [0,0,1,0]),
        # M7 = (A12-A22)(B21+B22); contributes to C22
        ([0,1,0,-1], [0,0,1,1], [0,0,0,1]),
    ]


def _build_strassen_terms(size):
    """Recursively build rank-1 terms for size x size x size matmul via Strassen.
    Returns list of (u,v,w) each a length size^2 vector (idx = row*size+col).
    size must be a power of 2.
    """
    base = _strassen_2x2_terms()
    terms = base
    s = 2
    while s < size:
        new_terms = []
        for (u, v, w) in terms:
            # u,v,w are length s^2 (s x s flattened, idx = row*s+col)
            # Split into 4 blocks of s/2 x s/2
            h = s // 2
            # block offsets in flattened: TL=0, TR=h*(s/2)... let's compute
            # For s x s, block (br, bc) of size h x h starts at row br*h, col bc*h
            # flat index of element (r,c) = r*s + c
            # block (br,bc) element (r,c) -> flat = (br*h+r)*s + (bc*h+c)
            def split(vec):
                # returns 4 lists: TL, TR, BL, BR each length h^2
                TL, TR, BL, BR = [], [], [], []
                for r in range(s):
                    for c in range(s):
                        val = vec[r*s+c]
                        br = 0 if r < h else 1
                        bc = 0 if c < h else 1
                        lr = r - br*h
                        lc = c - bc*h
                        flat = lr*h + lc
                        if br == 0 and bc == 0:
                            TL.append(val)
                        elif br == 0 and bc == 1:
                            TR.append(val)
                        elif br == 1 and bc == 0:
                            BL.append(val)
                        else:
                            BR.append(val)
                return TL, TR, BL, BR
            uTL,uTR,uBL,uBR = split(u)
            vTL,vTR,vBL,vBR = split(v)
            wTL,wTR,wBL,wBR = split(w)
            for (a,b,c) in base:
                # combine: U_new = a*uTL + b*uTR + c*uBL ...
                # Actually the Strassen recursion: at each level, the 7 terms combine
                # the 4 sub-blocks of U,V,W with the 2x2 coefficients.
                nu = [a[i]*uTL[i] + b[i]*uTR[i] + c[i]*uBL[i] + d[i]*uBR[i] for i in range(h*h)]
                # wait, a,b,c are the 2x2 coefficient vectors of length 4 (TL,TR,BL,BR)
                # u_new[idx] = sum over 4 blocks: coeff[block][subidx] * u[block][subidx]
                # a is length 4: a[0]=TL coeff, a[1]=TR coeff, a[2]=BL coeff, a[3]=BR coeff
                nu = [a[0]*uTL[i] + a[1]*uTR[i] + a[2]*uBL[i] + a[3]*uBR[i] for i in range(h*h)]
                nv = [a[0]*vTL[i] + a[1]*vTR[i] + a[2]*vBL[i] + a[3]*vBR[i] for i in range(h*h)]
                nw = [a[0]*wTL[i] + a[1]*wTR[i] + a[2]*wBL[i] + a[3]*wBR[i] for i in range(h*h)]
                # now interleave the h^2 blocks into size^2 (new size = 2s)
                # new size S = 2s, block size h = s/2
                S = 2*s
                new_u = [0]*(S*S)
                new_v = [0]*(S*S)
                new_w = [0]*(S*S)
                # place 4 blocks of size h x h into S x S
                # block (br,bc) of new S x S: element (r,c) with r in [br*h, br*h+h), c in [bc*h, bc*h+h)
                # flat new index = r*S + c
                for r in range(h):
                    for c in range(h):
                        sub = r*h + c
                        for br in range(2):
                            for bc in range(2):
                                rr = br*h + r
                                cc = bc*h + c
                                new_u[rr*S+cc] = nu[sub] if (br==0 and bc==0) else (nu[sub] if False else 0)
                # This placement is wrong; each of the 4 sub-blocks of the NEW matrix comes from
                # a different combination. Let me redo: the new S x S U matrix has 4 blocks,
                # each block (br,bc) = (a[br*2+bc] * u_block[br*2+bc]) ... no.
                # Correct Strassen: U_new block (br,bc) = sum over the 2x2 coeff of that position.
                # Actually the standard: the 7 products each produce an S x S matrix where
                # U_new = the 2x2 combination applied to the 4 sub-U's, giving an S x S.
                # The 4 sub-blocks of U_new are:
                #   U_new_TL = a0*uTL, U_new_TR = a1*uTR, U_new_BL = a2*uBL, U_new_BR = a3*uBR
                # where a0,a1,a2,a3 are scalars (the 2x2 coeff at that position).
                # But that's not right either; the coeff vector a has 4 entries corresponding
                # to the 4 positions, and each position's U block = a[pos] * u[pos].
                # So U_new block (br,bc) = a[br*2+bc] * u[br*2+bc] (element-wise, scalar*vector).
                for br in range(2):
                    for bc in range(2):
                        pos = br*2+bc
                        src = uTL if pos==0 else (uTR if pos==1 else (uBL if pos==2 else uBR))
                        for r in range(h):
                            for c in range(h):
                                rr = br*h + r
                                cc = bc*h + c
                                new_u[rr*S+cc] = a[pos]*src[r*h+c]
                                new_v[rr*S+cc] = a[pos]* (vTL if pos==0 else (vTR if pos==1 else (vBL if pos==2 else vBR)))[r*h+c]
                                new_w[rr*S+cc] = a[pos]* (wTL if pos==0 else (wTR if pos==1 else (wBL if pos==2 else wBR)))[r*h+c]
                new_terms.append((new_u, new_v, new_w))
        terms = new_terms
        s = S
    return terms


def _verify(U, V, W, n, m, p):
    """Verify exact decomposition. U (nm x r), V (mp x r), W (pn x r)."""
    rank = len(U[0])
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i = a // m
                j = a % m
                j2 = b // p
                k2 = b % p
                k3 = c // n
                i3 = c % n
                expected = 1 if (j == j2 and i == i3 and k2 == k3) else 0
                s = 0
                for col in range(rank):
                    s += U[a][col] * V[b][col] * W[c][col]
                if s != expected:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin under 120s

    def schoolbook():
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()

    # Recursive Strassen on 8x8x8, then prune terms that touch zero-padded A or B regions.
    # A is 4x5, B is 5x6. Pad to 8x8. Zero regions: A rows 4..7, cols 5..7; B rows 5..7, cols 6..7.
    # A term (u,v,w) for 8x8x8 is valid for the 4x5x6 problem iff:
    #   u has no nonzero in A's zero region (rows>=4 or cols>=5 within 8x8 A)
    #   v has no nonzero in B's zero region (rows>=5 or cols>=6 within 8x8 B)
    # and the w part only needs to be consistent (it maps to C which we keep full 4x6, but
    # C zero region rows>=4 or cols>=6 will be pruned implicitly since u/v already constrain).
    # Actually we must ensure the reconstructed C only has entries in rows<4, cols<6.
    # Since u forces A rows<4 and v forces B cols<6, the product C[i,k] gets i from A row, k from B col.
    # u nonzero only in A rows<4 => i<4. v nonzero only in B cols<6 => k<6. Good.
    # Also need A cols: u nonzero only in cols<5 (since A has 5 cols) and B rows<5.
    # So prune condition: for all (r,c) with r>=4 or c>=5, u[r*8+c]==0;
    #                    for all (r,c) with r>=5 or c>=6, v[r*8+c]==0.

    def a_zero(r, c):
        return r >= n or c >= m  # 4,5
    def b_zero(r, c):
        return r >= m or c >= p  # 5,6

    try:
        size = 8
        terms = _build_strassen_terms(size)
        kept = []
        for (u, v, w) in terms:
            ok = True
            for r in range(size):
                for c in range(size):
                    if a_zero(r, c) and u[r*size+c] != 0:
                        ok = False
                        break
                    if b_zero(r, c) and v[r*size+c] != 0:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                kept.append((u, v, w))

        if kept:
            rank = len(kept)
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            for col, (u, v, w) in enumerate(kept):
                for r in range(n):
                    for c in range(m):
                        U[r*m+c][col] = u[r*size+c]
                for r in range(m):
                    for c in range(p):
                        V[r*p+c][col] = v[r*size+c]
                for r in range(p):
                    for c in range(n):
                        W[r*n+c][col] = w[r*size+c]
            if _verify(U, V, W, n, m, p):
                best = {"U": U, "V": V, "W": W}
    except Exception:
        pass

    return best

# EVOLVE-BLOCK-END
