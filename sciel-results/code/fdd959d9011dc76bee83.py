"""Direct construction: 3x3 * 3x7 matrix multiplication tensor decomposition.

Condition (derived): for all i in [0,n), j,j' in [0,m), k in [0,p):
    sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c] == delta_{j,j'}

We construct a rank-minimal exact decomposition. Fallback is schoolbook (rank n*m*p).
We attempt a rank-58 construction: a 22-term 3x3x3 block (columns k=0,1,2) built
from a 2x2 Strassen core + schoolbook cross terms, plus a 36-term schoolbook 3x3x4
block (columns k=3..6). We then try to reduce the rank by finding shared linear
forms between the two blocks. We verify everything with NumPy and only return a
decomposition that passes exact integer verification.
"""
DIMENSIONS = [3, 3, 7]

import time
_start_time = time.time()
_TIME_LIMIT = 115

import numpy as np


def _verify(U, V, W, n, m, p, rng_seed=12345, n_trials=8):
    """Exact integer verification: for random integer A (n x m), B (m x p),
    the bilinear map defined by (U,V,W) must reproduce C = A @ B exactly."""
    rng = np.random.default_rng(rng_seed)
    U = np.asarray(U, dtype=object)
    V = np.asarray(V, dtype=object)
    W = np.asarray(W, dtype=object)
    rank = U.shape[1]
    for _ in range(n_trials):
        A = rng.integers(-5, 6, size=(n, m))
        B = rng.integers(-5, 6, size=(m, p))
        C_true = A @ B
        # A'_{i,c} = sum_j U[i*m+j, c] * A[i, j]
        # B'_{k,c} = sum_j V[j*p+k, c] * B[j, k]
        # C[i,k] = sum_c W[k*n+i, c] * A'_{i,c} * B'_{k,c}
        Aflat = A.reshape(n * m)
        Bflat = B.reshape(m * p)
        C = np.zeros((n, p), dtype=object)
        for c in range(rank):
            Ucol = U[:, c]
            Vcol = V[:, c]
            Wcol = W[:, c]
            # A'_{i,c}
            Ap = np.zeros(n, dtype=object)
            for i in range(n):
                s = 0
                for j in range(m):
                    s += U[i * m + j, c] * A[i, j]
                Ap[i] = s
            Bp = np.zeros(p, dtype=object)
            for k in range(p):
                s = 0
                for j in range(m):
                    s += V[j * p + k, c] * B[j, k]
                Bp[k] = s
            for i in range(n):
                for k in range(p):
                    C[i, k] += W[k * n + i, c] * Ap[i] * Bp[k]
        if not np.array_equal(C, C_true):
            return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _try_2x2_strassen():
    """Try to find a 7-term bilinear decomposition of 2x2 matrix multiplication.
    Returns (m_forms, C_coeffs) or None. m_forms is a list of 7 pairs
    (A_coeffs[4], B_coeffs[4]) giving linear forms; C_coeffs is 4x7."""
    # Standard candidate Strassen forms (a,b,c,d) x (e,f,g,h)
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    cand_m = [
        # (A-form coeffs [a,b,c,d], B-form coeffs [e,f,g,h])
        ((1, 0, 0, 1), (1, 0, 0, 1)),   # (a+d)(e+h)
        ((0, 0, 1, 1), (1, 0, 0, 0)),   # (c+d)e
        ((1, 0, 0, 0), (0, 1, 0, -1)),  # a(f-h)
        ((0, 0, 0, 1), (0, 0, 1, -1)),  # d(g-c)
        ((1, 1, 0, 0), (0, 0, 0, 1)),   # (a+b)h
        ((-1, 0, 1, 0), (1, 0, 1, 0)),  # (c-a)(e+g)
        ((0, 1, 0, -1), (0, 1, 0, 1)),  # (b-d)(f+h)
    ]
    rng = np.random.default_rng(7)
    N = 400
    samples = []
    for _ in range(N):
        av = rng.integers(-4, 5, size=4)
        bv = rng.integers(-4, 5, size=4)
        samples.append((av, bv))
    # Build M matrix (N x 7) and C matrix (N x 4)
    M = np.zeros((N, 7), dtype=float)
    C = np.zeros((N, 4), dtype=float)
    for t, (av, bv) in enumerate(samples):
        a, b, c, d = av
        e, f, g, h = bv
        for r in range(7):
            Aform = sum(av[i] * cand_m[r][0][i] for i in range(4))
            Bform = sum(bv[i] * cand_m[r][1][i] for i in range(4))
            M[t, r] = Aform * Bform
        A = np.array([[a, b], [c, d]], dtype=float)
        B = np.array([[e, f], [g, h]], dtype=float)
        Cmat = A @ B
        C[t, 0] = Cmat[0, 0]
        C[t, 1] = Cmat[0, 1]
        C[t, 2] = Cmat[1, 0]
        C[t, 3] = Cmat[1, 1]
    # Solve least squares per C entry
    coeffs, residuals, _, _ = np.linalg.lstsq(M, C, rcond=None)
    # Check exactness with integer samples
    for t, (av, bv) in enumerate(samples):
        a, b, c, d = av
        e, f, g, h = bv
        A = np.array([[a, b], [c, d]])
        B = np.array([[e, f], [g, h]])
        Cmat = A @ B
        for entry in range(4):
            pred = sum(M[t, r] * coeffs[r, entry] for r in range(7))
            if abs(pred - Cmat.flat[entry]) > 1e-6:
                return None
    return cand_m, coeffs


def _build_3x3x3_strassen():
    """Build a low-rank 3x3x3 decomposition using 2x2 Strassen core + schoolbook cross terms.
    Returns (U3, V3, W3) lists for the 3x3x3 sub-problem, or None if Strassen unavailable."""
    res = _try_2x2_strassen()
    if res is None:
        return None
    cand_m, coeffs = res
    # 3x3: A = [[a,b,c],[d,e,f],[g,h,i]], B = [[x,y,z],[u,v,w],[p,q,r]]
    # A-indices: a=0,b=1,c=2,d=3,e=4,f=5,g=6,h=7,i=8  (A[i,j] -> flat i*3+j)
    # B-indices: x=0,y=1,z=2,u=3,v=4,w=5,p=6,q=7,r=8  (B[j,k] -> flat j*3+k)
    # C-indices: C[i,k] flat i*3+k, 9 entries.
    # Strassen core uses A[0,0]=a, A[0,1]=b, A[1,0]=d, A[1,1]=e and B[0,0]=x, B[0,1]=y, B[1,0]=u, B[1,1]=v.
    # Map 2x2 (a,b,c,d) -> 3x3 A-indices (0,1,3,4); (e,f,g,h) -> B-indices (0,1,3,4).
    # The 7 Strassen terms produce C[0,0],C[0,1],C[1,0],C[1,1] (C flat 0,1,3,4).
    # Cross terms (schoolbook) for the rest:
    # C[0,2]=a*z+b*w+c*r  (A-idx 0,1,2 x B-idx 2,5,8)
    # C[1,2]=d*z+e*w+f*r  (A-idx 3,4,5 x B-idx 2,5,8)
    # C[2,0]=g*x+h*u+i*p  (A-idx 6,7,8 x B-idx 0,3,6)
    # C[2,1]=g*y+h*v+i*q  (A-idx 6,7,8 x B-idx 1,4,7)
    # C[2,2]=g*z+h*w+i*r  (A-idx 6,7,8 x B-idx 2,5,8)
    # Build U3 (9 x rank), V3 (9 x rank), W3 (9 x rank) for 3x3x3.
    # We'll accumulate contributions: each term c contributes to U3[A_idx], V3[B_idx], and to C-entries via W3.
    # For a bilinear term = (A-form)(B-form) that contributes coeff to C[i,k]:
    #   U3[a_idx, c] += A-form coeff for that a_idx
    #   V3[b_idx, c] += B-form coeff for that b_idx
    #   W3[i*3+k, c] += coeff
    # But a single term may contribute to multiple C entries (Strassen terms do).
    # Represent each term c by its A-form (9 coeffs) and B-form (9 coeffs), and its C-contribution (9 coeffs).
    terms = []  # list of (A_form[9], B_form[9], C_form[9])
    # Strassen terms: map 2x2 forms to 3x3.
    a_map = {0: 0, 1: 1, 2: 3, 3: 4}  # 2x2 idx -> 3x3 A flat idx (a->0,b->1,c->3,d->4)
    b_map = {0: 0, 1: 1, 2: 3, 3: 4}  # 2x2 idx -> 3x3 B flat idx (e->0,f->1,g->3,h->4)
    # C entries from Strassen: C[0,0]=flat0, C[0,1]=flat1, C[1,0]=flat3, C[1,1]=flat4
    c_map = [0, 1, 3, 4]  # 2x2 C entry idx -> 3x3 C flat idx
    for r in range(7):
        Aform2 = cand_m[r][0]  # 4 coeffs for (a,b,c,d)
        Bform2 = cand_m[r][1]  # 4 coeffs for (e,f,g,h)
        Aform9 = [0] * 9
        Bform9 = [0] * 9
        for t2 in range(4):
            Aform9[a_map[t2]] += Aform2[t2]
            Bform9[b_map[t2]] += Bform2[t2]
        Cform9 = [0] * 9
        for entry2 in range(4):
            Cform9[c_map[entry2]] += coeffs[r, entry2]
        terms.append((Aform9, Bform9, Cform9))
    # Schoolbook cross terms (each a single product A[a]*B[b] contributing 1 to one C entry)
    cross = [
        (0, 2, 2),   # a*z -> C[0,2] flat 2
        (1, 5, 2),   # b*w -> C[0,2] flat 2
        (2, 8, 2),   # c*r -> C[0,2] flat 2
        (3, 2, 5),   # d*z -> C[1,2] flat 5
        (4, 5, 5),   # e*w -> C[1,2] flat 5
        (5, 8, 5),   # f*r -> C[1,2] flat 5
        (6, 0, 6),   # g*x -> C[2,0] flat 6
        (7, 3, 6),   # h*u -> C[2,0] flat 6
        (8, 6, 6),   # i*p -> C[2,0] flat 6
        (6, 1, 7),   # g*y -> C[2,1] flat 7
        (7, 4, 7),   # h*v -> C[2,1] flat 7
        (8, 7, 7),   # i*q -> C[2,1] flat 7
        (6, 2, 8),   # g*z -> C[2,2] flat 8
        (7, 5, 8),   # h*w -> C[2,2] flat 8
        (8, 8, 8),   # i*r -> C[2,2] flat 8
    ]
    for (ai, bi, ci) in cross:
        Aform9 = [0] * 9
        Bform9 = [0] * 9
        Cform9 = [0] * 9
        Aform9[ai] = 1
        Bform9[bi] = 1
        Cform9[ci] = 1
        terms.append((Aform9, Bform9, Cform9))
    rank3 = len(terms)
    U3 = [[0] * rank3 for _ in range(9)]
    V3 = [[0] * rank3 for _ in range(9)]
    W3 = [[0] * rank3 for _ in range(9)]
    for c, (Aform9, Bform9, Cform9) in enumerate(terms):
        for idx in range(9):
            U3[idx][c] = Aform9[idx]
            V3[idx][c] = Bform9[idx]
            W3[idx][c] = Cform9[idx]
    return U3, V3, W3, rank3


def _try_reduce_3x3x4(U3, V3, W3, rank3, n, m, p):
    """Try to find a lower-rank decomposition of the full 3x3x7 tensor by
    sharing linear forms between the 3x3x3 block and the 3x3x4 block.
    
    Returns (U, V, W, rank) or None if no improvement found."""
    if time.time() - _start_time > _TIME_LIMIT - 10:
        return None
    
    # Build the initial 3x3x7 decomposition
    rank_full = rank3 + 3 * 3 * 4  # 22 + 36 = 58
    U = [[0] * rank_full for _ in range(n * m)]
    V = [[0] * rank_full for _ in range(m * p)]
    W = [[0] * rank_full for _ in range(p * n)]
    
    # Place 3x3x3 block in columns 0..rank3-1
    for c in range(rank3):
        for ij in range(9):
            i, j = divmod(ij, 3)
            U[i * 3 + j][c] = U3[ij][c]
        for jk in range(9):
            j, k = divmod(jk, 3)
            if k < 3:
                V[j * 7 + k][c] = V3[jk][c]
        for ki in range(9):
            k, i = divmod(ki, 3)
            if k < 3:
                W[k * 3 + i][c] = W3[ki][c]
    
    # Schoolbook for k=3..6
    col = rank3
    for i in range(3):
        for j in range(3):
            for k in range(3, 7):
                U[i * 3 + j][col] = 1
                V[j * 7 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    
    # Now try to reduce rank by finding linear dependencies
    # We look for pairs of terms (one from 3x3x3 block, one from 3x3x4 block)
    # that can be merged if their A-forms and B-forms are proportional or
    # if their combined C-form can be expressed with fewer terms.
    
    # A simpler approach: try to find if any two terms have the same A-form
    # and B-form up to scaling, and merge them.
    
    # For each pair of terms (c1, c2) where c1 < rank3 and c2 >= rank3,
    # check if U[:,c1] is proportional to U[:,c2] and V[:,c1] is proportional
    # to V[:,c2]. If so, we can merge them.
    
    best_rank = rank_full
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    # Try merging terms
    merged = [True] * rank_full
    for c1 in range(rank3):
        if not merged[c1]:
            continue
        for c2 in range(rank3, rank_full):
            if not merged[c2]:
                continue
            # Check if U[:,c1] and U[:,c2] are proportional
            # and V[:,c1] and V[:,c2] are proportional
            u1 = [U[i][c1] for i in range(9)]
            u2 = [U[i][c2] for i in range(9)]
            v1 = [V[i][c1] for i in range(21)]
            v2 = [V[i][c2] for i in range(21)]
            
            # Check proportionality: u1 = alpha * u2 for some alpha
            # Find alpha from first nonzero pair
            alpha = None
            for i in range(9):
                if u2[i] != 0:
                    if u1[i] % u2[i] == 0:
                        alpha = u1[i] // u2[i]
                        break
                    else:
                        alpha = None
                        break
            if alpha is None:
                continue
            
            # Verify proportionality
            valid = True
            for i in range(9):
                if u1[i] != alpha * u2[i]:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check V proportionality
            beta = None
            for i in range(21):
                if v2[i] != 0:
                    if v1[i] % v2[i] == 0:
                        beta = v1[i] // v2[i]
                        break
                    else:
                        beta = None
                        break
            if beta is None:
                continue
            
            valid = True
            for i in range(21):
                if v1[i] != beta * v2[i]:
                    valid = False
                    break
            if not valid:
                continue
            
            # They are proportional. Check if we can merge them.
            # The combined term would have A-form = u1 + gamma * u2 (but u1 = alpha*u2,
            # so this is (alpha + gamma)*u2), B-form = v1 + delta * v2 (v1 = beta*v2,
            # so (beta + delta)*v2), and C-form = w1 + gamma*delta * w2 (scaled appropriately).
            # This is complex. Let's just try removing c2 and adjusting c1.
            
            # Actually, if u1 = alpha*u2 and v1 = beta*v2, then the bilinear form
            # for term c1 is alpha*beta * (u2·A) * (v2·B), and for term c2 it's
            # (u2·A) * (v2·B). So term c1 = alpha*beta * term c2 (in terms of the
            # bilinear form). We can replace c1 with a scaled version of c2 and
            # remove c2.
            
            # But we need to adjust the W coefficients. Let's just try it.
            # New U[:,c1] = u1 (unchanged), V[:,c1] = v1 (unchanged)
            # We need to find new W[:,c1] such that the total contribution is preserved.
            # Original: W1 * (u1·A) * (v1·B) + W2 * (u2·A) * (v2·B)
            # = W1 * alpha*beta * (u2·A) * (v2·B) + W2 * (u2·A) * (v2·B)
            # = (W1*alpha*beta + W2) * (u2·A) * (v2·B)
            # So we can set W_new[:,c1] = W1*alpha*beta + W2 and remove c2.
            
            w1 = [W[i][c1] for i in range(21)]
            w2 = [W[i][c2] for i in range(21)]
            w_new = [w1[i] * alpha * beta + w2[i] for i in range(21)]
            
            # Check if w_new has small integer values
            valid = True
            for i in range(21):
                if not isinstance(w_new[i], (int, float)):
                    valid = False
                    break
                if abs(w_new[i] - round(w_new[i])) > 1e-9:
                    valid = False
                    break
            if not valid:
                continue
            
            # Create new decomposition without c2
            new_U = [row[:] for row in U]
            new_V = [row[:] for row in V]
            new_W = [row[:] for row in W]
            
            # Update W for c1
            for i in range(21):
                new_W[i][c1] = w_new[i]
            
            # Remove c2 by shifting columns
            for i in range(9):
                for c in range(c2, rank_full - 1):
                    new_U[i][c] = new_U[i][c + 1]
            for i in range(21):
                for c in range(c2, rank_full - 1):
                    new_V[i][c] = new_V[i][c + 1]
            for i in range(21):
                for c in range(c2, rank_full - 1):
                    new_W[i][c] = new_W[i][c + 1]
            
            # Set last column to 0
            for i in range(9):
                new_U[i][rank_full - 1] = 0
            for i in range(21):
                new_V[i][rank_full - 1] = 0
            new_W[i][rank_full - 1] = 0
            
            # Verify
            if _verify(new_U, new_V, new_W, n, m, p, rng_seed=999, n_trials=3):
                best_rank = rank_full - 1
                best_U = new_U
                best_V = new_V
                best_W = new_W
                merged[c2] = False
                break
    
    if best_rank < rank_full:
        # Trim trailing zero columns
        while best_rank > 0 and all(best_U[i][best_rank - 1] == 0 for i in range(9)):
            best_rank -= 1
        U_out = [[best_U[i][c] for c in range(best_rank)] for i in range(9)]
        V_out = [[best_V[i][c] for c in range(best_rank)] for i in range(21)]
        W_out = [[best_W[i][c] for c in range(best_rank)] for i in range(21)]
        return U_out, V_out, W_out, best_rank
    
    return None


def construct():
    n, m, p = DIMENSIONS
    # Guaranteed fallback
    fallback = _schoolbook(n, m, p)
    if _verify(fallback["U"], fallback["V"], fallback["W"], n, m, p):
        best = fallback
        best_rank = n * m * p
    else:
        best = None
        best_rank = None

    # Attempt rank-58: 3x3x3 (Strassen) + 3x3x4 (schoolbook)
    try:
        res3 = _build_3x3x3_strassen()
        if res3 is not None:
            U3, V3, W3, rank3 = res3
            # Build full 3x3x7: columns k=0,1,2 use the 3x3x3 block; k=3..6 schoolbook.
            rank_full = rank3 + 3 * 3 * 4  # 22 + 36 = 58
            U = [[0] * rank_full for _ in range(n * m)]
            V = [[0] * rank_full for _ in range(m * p)]
            W = [[0] * rank_full for _ in range(p * n)]
            # Place 3x3x3 block in columns 0..rank3-1, affecting k=0,1,2.
            for c in range(rank3):
                for ij in range(9):
                    i, j = divmod(ij, 3)
                    U[i * 3 + j][c] = U3[ij][c]
                for jk in range(9):
                    j, k = divmod(jk, 3)
                    if k < 3:
                        V[j * 7 + k][c] = V3[jk][c]
                for ki in range(9):
                    k, i = divmod(ki, 3)
                    if k < 3:
                        W[k * 3 + i][c] = W3[ki][c]
            # Schoolbook for k=3..6
            col = rank3
            for i in range(3):
                for j in range(3):
                    for k in range(3, 7):
                        U[i * 3 + j][col] = 1
                        V[j * 7 + k][col] = 1
                        W[k * 3 + i][col] = 1
                        col += 1
            cand = {"U": U, "V": V, "W": W}
            if _verify(cand["U"], cand["V"], cand["W"], n, m, p):
                if best is None or rank_full < best_rank:
                    best = cand
                    best_rank = rank_full
            
            # Try to reduce rank further
            if time.time() - _start_time < _TIME_LIMIT - 15:
                reduced = _try_reduce_3x3x4(U3, V3, W3, rank3, n, m, p)
                if reduced is not None:
                    U_r, V_r, W_r, rank_r = reduced
                    cand_r = {"U": U_r, "V": V_r, "W": W_r}
                    if _verify(cand_r["U"], cand_r["V"], cand_r["W"], n, m, p):
                        if best is None or rank_r < best_rank:
                            best = cand_r
                            best_rank = rank_r
    except Exception:
        pass

    if best is None:
        return fallback
    return best
