import numpy as np

DIMENSIONS = [2, 4, 8]

def _build_tensor(n, m, p):
    # T[a, b, c] = 1 iff a=(i,j), b=(j',k), c=(i',k') with i=i', j=j', k=k'
    # a = i*m + j ; b = j'*p + k ; c = k'*n + i
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify(T, U, V, W):
    # check sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c] exactly
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Compute M = U @ V.T  -> (n*m, m*p); then for each r? No.
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # = sum_c (sum_r U[a,r] V[b,r] W[c,r])
    # Let's compute: for each r, outer(U[:,r], V[:,r]) * W[:,r] ... 
    # Better: reshape. Compute S[a,c] for each b? 
    # Efficient: build via einsum
    # T_recon[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # = np.einsum('ar,br,cr->abc', U, V, W)
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.array_equal(recon, T)

def _to_exact(arr, tol=1e-7):
    # round near-integers/half-integers to exact rationals represented as float multiples
    # We'll keep as float but snap to nearest 0.5
    return np.round(arr * 2) / 2.0

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)

    # Strategy: build a candidate set of rank-1 tensors (random + structured),
    # then find a small subset whose sum equals T exactly via linear programming /
    # least squares with exact verification.
    #
    # We'll generate a pool of rank-1 terms from structured bilinear forms,
    # then solve for exact integer/half-integer combination.
    #
    # Pool: all "Strassen-like" and "direct" terms.
    # Direct terms: for each (i,j,k): e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}  (rank-1, value 1)
    # That's n*m*p = 64 terms, rank 64 trivially.
    #
    # To reduce rank, we need combinations. Use a greedy + ILP-style approach:
    # Generate many random rank-1 tensors, then find a subset that sums to T.
    #
    # Better approach: use the fact that the multiplication tensor has a known
    # low-rank decomposition. We'll use a structured pool and solve a linear system.
    #
    # Pool of rank-1 terms:
    # 1. All direct terms (64)
    # 2. Strassen-style combinations on 2x2x2 blocks
    # 3. Random bilinear combinations
    #
    # We'll collect a pool, then use scipy to find a non-negative / exact combination.

    pool = []
    # Direct terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
                u[i*m + j] = 1
                v[j*p + k] = 1
                w[k*n + i] = 1
                pool.append((u, v, w))

    # Strassen 2x2x2 blocks (7 terms per block)
    S = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)], [(1,1),(3,-1)]),
        ([(0,1)], [(1,1),(3,-1)], [(2,1),(3,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(1,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(2,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]
    for j_off in [0, 2]:
        for k_off in [0, 2, 4, 6]:
            for r in range(7):
                u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
                u_entries, v_entries, w_entries = S[r]
                for idx, val in u_entries:
                    li, lj = divmod(idx, 2)
                    u[(li)*m + (lj + j_off)] += val
                for idx, val in v_entries:
                    lj, lk = divmod(idx, 2)
                    v[(lj + j_off)*p + (lk + k_off)] += val
                for idx, val in w_entries:
                    lk, li = divmod(idx, 2)
                    w[(lk + k_off)*n + li] += val
                pool.append((u, v, w))

    # Random structured terms: random bilinear forms with small integer entries
    rng = np.random.RandomState(42)
    for _ in range(500):
        u = rng.randint(-1, 2, size=n*m)
        v = rng.randint(-1, 2, size=m*p)
        w = rng.randint(-1, 2, size=p*n)
        # Only keep if the term is "useful" (non-zero)
        if np.any(u) and np.any(v) and np.any(w):
            pool.append((u.astype(float), v.astype(float), w.astype(float)))

    # Now find a subset of pool terms whose sum equals T exactly.
    # This is a subset-sum / linear combination problem over the tensor space.
    # We'll use a greedy approach: start with all direct terms (rank 64), 
    # then try to replace groups with Strassen terms to reduce rank.
    #
    # Simpler: use the 8 Strassen blocks (56 terms) which we know work, 
    # then try to find a smaller combination.
    #
    # For now, use the 56-term Strassen construction and verify.
    # Then attempt to compress via rank-revealing SVD on the coefficient matrix.

    # Build the 56-term solution
    U = np.zeros((n*m, 56)); V = np.zeros((m*p, 56)); W = np.zeros((p*n, 56))
    for blk, (j_off, k_off) in enumerate([(0,0),(0,2),(0,4),(0,6),(2,0),(2,2),(2,4),(2,6)]):
        for r in range(7):
            cr = blk*7 + r
            u_entries, v_entries, w_entries = S[r]
            for idx, val in u_entries:
                li, lj = divmod(idx, 2)
                U[li*m + (lj + j_off), cr] += val
            for idx, val in v_entries:
                lj, lk = divmod(idx, 2)
                V[(lj + j_off)*p + (lk + k_off), cr] += val
            for idx, val in w_entries:
                lk, li = divmod(idx, 2)
                W[(lk + k_off)*n + li, cr] += val

    # Verify
    if not _verify(T, U, V, W):
        # Fallback to direct
        U = np.zeros((n*m, n*m*p)); V = np.zeros((m*p, n*m*p)); W = np.zeros((p*n, n*m*p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U[i*m+j, r] = 1
                    V[j*p+k, r] = 1
                    W[k*n+i, r] = 1
        assert _verify(T, U, V, W)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to compress: find a lower-rank equivalent by solving for a basis
    # The 56 rank-1 tensors span a subspace. We want the minimal number.
    # Use SVD on the "flattened" coefficient structure.
    # 
    # For each output index c, the contribution is sum_r W[c,r] * (U[:,r] ⊗ V[:,r])
    # This is complex. Instead, try to find linear dependencies among the 56 terms.
    #
    # Build matrix M where each column is the vectorized rank-1 tensor U_r ⊗ V_r ⊗ W_r
    # That's too large (8*32*16 = 4096 per term, 56 terms -> 56x4096, feasible)
    vecs = np.array([np.einsum('a,b,c->abc', u, v, w).ravel() for u, v, w in 
                      [(U[:,r], V[:,r], W[:,r]) for r in range(56)]]).T  # 4096 x 56
    # Find rank of this matrix
    try:
        _, s, _ = np.linalg.svd(vecs, full_matrices=False)
        tol = 1e-10
        rank_found = np.sum(s > tol)
    except:
        rank_found = 56

    # If rank < 56, we can compress
    if rank_found < 56:
        # Find a basis
        U_svd, s, Vt = np.linalg.svd(vecs, full_matrices=False)
        # Keep top rank_found components
        # This gives a new decomposition but with real entries; snap to half-integers
        # For safety, keep 56 if compression introduces non-exact values
        pass

    # Return the verified 56-rank solution (snap to exact half-integers)
    U = _to_exact(U)
    V = _to_exact(V)
    W = _to_exact(W)
    
    if not _verify(T, U, V, W):
        # Re-snap carefully
        U = np.round(U * 2) / 2.0
        V = np.round(V * 2) / 2.0
        W = np.round(W * 2) / 2.0
        assert _verify(T, U, V, W), "Verification failed after snapping"

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
