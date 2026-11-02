import time
import numpy as np

# ---------- Fast verification helpers ----------

# 2x2x3 expected tensor: T[a,c,b,d,e,f] = 1 iff c==b and a==f and d==e
def _build_223():
    n, m, p = 2, 2, 3
    idx = np.arange(n * m * m * p * p * n)
    a = (idx // (m * m * p * p * n)) % n
    c = (idx // (m * p * p * n)) % m
    b = (idx // (p * p * n)) % m
    d = (idx // (p * n)) % p
    e = (idx // n) % p
    f = idx % n
    T = ((c == b) & (a == f) & (d == e)).astype(int)
    return T.reshape(n * m, m * p, p * n)

T223 = _build_223()
T222 = T223[:, :4, :4]  # 2x2x2 slice (d==e only for d,e in {0,1})


def _check_block(U, V, W, T):
    """Return True if sum_r U[i,r]*V[j,r]*W[k,r] == T[i,j,k] for all i,j,k."""
    res = np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)
    return bool(np.array_equal(res, T))


def _try_rank(nm, mp, pn, T, rank, val_options, iters, deadline):
    """Randomized search for a rank decomposition of T (shape nm x mp x pn)."""
    vals = np.array(val_options, dtype=float)
    nvals = len(vals)
    for it in range(iters):
        if time.time() > deadline:
            return None
        U = vals[np.random.randint(0, nvals, size=(nm, rank))]
        V = vals[np.random.randint(0, nvals, size=(mp, rank))]
        W = vals[np.random.randint(0, nvals, size=(pn, rank))]
        if _check_block(U, V, W, T):
            return (U.tolist(), V.tolist(), W.tolist())
    return None


def _try_rank_structured(nm, mp, pn, T, rank, deadline):
    """
    Structured search: build from Strassen 2x2x2 (rank 7) + extra terms for p=3.
    Try all ways to add rank-3 extension to Strassen rank-7 base.
    """
    # Strassen 2x2x2 components (indices: i in 0..3, j in 0..3, k in 0..3)
    # For 2x2x3: i in 0..3, j in 0..5, k in 0..5
    # The 2x2x2 part uses j=0,1,2,3 and k=0,1,2,3
    # The 2x2x1 part uses j=4,5 and k=4,5
    
    # Strassen matrices for 2x2x2
    A_s = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ], dtype=float)
    B_s = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ], dtype=float)
    C_s = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ], dtype=float)
    
    # Schoolbook 2x2x1: A_b (4x4), B_b (4x2), C_b (4x2)
    A_b = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], dtype=float)
    B_b = np.array([[1,0],[0,1],[1,0],[0,1]], dtype=float)
    C_b = np.array([[1,0],[1,0],[0,1],[0,1]], dtype=float)
    
    # Build base: Strassen 7 terms for 2x2x2 part
    # U[0:4, 0:7] = A_s, V[0:4, 0:7] = B_s, W[0:4, 0:7] = C_s
    # For 2x2x3: V has 6 rows (2x3), W has 6 rows (3x2)
    # V indices: [0,1,2,3,4,5] where 0-3 are 2x2 part, 4-5 are 2x1 part
    # W indices: [0,1,2,3,4,5] where 0-3 are 2x2 part, 4-5 are 2x1 part
    
    U_base = np.zeros((4, rank), dtype=float)
    V_base = np.zeros((6, rank), dtype=float)
    W_base = np.zeros((6, rank), dtype=float)
    
    U_base[:, :7] = A_s
    V_base[:4, :7] = B_s
    W_base[:4, :7] = C_s
    
    # Now we need to fill columns 7..rank-1 to handle the 2x2x1 part
    # The 2x2x1 part: multiply 2x2 by 2x1 (column vector)
    # T223[i, j, k] for j in {4,5}, k in {4,5}:
    #   j=4: b=0, d=2; j=5: b=1, d=2
    #   k=4: e=2, f=0; k=5: e=2, f=1
    #   T[i,4,4] = 1 iff c==0 and a==0 (i.e., i=0)
    #   T[i,4,5] = 1 iff c==0 and a==1 (i.e., i=2)
    #   T[i,5,4] = 1 iff c==1 and a==0 (i.e., i=1)
    #   T[i,5,5] = 1 iff c==1 and a==1 (i.e., i=3)
    
    # Also cross terms: j in {0,1,2,3}, k in {4,5} and j in {4,5}, k in {0,1,2,3}
    # T[i,j,k] for j<4, k>=4: d = j%3, e = k//2 = 2, f = k%2
    #   Need d==e, so j%3==2, i.e., j=2. And a==f, c==b.
    #   T[i,2,k] = 1 iff c==1 (j=2 means b=1) and a==f and d==e (d=2,e=2 ok)
    #   k=4: f=0, so a=0, i=2c+0=2 (c=1)
    #   k=5: f=1, so a=1, i=2c+1=3 (c=1)
    #   So T[2,2,4]=1, T[3,2,5]=1
    
    # T[i,j,k] for j>=4, k<4: d = j%3 = 2 (j=4 or 5), e = k//2, f = k%2
    #   Need d==e, so e=2, but k<4 means e in {0,1}. So no cross terms here.
    #   Wait: j=4: b=0, d=2. j=5: b=1, d=2.
    #   k=0: e=0,f=0; k=1: e=0,f=1; k=2: e=1,f=0; k=3: e=1,f=1
    #   d=2 != e for e in {0,1}. So T[i,j,k]=0 for j>=4, k<4. Good.
    
    # So the residual to fill (after Strassen handles 2x2x2 part) is:
    # R[i,j,k] = T223[i,j,k] - sum_{r=0}^{6} A_s[i,r]*B_s[j,r]*C_s[k,r] for j,k<4
    # For j,k >= 4: R = T223 values
    # For j<4, k>=4: R = T223 values (Strassen gives 0 for k>=4 since C_s only has 4 cols)
    
    # Actually, Strassen only covers j in 0..3, k in 0..3. For j or k >= 4, Strassen contributes 0.
    # So R[i,j,k] = T223[i,j,k] for any (j,k) where j>=4 or k>=4
    # And R[i,j,k] = T223[i,j,k] - (Strassen contribution) for j<4 and k<4
    
    # Let's compute the residual
    U_full = np.zeros((4, rank), dtype=float)
    V_full = np.zeros((6, rank), dtype=float)
    W_full = np.zeros((6, rank), dtype=float)
    U_full[:, :7] = A_s
    V_full[:4, :7] = B_s
    W_full[:4, :7] = C_s
    
    # Compute current contribution
    current = np.einsum('ir,jr,kr->ijk', U_full, V_full, W_full, optimize=True)
    residual = T223.astype(float) - current
    
    # Now we need to decompose residual with (rank-7) additional terms
    extra_rank = rank - 7
    if extra_rank < 0:
        return None
    
    # The residual is nonzero only for:
    # - j in {0,1,2,3}, k in {4,5}: R[2,2,4]=1, R[3,2,5]=1 (from above)
    # - j in {4,5}, k in {0,1,2,3}: all 0
    # - j in {4,5}, k in {4,5}: R[0,4,4]=1, R[2,4,5]=1, R[1,5,4]=1, R[3,5,5]=1
    
    # Let's verify by printing residual
    nz = np.argwhere(residual != 0)
    # Expected: (2,2,4), (3,2,5), (0,4,4), (2,4,5), (1,5,4), (3,5,5)
    
    # We need to represent these 6 nonzero entries with extra_rank terms
    # Each term r: U[i,r] * V[j,r] * W[k,r]
    
    # For the 4 "diagonal" entries (0,4,4), (2,4,5), (1,5,4), (3,5,5):
    # These form a 2x2x1 schoolbook pattern on the "third" column
    # Can be done with 4 terms (schoolbook) or potentially fewer
    
    # For the 2 "cross" entries (2,2,4), (3,2,5):
    # These need separate handling
    
    # Total: 4 + 2 = 6 terms needed naively, but maybe we can share
    
    # Let's try: use extra_rank terms to cover all 6 nonzero residual entries
    # Try random search for the extra terms
    val_options = [-2, -1, 0, 1, 2]
    vals = np.array(val_options, dtype=float)
    nvals = len(vals)
    
    iters = 0
    max_iters = 500000
    while iters < max_iters:
        if time.time() > deadline:
            break
        iters += 1
        U_extra = vals[np.random.randint(0, nvals, size=(4, extra_rank))]
        V_extra = vals[np.random.randint(0, nvals, size=(6, extra_rank))]
        W_extra = vals[np.random.randint(0, nvals, size=(6, extra_rank))]
        
        U_test = np.hstack([U_full[:, :7], U_extra])
        V_test = np.hstack([V_full[:, :7], V_extra])
        W_test = np.hstack([W_full[:, :7], W_extra])
        
        if _check_block(U_test, V_test, W_test, T223):
            return (U_test.tolist(), V_test.tolist(), W_test.tolist())
    
    return None


# ---------- 2x2x2 Strassen (rank 7) ----------
A_s = [
    [1, 0, 0, 1],
    [0, 0, 1, 1],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [1, 1, 0, 0],
    [-1, 0, 1, 0],
    [0, 1, 0, -1],
]
B_s = [
    [1, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 1, 0, -1],
    [-1, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
]
C_s = [
    [1, 0, 0, 1],
    [0, 0, 1, -1],
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [-1, 1, 0, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0],
]

# 2x2x1 schoolbook (rank 4)
A_b = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
B_b = [[1,0],[0,1],[1,0],[0,1]]
C_b = [[1,0],[1,0],[0,1],[0,1]]


def _assemble_53():
    n, m, p = 2, 2, 15
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col_offset = 0
    for block in range(7):
        k0 = block * 2
        k1 = k0 + 1
        for r in range(7):
            c = col_offset + r
            for i in range(4):
                U[i][c] = A_s[r][i]
            for j in range(2):
                V[j * p + k0][c] = B_s[r][j * 2 + 0]
                V[j * p + k1][c] = B_s[r][j * 2 + 1]
            W[k0 * n + 0][c] = C_s[r][0]
            W[k1 * n + 0][c] = C_s[r][1]
            W[k0 * n + 1][c] = C_s[r][2]
            W[k1 * n + 1][c] = C_s[r][3]
        col_offset += 7
    k_last = 14
    for r in range(4):
        c = col_offset + r
        for i in range(4):
            U[i][c] = A_b[r][i]
        V[0 * p + k_last][c] = B_b[r][0]
        V[1 * p + k_last][c] = B_b[r][1]
        W[k_last * n + 0][c] = C_b[r][0]
        W[k_last * n + 1][c] = C_b[r][1]
    return {"U": U, "V": V, "W": W}


def _assemble_50(block):
    """Use 5 copies of a rank-10 2x2x3 block."""
    n, m, p = 2, 2, 15
    U223, V223, W223 = block
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col_offset = 0
    for blk in range(5):
        k0 = blk * 3
        for r in range(10):
            c = col_offset + r
            for i in range(4):
                U[i][c] = U223[i][r]
            for b in range(2):
                for d in range(3):
                    V[b * p + k0 + d][c] = V223[b * 3 + d][r]
            for e in range(3):
                for f in range(2):
                    W[(k0 + e) * n + f][c] = W223[e * 2 + f][r]
        col_offset += 10
    return {"U": U, "V": V, "W": W}


def _verify_full(n, m, p, U, V, W):
    """Full verification of the assembled decomposition against the true tensor."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    res = np.einsum('ir,jr,kr->ijk', Ua, Va, Wa, optimize=True)
    # Build expected tensor T[i,j,k]
    i = np.arange(n * m)
    a = i // m
    c = i % m
    j = np.arange(m * p)
    c2 = j // p
    kcol = j % p
    k = np.arange(p * n)
    krow = k // n
    a2 = k % n
    # T[i,j,k] = 1 iff c==c2 and a==a2 and kcol==krow
    T = np.zeros((n * m, m * p, p * n), dtype=int)
    for ii in range(n * m):
        for jj in range(m * p):
            for kk in range(p * n):
                if (c[ii] == c2[jj] and a[ii] == a2[kk] and kcol[jj] == krow[kk]):
                    T[ii, jj, kk] = 1
    return bool(np.array_equal(res, T))


def construct():
    start_time = time.time()
    deadline = start_time + 120
    n, m, p = 2, 2, 15

    # Guaranteed fallback
    fallback = _assemble_53()

    # Try structured search for rank-10 2x2x3 block
    search_deadline = start_time + 100  # leave 20s safety margin
    np.random.seed(42)
    
    # First try structured approach
    block = _try_rank_structured(4, 6, 6, T223, rank=10, deadline=search_deadline)
    
    if block is None:
        # Fallback to pure random search
        block = _try_rank(4, 6, 6, T223, rank=10,
                        val_options=[-1, 0, 1], iters=200000,
                        deadline=search_deadline)

    if block is not None:
        cand = _assemble_50(block)
        # Verify the full assembled result before trusting it.
        if _verify_full(n, m, p, cand["U"], cand["V"], cand["W"]):
            return cand
        # If full verify fails (shouldn't if block is exact), keep fallback.

    return fallback
