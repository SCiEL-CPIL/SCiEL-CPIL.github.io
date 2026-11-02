"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _round_to_half_integer(x):
    """Round x to the nearest half-integer."""
    return round(x * 2) / 2.0

def _strassen_block_terms(ai, aj, bj, bk, n, m, p):
    """
    Generate the 9 terms for a 2x2x3 block.
    A block: rows ai, ai+1; cols aj, aj+1 (2x2)
    B block: rows bj, bj+1; cols bk, bk+1, bk+2 (2x3)
    C block: rows ai, ai+1; cols bk, bk+1, bk+2 (2x3)
    
    Returns list of (u, v, w) where each is a dict {index: value}.
    """
    def u_idx(gi, gj):
        return gi * m + gj
    def v_idx(gj, gk):
        return gj * p + gk
    def w_idx(gk, gi):
        return gk * n + gi
    
    terms = []
    
    # --- Strassen on B cols bk, bk+1 (2x2x2 part) ---
    # M1 = (a+d)(e+h) -> C[ai,bk] += M1, C[ai+1,bk+1] += M1
    u = {u_idx(ai, aj): 1, u_idx(ai+1, aj+1): 1}
    v = {v_idx(bj, bk): 1, v_idx(bj+1, bk+1): 1}
    w = {w_idx(bk, ai): 1, w_idx(bk+1, ai+1): 1}
    terms.append((u, v, w))
    
    # M2 = (c+d)e -> C[ai+1,bk] += M2, C[ai+1,bk+1] -= M2
    u = {u_idx(ai+1, aj): 1, u_idx(ai+1, aj+1): 1}
    v = {v_idx(bj, bk): 1}
    w = {w_idx(bk, ai+1): 1, w_idx(bk+1, ai+1): -1}
    terms.append((u, v, w))
    
    # M3 = a(f-h) -> C[ai,bk+1] += M3, C[ai+1,bk+1] += M3
    u = {u_idx(ai, aj): 1}
    v = {v_idx(bj, bk+1): 1, v_idx(bj+1, bk+1): -1}
    w = {w_idx(bk+1, ai): 1, w_idx(bk+1, ai+1): 1}
    terms.append((u, v, w))
    
    # M4 = d(g-e) -> C[ai,bk] += M4, C[ai+1,bk] += M4
    u = {u_idx(ai+1, aj+1): 1}
    v = {v_idx(bj+1, bk): 1, v_idx(bj, bk): -1}
    w = {w_idx(bk, ai): 1, w_idx(bk, ai+1): 1}
    terms.append((u, v, w))
    
    # M5 = (a+b)h -> C[ai,bk] -= M5, C[ai,bk+1] += M5
    u = {u_idx(ai, aj): 1, u_idx(ai, aj+1): 1}
    v = {v_idx(bj+1, bk+1): 1}
    w = {w_idx(bk, ai): -1, w_idx(bk+1, ai): 1}
    terms.append((u, v, w))
    
    # M6 = (c-a)(e+f) -> C[ai+1,bk+1] += M6
    u = {u_idx(ai+1, aj): 1, u_idx(ai, aj): -1}
    v = {v_idx(bj, bk): 1, v_idx(bj, bk+1): 1}
    w = {w_idx(bk+1, ai+1): 1}
    terms.append((u, v, w))
    
    # M7 = (b-d)(g+h) -> C[ai,bk] += M7
    u = {u_idx(ai, aj+1): 1, u_idx(ai+1, aj+1): -1}
    v = {v_idx(bj+1, bk): 1, v_idx(bj+1, bk+1): 1}
    w = {w_idx(bk, ai): 1}
    terms.append((u, v, w))
    
    # --- Naive for B col bk+2 ---
    # C[ai, bk+2] = a * g + c * j  (where g=B[bj,bk+2], j=B[bj+1,bk+2])
    # Term 1: a * g
    u = {u_idx(ai, aj): 1}
    v = {v_idx(bj, bk+2): 1}
    w = {w_idx(bk+2, ai): 1}
    terms.append((u, v, w))
    
    # Term 2: c * j
    u = {u_idx(ai+1, aj): 1}
    v = {v_idx(bj+1, bk+2): 1}
    w = {w_idx(bk+2, ai+1): 1}
    terms.append((u, v, w))
    
    return terms

def _build_strassen_decomposition(n, m, p):
    """
    Build the full Strassen-based decomposition for 4x4x5.
    4 blocks: (ai=0,2) x (bj=0,2), each with B cols in groups of 2.
    B has 5 columns: groups are (0,1,2) and (3,4) -- wait, p=5.
    Actually B is 4x5. Rows are 0,1,2,3. Cols are 0,1,2,3,4.
    A is 4x4. Rows 0,1,2,3. Cols 0,1,2,3.
    C is 5x4. Rows 0,1,2,3,4. Cols 0,1,2,3.
    
    Block structure:
    A rows: ai = 0, 2 (two groups of 2)
    A cols: aj = 0, 2 (two groups of 2)
    B rows: bj = 0, 2 (two groups of 2)
    B cols: p=5, so we need to handle 5 columns.
    
    For each (ai, aj, bj) block:
    - B cols: we can do Strassen on pairs of B cols, then naive for remaining.
    - B has 5 cols: cols 0,1,2,3,4.
    - Strassen on cols (0,1), naive on col 2; Strassen on cols (3,4), no naive.
    - Or: Strassen on (0,1), (2,3), naive on 4.
    - Or: Strassen on (0,1), naive on 2,3,4.
    
    To minimize rank, let's do:
    - For B cols 0,1: Strassen (7 terms)
    - For B cols 2,3: Strassen (7 terms)  
    - For B col 4: naive (2 terms)
    Total per (ai,aj,bj) block: 16 terms
    4 blocks (ai in {0,2}, bj in {0,2}) x 16 = 64 terms.
    
    Wait, that's a lot. Let me reconsider.
    
    Actually, the 4 blocks are:
    (ai=0, aj=0, bj=0): A[0:2,0:2] * B[0:2,0:5] -> C[0:5,0:2]
    (ai=0, aj=2, bj=0): A[0:2,2:4] * B[0:2,0:5] -> C[0:5,2:4]
    (ai=2, aj=0, bj=0): A[2:4,0:2] * B[0:2,0:5] -> C[0:5,0:2]
    (ai=2, aj=2, bj=0): A[2:4,2:4] * B[0:2,0:5] -> C[0:5,2:4]
    (ai=0, aj=0, bj=2): A[0:2,0:2] * B[2:4,0:5] -> C[0:5,0:2]
    (ai=0, aj=2, bj=2): A[0:2,2:4] * B[2:4,0:5] -> C[0:5,2:4]
    (ai=2, aj=0, bj=2): A[2:4,0:2] * B[2:4,0:5] -> C[0:5,0:2]
    (ai=2, aj=2, bj=2): A[2:4,2:4] * B[2:4,0:5] -> C[0:5,2:4]
    
    So 8 blocks total (2 ai x 2 aj x 2 bj).
    Each block: A is 2x2, B is 2x5, C is 5x2.
    
    For each 2x2x5 block:
    - Strassen on B cols (0,1): 7 terms
    - Strassen on B cols (2,3): 7 terms
    - Naive on B col 4: 2 terms
    Total: 16 terms per block.
    
    8 blocks x 16 = 128 terms. That's too many.
    
    Let me reconsider. Maybe I should use a different blocking.
    
    Alternative: Use Strassen on the full 4x4x4 part, then handle the 5th column separately.
    
    4x4x4: Use 2x2x2 Strassen blocks.
    A is 4x4, B is 4x4 (first 4 cols), C is 4x4.
    4 blocks of 2x2x2, each with 7 terms = 28 terms.
    
    Then for the 5th column of B (B[:,4]):
    A (4x4) * B[:,4] (4x1) -> C[:,4] (4x1).
    This is a 4x4x1 matrix-vector multiply.
    Naive: 16 terms.
    Or use Strassen-like: 4x4x1 can be done in fewer terms?
    Actually, 4x4x1 is just a matrix-vector product. The tensor is rank 16 in general.
    But we can use the structure: C[:,4] = A * B[:,4].
    Each C[i,4] = sum_j A[i,j] * B[j,4].
    This is 16 scalar products, so 16 rank-1 terms in the naive case.
    
    Total: 28 + 16 = 44 terms.
    
    But wait, the C matrix is 5x4. C has 5 rows (k=0..4) and 4 cols (i=0..3).
    The 5th row of C (k=4) comes from the 5th column of B? No.
    
    Let me re-read the problem.
    A is 4x4 (n=4, m=4): A[i,j] for i in 0..3, j in 0..3.
    B is 4x5 (m=4, p=5): B[j,k] for j in 0..3, k in 0..4.
    C is 5x4 (p=5, n=4): C[k,i] for k in 0..4, i in 0..3.
    C[k,i] = sum_j A[i,j] * B[j,k].
    
    So C has 5 rows (k=0..4) and 4 cols (i=0..3).
    The 5th row of C (k=4) is C[4,i] = sum_j A[i,j] * B[j,4].
    
    So the "5th column" of B is B[j,4] for j=0..3, which contributes to C[4,i].
    
    For the 4x4x4 part (k=0..3):
    C[k,i] = sum_j A[i,j] * B[j,k] for k=0..3.
    This is a 4x4x4 tensor. Use Strassen: 28 terms.
    
    For the 5th row of C (k=4):
    C[4,i] = sum_j A[i,j] * B[j,4].
    This is a 4x4x1 tensor. Naive: 16 terms.
    
    Total: 28 + 16 = 44 terms.
    
    Let me verify this is correct by constructing it.
    """
    n, m, p = 4, 4, 5
    rank = 44
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    r = 0
    # 4x4x4 Strassen part
    for ai in [0, 2]:
        for aj in [0, 2]:
            for bk in [0, 2]:
                # 2x2x2 Strassen block
                # A block: A[ai:ai+2, aj:aj+2]
                # B block: B[aj:aj+2, bk:bk+2]
                # C block: C[bk:bk+2, ai:ai+2]
                
                # M1 = (a+d)(e+h)
                U[ai*m+aj, r] = 1
                U[(ai+1)*m+(aj+1), r] = 1
                V[aj*p+bk, r] = 1
                V[(aj+1)*p+(bk+1), r] = 1
                W[bk*n+ai, r] = 1
                W[(bk+1)*n+(ai+1), r] = 1
                r += 1
                
                # M2 = (c+d)e
                U[(ai+1)*m+aj, r] = 1
                U[(ai+1)*m+(aj+1), r] = 1
                V[aj*p+bk, r] = 1
                W[bk*n+(ai+1), r] = 1
                W[(bk+1)*n+(ai+1), r] = -1
                r += 1
                
                # M3 = a(f-h)
                U[ai*m+aj, r] = 1
                V[aj*p+(bk+1), r] = 1
                V[(aj+1)*p+(bk+1), r] = -1
                W[(bk+1)*n+ai, r] = 1
                W[(bk+1)*n+(ai+1), r] = 1
                r += 1
                
                # M4 = d(g-e)
                U[(ai+1)*m+(aj+1), r] = 1
                V[(aj+1)*p+bk, r] = 1
                V[aj*p+bk, r] = -1
                W[bk*n+ai, r] = 1
                W[bk*n+(ai+1), r] = 1
                r += 1
                
                # M5 = (a+b)h
                U[ai*m+aj, r] = 1
                U[ai*m+(aj+1), r] = 1
                V[(aj+1)*p+(bk+1), r] = 1
                W[bk*n+ai, r] = -1
                W[(bk+1)*n+ai, r] = 1
                r += 1
                
                # M6 = (c-a)(e+f)
                U[(ai+1)*m+aj, r] = 1
                U[ai*m+aj, r] = -1
                V[aj*p+bk, r] = 1
                V[aj*p+(bk+1), r] = 1
                W[(bk+1)*n+(ai+1), r] = 1
                r += 1
                
                # M7 = (b-d)(g+h)
                U[ai*m+(aj+1), r] = 1
                U[(ai+1)*m+(aj+1), r] = -1
                V[(aj+1)*p+bk, r] = 1
                V[(aj+1)*p+(bk+1), r] = 1
                W[bk*n+ai, r] = 1
                r += 1
    
    # 4x4x1 part for C row k=4
    # C[4,i] = sum_j A[i,j] * B[j,4]
    for i in range(4):
        for j in range(4):
            U[i*m+j, r] = 1
            V[j*p+4, r] = 1
            W[4*n+i, r] = 1
            r += 1
    
    assert r == rank, f"Expected {rank} terms, got {r}"
    return U, V, W

def _als_pass_fast(U, V, W, n, m, p, rank, max_iters=100, tol=1e-10):
    """Fast ALS to refine U, V, W."""
    prev_err = float('inf')
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(max_iters):
        for i in range(n):
            for j in range(m):
                A_mat = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                b_vec = np.ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        for j in range(m):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                b_vec = np.ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        for i in range(n):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                b_vec = np.ones(m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    err += (s - 1.0) ** 2
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    return U, V, W

def _polish_high(U, V, W, n, m, p, rank, max_iters=200):
    """High-precision polish."""
    return _als_pass_fast(U, V, W, n, m, p, rank, max_iters=max_iters, tol=1e-14)

def _try_rank(rank, n, m, p, seed=42, max_iters=80, polish_iters=100):
    """Try to find a decomposition at given rank using ALS refinement."""
    np.random.seed(seed)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    half_ints = [-1.0, -0.5, 0.5, 1.0]
    density = 0.3
    for r in range(rank):
        for idx in range(n*m):
            if np.random.random() < density:
                U[idx, r] = np.random.choice(half_ints)
        for idx in range(m*p):
            if np.random.random() < density:
                V[idx, r] = np.random.choice(half_ints)
        for idx in range(p*n):
            if np.random.random() < density:
                W[idx, r] = np.random.choice(half_ints)
    
    U, V, W = _als_pass_fast(U, V, W, n, m, p, rank, max_iters=max_iters)
    U, V, W = _polish_high(U, V, W, n, m, p, rank, max_iters=polish_iters)
    
    U_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in U])
    V_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in V])
    W_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in W])
    
    if _verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
        return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Build the guaranteed Strassen decomposition (rank 44)
    U_str, V_str, W_str = _build_strassen_decomposition(n, m, p)
    best_result = {"U": U_str.tolist(), "V": V_str.tolist(), "W": W_str.tolist()}
    best_rank = 44
    
    # Try to find lower rank using ALS
    for rank in range(43, 38, -1):
        if time.time() > deadline - 30:
            break
        
        found = False
        for seed in range(10):
            if time.time() > deadline - 30:
                break
            
            result = _try_rank(rank, n, m, p, seed=seed, max_iters=50, polish_iters=60)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                found = True
                break
        
        if found:
            continue
    
    return best_result
# EVOLVE-BLOCK-END
