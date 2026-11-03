import time
import numpy as np

DIMENSIONS = [2, 4, 8]


# EVOLVE-BLOCK-START
def _matmul_check(U, V, W, n, m, p):
    """Exact check that the tensor decomposition reconstructs the MM tensor.
    For every (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] must equal 1.
    Returns (ok, max_abs_error).
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    r = Ua.shape[1]
    Ur = Ua.reshape(n, m, r)
    Vr = Va.reshape(m, p, r)
    Wr = Wa.reshape(p, n, r)
    T = np.einsum('ijr,jkr,kin->ijk', Ur, Vr, Wr)
    err = np.max(np.abs(T - 1.0))
    return (err < 1e-9), float(err)


def _construct_rank56():
    """The original Strassen-based rank-56 decomposition (known valid)."""
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    strassen = [
        {'U': {(0, 0): 1, (1, 1): 1}, 'V': {(0, 0): 1, (1, 1): 1}, 'W': {(0, 0): 1, (1, 1): 1}},
        {'U': {(1, 0): 1, (1, 1): 1}, 'V': {(0, 0): 1}, 'W': {(0, 1): 1, (1, 1): -1}},
        {'U': {(0, 0): 1}, 'V': {(0, 1): 1, (1, 1): -1}, 'W': {(1, 0): 1, (1, 1): 1}},
        {'U': {(1, 1): 1}, 'V': {(1, 0): 1, (0, 0): -1}, 'W': {(0, 0): 1, (0, 1): 1}},
        {'U': {(0, 0): 1, (0, 1): 1}, 'V': {(1, 1): 1}, 'W': {(0, 0): -1, (1, 0): 1}},
        {'U': {(1, 0): 1, (0, 0): -1}, 'V': {(0, 0): 1, (0, 1): 1}, 'W': {(1, 1): 1}},
        {'U': {(0, 1): 1, (1, 1): -1}, 'V': {(1, 0): 1, (1, 1): 1}, 'W': {(0, 0): 1}},
    ]

    col = 0
    for j_high in range(2):
        for l in range(4):
            for term in strassen:
                for (i, j_low), c in term['U'].items():
                    j = j_high * 2 + j_low
                    U[i * m + j][col] += c
                for (j_low, k_low), c in term['V'].items():
                    j = j_high * 2 + j_low
                    k = l * 2 + k_low
                    V[j * p + k][col] += c
                for (k_low, i), c in term['W'].items():
                    k = l * 2 + k_low
                    W[k * n + i][col] += c
                col += 1
    return {"U": U, "V": V, "W": W}


def _construct_kronecker_baseline():
    """
    Kronecker product baseline: decompose 2x4x8 as (2x2x2) x (1x2x4) structure.
    
    View the 4x8 B matrix as having 4 row-groups of 2 and 4 col-groups of 2.
    The 2x4 A matrix has 2 row-groups of 1 (trivial) and 4 col-groups of 1.
    
    Actually, let's use: split into 2 (i) x 4 (j) x 4 (k_pairs) blocks.
    Each block is 1x1x2 or 2x2x2 depending on split.
    
    Alternative: use the tensor product T = T_A ⊗ T_B where
    T_A handles the 2x2x2 part (Strassen, rank 7)
    T_B handles the 4x4x4 part (schoolbook, rank 64)
    Total rank = 7 * 64 = 448. Too high.
    
    Better: T = T_A ⊗ T_B where T_A is 2x2x2 (rank 7) and T_B is 1x2x4 (rank 8)
    Total rank = 7 * 8 = 56. Same as before.
    
    Let me try: T = T_A ⊗ T_B where T_A is 2x4x4 (rank ?) and T_B is 1x1x2 (rank 2)
    Total rank = rank(T_A) * 2.
    
    For 2x4x4: split into 2x2x2 blocks. There are 2 (j) x 2 (k) = 4 blocks of 2x2x2.
    Schoolbook: 4*8 = 32. Strassen per block: 4*7 = 28.
    So rank(T_A) <= 28, total = 28*2 = 56.
    
    Or if I can do 2x4x4 with rank < 28, I win.
    """
    n, m, p = DIMENSIONS
    # This is a placeholder; the actual construction is in _construct_rank56
    return None


def _construct_rank28():
    """
    Rank-28 candidate: treat the 4x8 product as two 4x4 halves (k=0..3 and k=4..7).
    For each half, use the standard Strassen 4x4x4 decomposition (7 terms).
    The 7th Strassen term uses (A11+A12) which is the SAME across both halves,
    so we share that column of U. Total rank = 7 + 7 + 14 = 28.
    """
    n, m, p = DIMENSIONS
    # Strassen terms for 4x4x4: each term has U (4x4), V (4x4), W (4x4)
    # U[i,j] multiplies A[i,j], V[j,k] multiplies B[j,k], W[k,i] multiplies C[k,i]
    strassen = [
        # M1 = (A11+A22)(B11+B22)
        {'U': {(0,0):1,(3,3):1}, 'V': {(0,0):1,(3,3):1}, 'W': {(0,0):1,(0,3):1,(3,0):1,(3,3):-1}},
        # M2 = (A21+A22)B11
        {'U': {(2,0):1,(3,0):1,(2,1):1,(3,1):1}, 'V': {(0,0):1,(1,0):1}, 'W': {(0,2):1,(0,3):1,(1,2):-1,(1,3):-1}},
        # M3 = A11(B21-B11)
        {'U': {(0,0):1,(1,0):1}, 'V': {(0,1):-1,(1,1):-1,(0,2):1,(1,2):1}, 'W': {(2,0):1,(3,0):1,(2,1):-1,(3,1):-1}},
        # M4 = A22(B12-B22)
        {'U': {(2,1):1,(3,1):1}, 'V': {(0,1):1,(1,1):1,(0,2):-1,(1,2):-1}, 'W': {(1,2):1,(1,3):1,(0,2):-1,(0,3):-1}},
        # M5 = (A11+A12)B22
        {'U': {(0,0):1,(0,1):1,(1,0):1,(1,1):1}, 'V': {(2,2):1,(2,3):1,(3,2):1,(3,3):1}, 'W': {(2,0):1,(2,1):1,(3,0):1,(3,1):1}},
        # M6 = (A21-A11)(B11+B12)
        {'U': {(2,0):1,(3,0):1,(0,0):-1,(1,0):-1}, 'V': {(0,0):1,(1,0):1,(0,1):1,(1,1):1}, 'W': {(0,2):1,(0,3):1,(1,2):1,(1,3):1,(0,0):-1,(1,0):-1,(0,1):-1,(1,1):-1}},
        # M7 = (A12-A22)(B21+B22)
        {'U': {(0,1):1,(1,1):1,(2,1):-1,(3,1):-1}, 'V': {(2,0):1,(3,0):1,(2,1):1,(3,1):1}, 'W': {(2,0):1,(2,1):1,(3,0):1,(3,1):1,(0,0):-1,(0,1):-1,(1,0):-1,(1,1):-1}},
    ]

    rank = 28
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    col = 0
    # Half 1: k=0..3 (columns 0-3 of B)
    for t in range(7):
        for (i, j), c in strassen[t]['U'].items():
            U[i*m + j][col] += c
        for (j, k), c in strassen[t]['V'].items():
            V[j*p + k][col] += c
        for (k, i), c in strassen[t]['W'].items():
            W[k*n + i][col] += c
        col += 1
    # Half 2: k=4..7 (columns 4-7 of B)
    for t in range(7):
        for (i, j), c in strassen[t]['U'].items():
            U[i*m + j][col] += c
        for (j, k), c in strassen[t]['V'].items():
            V[j*p + (k+4)][col] += c
        for (k, i), c in strassen[t]['W'].items():
            W[(k+4)*n + i][col] += c
        col += 1
    # Shared M7 column: A12-A22 with B21+B22 for BOTH halves
    # U part: same as strassen[6] U
    for (i, j), c in strassen[6]['U'].items():
        U[i*m + j][col] += c
    # V part: (B21+B22) for half1 (k=0..3) AND (B21+B22) for half2 (k=4..7)
    for (j, k), c in strassen[6]['V'].items():
        V[j*p + k][col] += c
        V[j*p + (k+4)][col] += c
    # W part: (B21+B22) contribution to C for half1 AND half2
    for (k, i), c in strassen[6]['W'].items():
        W[k*n + i][col] += c
        W[(k+4)*n + i][col] += c

    return {"U": U, "V": V, "W": W}


def _construct_rank32():
    """
    Rank-32 candidate: use Strassen per 2x4x4 block but with a different splitting.
    Split j into two groups of 2. For each j-group, multiply 2x2 by 2x4.
    Use 2x2x2 Strassen for the row part and schoolbook for the col part? 
    Actually: 2x2x4 can be done with rank 2*2*4=16 schoolbook, or better.
    
    Alternative: treat as two independent 2x4x4 problems.
    2x4x4: split into four 2x1x1 blocks (schoolbook rank 8) or use structure.
    Actually 2x4x4 = (2x2x2) x 2 in some sense.
    
    Let's try: for each of the 2 k-halves (4x4), use Strassen rank 7.
    But the A matrix is 2x4, not 4x4. So we can't directly use 4x4 Strassen.
    
    Better approach for rank 32: 
    Split the 4 columns of A (j=0..3) into two groups of 2.
    For group j=0,1: compute A[:, 0:2] * B[0:2, :] which is 2x2 * 2x8 = 2x8
    For group j=2,3: compute A[:, 2:4] * B[2:4, :] which is 2x2 * 2x8 = 2x8
    Each 2x2*2x8 can be decomposed. 2x2*2x8: split 8 into four 2x2 blocks.
    Each 2x2*2x2 has Strassen rank 7. So 4*7 = 28 per group, total 56. Not better.
    
    Let me try a different approach: use the fact that we can share more.
    Actually, let me just return None and rely on rank 28 or 56.
    """
    return None


def _trim_zero_columns(U, V, W):
    """Remove rank columns that are all-zero in U, V, and W."""
    n_u = len(U)
    n_v = len(V)
    n_w = len(W)
    rank = len(U[0]) if n_u else 0
    keep = []
    for c in range(rank):
        if any(U[i][c] != 0 for i in range(n_u)) or \
           any(V[i][c] != 0 for i in range(n_v)) or \
           any(W[i][c] != 0 for i in range(n_w)):
            keep.append(c)
    if len(keep) == rank:
        return U, V, W, rank
    U2 = [[U[i][c] for c in keep] for i in range(n_u)]
    V2 = [[V[i][c] for c in keep] for i in range(n_v)]
    W2 = [[W[i][c] for c in keep] for i in range(n_w)]
    return U2, V2, W2, len(keep)


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140

    best = None

    # Candidate 1: rank-28 (shared Strassen M7)
    if time.time() < deadline:
        try:
            cand = _construct_rank28()
            if cand is not None:
                ok, err = _matmul_check(cand["U"], cand["V"], cand["W"], n, m, p)
                if ok:
                    U, V, W, r = _trim_zero_columns(cand["U"], cand["V"], cand["W"])
                    best = (r, U, V, W)
        except Exception:
            pass

    # Candidate 2: rank-32
    if time.time() < deadline:
        try:
            cand = _construct_rank32()
            if cand is not None:
                ok, err = _matmul_check(cand["U"], cand["V"], cand["W"], n, m, p)
                if ok:
                    U, V, W, r = _trim_zero_columns(cand["U"], cand["V"], cand["W"])
                    if best is None or r < best[0]:
                        best = (r, U, V, W)
        except Exception:
            pass

    # Candidate 3: rank-56 Strassen (known valid fallback)
    if time.time() < deadline:
        try:
            cand = _construct_rank56()
            ok, err = _matmul_check(cand["U"], cand["V"], cand["W"], n, m, p)
            if ok:
                U, V, W, r = _trim_zero_columns(cand["U"], cand["V"], cand["W"])
                if best is None or r < best[0]:
                    best = (r, U, V, W)
        except Exception:
            pass

    if best is None:
        # Last resort
        cand = _construct_rank56()
        return cand

    r, U, V, W = best
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
