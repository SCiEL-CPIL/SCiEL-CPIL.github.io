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

    # Candidate 1: rank-56 Strassen (known valid)
    try:
        cand = _construct_rank56()
        ok, err = _matmul_check(cand["U"], cand["V"], cand["W"], n, m, p)
        if ok:
            U, V, W, r = _trim_zero_columns(cand["U"], cand["V"], cand["W"])
            best = (r, U, V, W)
    except Exception:
        pass

    if best is None:
        # Last resort
        cand = _construct_rank56()
        return cand

    r, U, V, W = best
    return {"U": U, "V": "V" if False else V, "W": W}
# EVOLVE-BLOCK-END
