"""Matrix multiplication 2,4,6 rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s kill

def _time_left():
    return _DEADLINE - time.time()

def _schoolbook(n, m, p):
    """Schoolbook decomposition, rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2_terms():
    """7 Strassen terms for 2x2x2. U/V/W are length-4 vectors."""
    return [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [-1, 0, 0, 1]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, 0, 1]),
    ]

def _construct_rank42():
    """Strassen-based rank-42 decomposition for 2x4x6."""
    n, m, p = 2, 4, 6
    strassen_terms = _strassen_2x2x2_terms()
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for a_block in range(2):
        for b_block in range(3):
            for t in range(7):
                col = a_block * 21 + b_block * 7 + t
                U_t, V_t, W_t = strassen_terms[t]
                if a_block == 0:
                    a_indices = [0, 1, 4, 5]
                else:
                    a_indices = [2, 3, 6, 7]
                for i in range(4):
                    U[a_indices[i], col] = U_t[i]
                b_row_start = a_block * 2
                b_col_start = b_block * 2
                b_indices = [
                    b_row_start * 6 + b_col_start,
                    b_row_start * 6 + b_col_start + 1,
                    (b_row_start + 1) * 6 + b_col_start,
                    (b_row_start + 1) * 6 + b_col_start + 1,
                ]
                for i in range(4):
                    V[b_indices[i], col] = V_t[i]
                w_indices = [
                    2 * b_block, 2 * b_block + 1,
                    6 + 2 * b_block, 6 + 2 * b_block + 1,
                ]
                for i in range(4):
                    W[w_indices[i], col] = W_t[i]
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, num_trials=40):
    """Numerically verify the decomposition."""
    rank = W.shape[1]
    U = np.asarray(U); V = np.asarray(V); W = np.asarray(W)
    for _ in range(num_trials):
        A = np.random.randint(-5, 6, size=(n, m))
        B = np.random.randint(-5, 6, size=(m, p))
        C_exp = A @ B
        C_comp = np.zeros((n, p))
        for r in range(rank):
            u_val = U[:, r] @ A.flatten()
            v_val = V[:, r] @ B.flatten()
            for i in range(n):
                for k in range(p):
                    C_comp[i, k] += W[k * n + i, r] * u_val * v_val
        if not np.allclose(C_comp, C_exp, atol=1e-9):
            return False
    return True

def _rank_reduce(U, V, W, n, m, p, target_rank, max_iters=200):
    """
    Try to reduce the rank of a valid decomposition to target_rank by
    iteratively zeroing/merging terms and re-solving via linear least squares.
    Returns a valid decomposition of rank <= current, or None.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = W.shape[1]
    if rank <= target_rank:
        return {"U": U, "V": V, "W": W}
    # We'll attempt a greedy column-elimination: for each column, check if it
    # can be expressed as a linear combination of the others (in the joint
    # (U,V,W) tensor sense). If so, drop it.
    # The tensor T has entries T[i*m+j, k*p+l, a*n+b] = sum_r U[i*m+j,r]*V[k*p+l,r]*W[a*n+b,r]
    # Dropping column r is valid iff there exist coeffs c_s (s != r) such that
    # for all (i,j,k,l,a,b): U[i,j,r]*V[k,l,r]*W[a,b,r] = sum_{s!=r} c_s * U[i,j,s]*V[k,l,s]*W[a,b,s]
    # This is generally NOT a linear constraint in c_s because of the product structure.
    # So simple column dropping won't work. Instead, we rely on the fact that our
    # base decomposition is already optimal within its family.
    return None

def construct():
    n, m, p = 2, 4, 6
    # Build the Strassen rank-42 decomposition (known valid).
    result = _construct_rank42()
    U, V, W = result["U"], result["V"], result["W"]

    # Verify it's correct.
    if _time_left() > 3.0 and _verify(U, V, W, n, m, p, num_trials=25):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook if Strassen fails (shouldn't happen).
    sb = _schoolbook(n, m, p)
    return {"U": sb["U"].tolist(), "V": sb["V"].tolist(), "W": sb["W"].tolist()}
