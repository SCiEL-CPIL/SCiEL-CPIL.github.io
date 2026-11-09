import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

_T0 = time.time()
_LIMIT = 150.0
_MARGIN = 5.0

def _deadline_exceeded():
    return (time.time() - _T0) > (_LIMIT - _MARGIN)


def _construct_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                idx += 1
    return U, V, W


def _strassen_2x2_terms():
    return [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]


def _strassen_4x4_terms():
    s2 = _strassen_2x2_terms()
    macro_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    terms = []
    for A_m, B_m, C_m in macro_terms:
        for u2, v2, w2 in s2:
            u_coeffs = {}
            v_coeffs = {}
            w_coeffs = {}
            for (li, lj), u_val in u2.items():
                for (si, sj), a_val in A_m.items():
                    gi = si * 2 + li
                    gj = sj * 2 + lj
                    u_coeffs[(gi, gj)] = u_coeffs.get((gi, gj), 0) + u_val * a_val
            for (lj, lk), v_val in v2.items():
                for (sj, sk), b_val in B_m.items():
                    gj = sj * 2 + lj
                    gk = sk * 2 + lk
                    v_coeffs[(gj, gk)] = v_coeffs.get((gj, gk), 0) + v_val * b_val
            for (li, lk), w_val in w2.items():
                for (si, sk), c_val in C_m.items():
                    gi = si * 2 + li
                    gk = sk * 2 + lk
                    w_coeffs[(gi, gk)] = w_coeffs.get((gi, gk), 0) + w_val * c_val
            u_coeffs = {k: v for k, v in u_coeffs.items() if v != 0}
            v_coeffs = {k: v for k, v in v_coeffs.items() if v != 0}
            w_coeffs = {k: v for k, v in w_coeffs.items() if v != 0}
            terms.append((u_coeffs, v_coeffs, w_coeffs))
    return terms


def _build_block_decomposition(n, m, p, block_sizes):
    """
    Build a block decomposition. block_sizes is a list of (size_i, size_j, size_k)
    triples describing contiguous blocks along each axis. For each block, if all
    three sizes are >= 2 we use the appropriate Strassen, else schoolbook.
    Returns (U, V, W) as dense numpy arrays.
    """
    # Precompute
    s2 = _strassen_2x2_terms()
    s4 = _strassen_4x4_terms()

    # Build index lists
    def _index_lists(sizes):
        lists = []
        off = 0
        for s in sizes:
            lists.append(list(range(off, off + s)))
            off += s
        return lists

    i_lists = _index_lists(block_sizes[0])
    j_lists = _index_lists(block_sizes[1])
    k_lists = _index_lists(block_sizes[2])

    columns = []

    def add_column(u_entries, v_entries, w_entries):
        u_d = {}
        for idx, val in u_entries:
            u_d[idx] = u_d.get(idx, 0) + val
        v_d = {}
        for idx, val in v_entries:
            v_d[idx] = v_d.get(idx, 0) + val
        w_d = {}
        for idx, val in w_entries:
            w_d[idx] = w_d.get(idx, 0) + val
        columns.append((u_d, v_d, w_d))

    def add_schoolbook_block(i_list, j_list, k_list):
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    add_column(
                        [(i * m + j, 1.0)],
                        [(j * p + k, 1.0)],
                        [(k * n + i, 1.0)]
                    )

    def add_strassen_block(i_list, j_list, k_list, terms):
        for u_coeffs, v_coeffs, w_coeffs in terms:
            u_entries = []
            v_entries = []
            w_entries = []
            for (li, lj), val in u_coeffs.items():
                gi = i_list[li]
                gj = j_list[lj]
                u_entries.append((gi * m + gj, val))
            for (lj, lk), val in v_coeffs.items():
                gj = j_list[lj]
                gk = k_list[lk]
                v_entries.append((gj * p + gk, val))
            for (li, lk), val in w_coeffs.items():
                gi = i_list[li]
                gk = k_list[lk]
                w_entries.append((gk * n + gi, val))
            add_column(u_entries, v_entries, w_entries)

    for bi in range(len(i_lists)):
        for bj in range(len(j_lists)):
            for bk in range(len(k_lists)):
                i_list = i_lists[bi]
                j_list = j_lists[bj]
                k_list = k_lists[bk]
                s = len(i_list)
                if s == 4 and len(j_list) == 4 and len(k_list) == 4:
                    add_strassen_block(i_list, j_list, k_list, s4)
                elif s == 2 and len(j_list) == 2 and len(k_list) == 2:
                    add_strassen_block(i_list, j_list, k_list, s2)
                else:
                    add_schoolbook_block(i_list, j_list, k_list)

    total_rank = len(columns)
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    for c, (u_d, v_d, w_d) in enumerate(columns):
        for idx, val in u_d.items():
            U[idx, c] = val
        for idx, val in v_d.items():
            V[idx, c] = val
        for idx, val in w_d.items():
            W[idx, c] = val
    return U, V, W


def _reduce_columns(U, V, W, tol=1e-9):
    """
    Given a valid decomposition (U, V, W), attempt to reduce the number of
    columns by identifying columns r where U[:,r], V[:,r], W[:,r] are linear
    combinations of the others. We use a simple greedy approach: for each
    column r, check if removing it still leaves a valid decomposition.
    This is O(rank^2 * verification_cost) which is expensive, so we use a
    smarter approach: find linear dependencies among the column vectors
    (U[:,r], V[:,r], W[:,r]) in the ambient space.
    """
    rank = U.shape[1]
    if rank < 2:
        return U, V, W

    # Stack into a single matrix for dependency detection
    # Each column r gives a vector in R^{3*25}
    stacked = np.vstack([U, V, W])  # shape (75, rank)

    # Find rank of the stacked matrix
    # Use SVD to find the numerical rank
    try:
        U_svd, s, Vt = np.linalg.svd(stacked, full_matrices=False)
    except np.linalg.LinAlgError:
        return U, V, W

    # Determine numerical rank
    max_s = s[0] if len(s) > 0 else 0
    if max_s == 0:
        return U, V, W
    num_rank = np.sum(s > max_s * tol)

    if num_rank >= rank:
        # No linear dependencies found
        return U, V, W

    # We found dependencies. Now we need to actually remove columns while
    # preserving the decomposition. This is trickier because we need to
    # find which columns to remove and how to adjust the others.
    #
    # For now, we just return the original since doing this correctly
    # requires solving a system. A simpler approach: try removing each
    # column and check validity, but this is expensive.
    #
    # Instead, let's just return the original. The reduction is a nice
    # optimization but the block decomposition already gives rank 110.
    return U, V, W


def _verify_decomposition_fast(U, V, W, n=5, m=5, p=5):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    rng = np.random.RandomState(12345)
    samples = 5000
    for _ in range(samples):
        a = rng.randint(0, 25)
        b = rng.randint(0, 25)
        c = rng.randint(0, 25)
        i = a // m
        j = a % m
        j2 = b // p
        k = b % p
        k2 = c // n
        i2 = c % n
        if j == j2 and k == k2 and i == i2:
            continue
        val = np.sum(U[a, :] * V[b, :] * W[c, :])
        if abs(val) > 1e-9:
            return False
    return True


def construct():
    start_time = time.time()
    deadline = 120.0
    n, m, p = DIMENSIONS

    # Fallback: schoolbook
    U_school, V_school, W_school = _construct_schoolbook(n, m, p)

    # Candidate 1: 4+1 partition (Strassen 4x4x4 + schoolbook)
    best = None
    try:
        if time.time() - start_time < deadline - 60:
            U1, V1, W1 = _build_block_decomposition(n, m, p, ([4,1], [4,1], [4,1]))
            rank1 = U1.shape[1]
            if time.time() - start_time < deadline - 30:
                if _verify_decomposition_fast(U1, V1, W1, n, m, p):
                    best = (rank1, U1, V1, W1)
    except Exception:
        pass

    # Candidate 2: 3+2 partition (schoolbook 3x3x3 + Strassen 2x2x2)
    try:
        if time.time() - start_time < deadline - 60:
            U2, V2, W2 = _build_block_decomposition(n, m, p, ([3,2], [3,2], [3,2]))
            rank2 = U2.shape[1]
            if time.time() - start_time < deadline - 30:
                if _verify_decomposition_fast(U2, V2, W2, n, m, p):
                    if best is None or rank2 < best[0]:
                        best = (rank2, U2, V2, W2)
    except Exception:
        pass

    if best is not None:
        return {
            "U": best[1].tolist(),
            "V": best[2].tolist(),
            "W": best[3].tolist()
        }

    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }

# EVOLVE-BLOCK-END
