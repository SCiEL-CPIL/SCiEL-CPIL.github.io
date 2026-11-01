import time
import numpy as np
from itertools import permutations

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2_terms(i0, j0, k0, n, m, p):
    """Generate 7 Strassen rank-1 terms for a 2x2x2 block at offset (i0,j0,k0)."""
    terms = []
    # A-space index: i*m + j
    a00 = i0 * m + j0
    a01 = i0 * m + j0 + 1
    a10 = (i0 + 1) * m + j0
    a11 = (i0 + 1) * m + j0 + 1
    # B-space index: j*p + k
    b00 = j0 * p + k0
    b01 = j0 * p + k0 + 1
    b10 = (j0 + 1) * p + k0
    b11 = (j0 + 1) * p + k0 + 1
    # C-space index: k*n + i
    c00 = k0 * n + i0
    c01 = k0 * n + i0 + 1
    c10 = (k0 + 1) * n + i0
    c11 = (k0 + 1) * n + i0 + 1

    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    terms.append(([(a00,1),(a11,1)], [(b00,1),(b11,1)], [(c00,1),(c11,1)]))
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    terms.append(([(a10,1),(a11,1)], [(b00,1)], [(c10,1),(c11,-1)]))
    # M3 = A00(B10-B00) -> C01 += M3, C11 += M3
    terms.append(([(a00,1)], [(b10,1),(b00,-1)], [(c01,1),(c11,1)]))
    # M4 = A11(B11-B01) -> C00 += M4, C10 += M4
    terms.append(([(a11,1)], [(b11,1),(b01,-1)], [(c00,1),(c10,1)]))
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    terms.append(([(a00,1),(a01,1)], [(b11,1)], [(c00,-1),(c01,1)]))
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    terms.append(([(a10,1),(a00,-1)], [(b00,1),(b01,1)], [(c11,1)]))
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    terms.append(([(a01,1),(a11,-1)], [(b10,1),(b11,1)], [(c00,1)]))
    return terms


def _schoolbook_term(i, j, k, n, m, p):
    """Single schoolbook rank-1 term."""
    a = i * m + j
    b = j * p + k
    c = k * n + i
    return ([(a, 1)], [(b, 1)], [(c, 1)])


def _build_from_partition(m_parts, p_parts, n, m, p):
    """
    Build decomposition using block-Strassen with given partitions of m and p.
    n=2 is always treated as a 2-row block.
    Returns (terms_list, rank).
    """
    terms = []
    # For each block (i-block, j-block, k-block):
    # i-block is always size 2 (n=2)
    # j-block from m_parts, k-block from p_parts
    for j_start, j_size in zip(_cumsum(m_parts), m_parts):
        for k_start, k_size in zip(_cumsum(p_parts), p_parts):
            if j_size == 2 and k_size == 2:
                terms.extend(_strassen_2x2x2_terms(0, j_start, k_start, n, m, p))
            elif j_size == 2 and k_size == 1:
                for i in range(n):
                    for j in range(j_start, j_start + j_size):
                        terms.append(_schoolbook_term(i, j, k_start, n, m, p))
            elif j_size == 1 and k_size == 2:
                for i in range(n):
                    for k in range(k_start, k_start + k_size):
                        terms.append(_schoolbook_term(i, j_start, k, n, m, p))
            elif j_size == 1 and k_size == 1:
                for i in range(n):
                    terms.append(_schoolbook_term(i, j_start, k_start, n, m, p))
            else:
                # Fallback: schoolbook for larger blocks
                for i in range(n):
                    for j in range(j_start, j_start + j_size):
                        for k in range(k_start, k_start + k_size):
                            terms.append(_schoolbook_term(i, j, k, n, m, p))
    return terms, len(terms)


def _cumsum(parts):
    """Return list of start indices for each part."""
    starts = []
    s = 0
    for x in parts:
        starts.append(s)
        s += x
    return starts


def _matrices_from_terms(terms, n, m, p):
    """Build U, V, W numpy arrays from a list of rank-1 terms."""
    rank = len(terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for r, (u_ent, v_ent, w_ent) in enumerate(terms):
        for idx, val in u_ent:
            U[idx, r] = val
        for idx, val in v_ent:
            V[idx, r] = val
        for idx, val in w_ent:
            W[idx, r] = val
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify decomposition against the matrix multiplication tensor."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    Rec = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.max(np.abs(T - Rec))
    return err < 1e-9, err


def _search_best_partition(n, m, p):
    """
    Search over all partitions of m and p into parts of size 1 or 2
    (since n=2, we can only use 2x2 Strassen blocks for the i-dimension).
    Return the best (terms, rank) found.
    """
    def gen_partitions(total, max_part=2):
        """Generate all compositions of total using parts 1 and 2."""
        if total == 0:
            yield []
            return
        if total >= 1:
            for rest in gen_partitions(total - 1, max_part):
                yield [1] + rest
        if total >= 2:
            for rest in gen_partitions(total - 2, max_part):
                yield [2] + rest

    best_terms = None
    best_rank = m * p * n  # schoolbook upper bound

    m_parts_list = list(gen_partitions(m))
    p_parts_list = list(gen_partitions(p))

    for mp in m_parts_list:
        for pp in p_parts_list:
            terms, rank = _build_from_partition(mp, pp, n, m, p)
            if rank < best_rank:
                best_rank = rank
                best_terms = terms

    return best_terms, best_rank


def construct():
    """
    Rank-minimal exact tensor decomposition for 2x5 * 5x5 matrix multiplication.
    Uses block-Strassen (2x2x2 Strassen + schoolbook for 1-wide blocks) with
    search over all 1/2-partitions of the inner dimensions.
    """
    start_time = time.time()
    deadline = start_time + 135

    n, m, p = DIMENSIONS

    # Search for the best partition
    best_terms, best_rank = _search_best_partition(n, m, p)

    if time.time() > deadline - 5:
        # Emergency: just use schoolbook
        nmp = n * m * p
        U = np.zeros((n * m, nmp))
        V = np.zeros((m * p, nmp))
        W = np.zeros((p * n, nmp))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Build matrices from best terms
    U, V, W = _matrices_from_terms(best_terms, n, m, p)

    # Verify
    valid, err = _verify(U, V, W, n, m, p)

    if not valid:
        # Fallback to schoolbook
        nmp = n * m * p
        U = np.zeros((n * m, nmp))
        V = np.zeros((m * p, nmp))
        W = np.zeros((p * n, nmp))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1

    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
