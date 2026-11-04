"""Combined program for 3x5x5 matrix multiplication tensor decomposition.
Uses block decomposition with Strassen for 2x2x2 sub-blocks.
Verifies Strassen correctness at startup; falls back to schoolbook if invalid.
"""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_222():
    """
    Return (U, V, W) for 2x2x2 Strassen multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    Standard Strassen:
      M1 = (A00+A11)(B00+B11)
      M2 = (A10+A11)B00
      M3 = A00(B01-B11)
      M4 = A11(B10-B00)
      M5 = (A00+A01)B11
      M6 = (A10-A00)(B00+B01)
      M7 = (A01-A11)(B10+B11)
      C00 = M1 + M4 - M5 + M7
      C01 = M3 + M5
      C10 = M2 + M4
      C11 = M1 - M2 + M3 + M6
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]

    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1

    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1

    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1

    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1

    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    return U, V, W


def _verify_strassen():
    """Exhaustively verify the Strassen 2x2x2 decomposition."""
    U, V, W = _strassen_222()
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = np.sum(W_np[k*2+i] * U_np[i*2+j] * V_np[j*2+k])
                expected = 1.0 if (i == k) else 0.0
                if not np.isclose(s, expected, atol=1e-10):
                    return False
    return True


def _schoolbook_factors(n, m, p):
    """Return (U, V, W) as lists of lists for schoolbook multiplication."""
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
    return U, V, W


def _estimate_rank(i_blocks, j_blocks, k_blocks):
    """Fast rank estimate without building matrices."""
    total = 0
    for i_set in i_blocks:
        for j_set in j_blocks:
            for k_set in k_blocks:
                if len(i_set) == 2 and len(j_set) == 2 and len(k_set) == 2:
                    total += 7
                else:
                    total += len(i_set) * len(j_set) * len(k_set)
    return total


def _build_and_verify(i_blocks, j_blocks, k_blocks, n, m, p):
    """
    Build a block decomposition using Strassen for 2x2x2 blocks and schoolbook otherwise.
    Returns (U, V, W, rank) or None if verification fails.
    """
    su, sv, sw = _strassen_222()
    su_np = np.array(su, dtype=float)
    sv_np = np.array(sv, dtype=float)
    sw_np = np.array(sw, dtype=float)

    total_rank = _estimate_rank(i_blocks, j_blocks, k_blocks)

    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)

    col = 0

    for i_set in i_blocks:
        for j_set in j_blocks:
            for k_set in k_blocks:
                if len(i_set) == 2 and len(j_set) == 2 and len(k_set) == 2:
                    for c in range(7):
                        for li in range(2):
                            gi = i_set[li]
                            for lj in range(2):
                                gj = j_set[lj]
                                U[gi * m + gj, col + c] = su_np[li * 2 + lj, c]
                        for lj in range(2):
                            gj = j_set[lj]
                            for lk in range(2):
                                gk = k_set[lk]
                                V[gj * p + gk, col + c] = sv_np[lj * 2 + lk, c]
                        for lk in range(2):
                            gk = k_set[lk]
                            for li in range(2):
                                gi = i_set[li]
                                W[gk * n + gi, col + c] = sw_np[lk * 2 + li, c]
                    col += 7
                else:
                    for gi in i_set:
                        for gj in j_set:
                            for gk in k_set:
                                U[gi * m + gj, col] = 1.0
                                V[gj * p + gk, col] = 1.0
                                W[gk * n + gi, col] = 1.0
                                col += 1

    if col != total_rank:
        return None

    # Fast verification using einsum: sum over r of W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r]
    # Reshape for batched verification
    # For each (i,j,k), we need: sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r]
    # This is equivalent to: for fixed (i,j), the vector W[:, i] (over k) dotted with elementwise product
    # Let's verify via tensor contraction
    # Build the full tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Expected: T[i,j,k] = 1 if i==k else 0

    # Efficient check: for each r, the contribution to T is outer product of U[:,r], V[:,r], W[:,r]
    # But that's expensive. Instead, verify a subset or use the fact that entries are small integers.
    
    # Direct verification but vectorized:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We can compute this as: for each r, add U_col[:,None,None] * V_col[None,:,None] * W_col[None,None,:]
    # But that's O(n*m*p*rank) which is fine for our sizes.
    
    # Actually, let's just do the direct check with numpy for speed
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j, :]
            for k in range(p):
                s = np.dot(W[k * n + i, :], u_row * V[j * p + k, :])
                expected = 1.0 if (i == k) else 0.0
                if not np.isclose(s, expected, atol=1e-10):
                    return None

    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return U_list, V_list, W_list, total_rank


def _all_partitions(size, max_block=3):
    """
    Generate all partitions of range(size) into blocks of size 1..max_block.
    Returns a list of tuples of tuples (each inner tuple sorted).
    """
    elements = list(range(size))

    def _gen(remaining, current):
        if not remaining:
            yield tuple(current)
            return
        first = remaining[0]
        rest = remaining[1:]
        # Block of size 1
        for sub in _gen(rest, current + [(first,)]):
            yield sub
        # Block of size 2
        if len(remaining) >= 2:
            for sub in _gen(rest[1:], current + [(first, rest[0])]):
                yield sub
        # Block of size 3
        if len(remaining) >= 3:
            for sub in _gen(rest[2:], current + [(first, rest[0], rest[1])]):
                yield sub

    results = []
    seen = set()
    for part in _gen(elements, []):
        normalized = tuple(tuple(sorted(b)) for b in part)
        if normalized not in seen:
            seen.add(normalized)
            results.append(normalized)
    return results


def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5

    deadline = time.time() + 130  # safety margin before 150s limit

    # Verify Strassen first
    strassen_ok = _verify_strassen()

    if not strassen_ok:
        # Fall back to schoolbook if Strassen verification fails
        U, V, W = _schoolbook_factors(n, m, p)
        return {"U": U, "V": V, "W": W}

    # Generate all partitions
    i_partitions = _all_partitions(3, max_block=3)
    j_partitions = _all_partitions(5, max_block=3)
    k_partitions = _all_partitions(5, max_block=3)

    # Estimate ranks and sort candidates
    candidates = []
    for i_blocks in i_partitions:
        for j_blocks in j_partitions:
            for k_blocks in k_partitions:
                est_rank = _estimate_rank(i_blocks, j_blocks, k_blocks)
                candidates.append((est_rank, i_blocks, j_blocks, k_blocks))

    candidates.sort(key=lambda x: x[0])

    best_result = None
    best_rank = float('inf')

    for est_rank, i_blocks, j_blocks, k_blocks in candidates:
        if time.time() > deadline:
            break
        if est_rank >= best_rank:
            continue
        try:
            result = _build_and_verify(i_blocks, j_blocks, k_blocks, n, m, p)
            if result is not None:
                U, V, W, rank = result
                if rank < best_rank:
                    best_rank = rank
                    best_result = (U, V, W)
        except Exception:
            pass

    if best_result is not None:
        U, V, W = best_result
        return {"U": U, "V": V, "W": W}
    else:
        U, V, W = _schoolbook_factors(n, m, p)
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
