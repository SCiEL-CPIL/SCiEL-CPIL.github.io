import time
import numpy as np

DIMENSIONS = [3, 4, 5]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 60


def _verify(U, V, W):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = N, M, P
    if not U or not V or not W:
        return False
    r = len(U[0])
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    for row in U:
        if len(row) != r:
            return False
    for row in V:
        if len(row) != r:
            return False
    for row in W:
        if len(row) != r:
            return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True


def _schoolbook():
    """Return the trivial schoolbook decomposition of rank n*m*p."""
    n, m, p = N, M, P
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


def _strassen_terms(i_off, j_off, k_off):
    """
    Return the 7 Strassen terms for a 2x2x2 block.
    Each term is (U_dict, V_dict, W_dict) where each dict maps global indices to coefficients.
    U_dict keys are (i, j) tuples, V_dict keys are (j, k) tuples, W_dict keys are (k, i) tuples.
    """
    terms = []

    # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # Contributes to C[0,0] with +1, C[1,1] with +1
    U1 = {(i_off + 0, j_off + 0): 1, (i_off + 1, j_off + 1): 1}
    V1 = {(j_off + 0, k_off + 0): 1, (j_off + 1, k_off + 1): 1}
    W1 = {(k_off + 0, i_off + 0): 1, (k_off + 1, i_off + 1): 1}
    terms.append((U1, V1, W1))

    # M2 = (A[1,0] + A[1,1]) * B[0,0]
    # Contributes to C[1,0] with +1, C[1,1] with -1
    U2 = {(i_off + 1, j_off + 0): 1, (i_off + 1, j_off + 1): 1}
    V2 = {(j_off + 0, k_off + 0): 1}
    W2 = {(k_off + 0, i_off + 1): 1, (k_off + 1, i_off + 1): -1}
    terms.append((U2, V2, W2))

    # M3 = A[0,0] * (B[0,1] - B[1,1])
    # Contributes to C[0,1] with +1, C[1,1] with +1
    U3 = {(i_off + 0, j_off + 0): 1}
    V3 = {(j_off + 0, k_off + 1): 1, (j_off + 1, k_off + 1): -1}
    W3 = {(k_off + 1, i_off + 0): 1, (k_off + 1, i_off + 1): 1}
    terms.append((U3, V3, W3))

    # M4 = A[1,1] * (B[1,0] - B[0,0])
    # Contributes to C[1,0] with +1, C[0,0] with +1
    U4 = {(i_off + 1, j_off + 1): 1}
    V4 = {(j_off + 1, k_off + 0): 1, (j_off + 0, k_off + 0): -1}
    W4 = {(k_off + 0, i_off + 1): 1, (k_off + 0, i_off + 0): 1}
    terms.append((U4, V4, W4))

    # M5 = (A[0,0] + A[0,1]) * B[1,1]
    # Contributes to C[0,1] with +1, C[0,0] with -1
    U5 = {(i_off + 0, j_off + 0): 1, (i_off + 0, j_off + 1): 1}
    V5 = {(j_off + 1, k_off + 1): 1}
    W5 = {(k_off + 1, i_off + 0): 1, (k_off + 0, i_off + 0): -1}
    terms.append((U5, V5, W5))

    # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    # Contributes to C[1,1] with +1
    U6 = {(i_off + 1, j_off + 0): 1, (i_off + 0, j_off + 0): -1}
    V6 = {(j_off + 0, k_off + 0): 1, (j_off + 0, k_off + 1): 1}
    W6 = {(k_off + 1, i_off + 1): 1}
    terms.append((U6, V6, W6))

    # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    # Contributes to C[0,0] with +1
    U7 = {(i_off + 0, j_off + 1): 1, (i_off + 1, j_off + 1): -1}
    V7 = {(j_off + 1, k_off + 0): 1, (j_off + 1, k_off + 1): 1}
    W7 = {(k_off + 0, i_off + 0): 1}
    terms.append((U7, W7) if False else (U7, V7, W7))

    return terms


def _build_strassen_decomposition(blocks):
    """
    Build a decomposition using the given Strassen blocks and schoolbook for the rest.
    blocks: list of (i_off, j_off, k_off) tuples
    Returns (result_dict, rank) or (None, None) if invalid.
    """
    n, m, p = N, M, P

    # Identify which (i,j,k) tuples are covered by Strassen blocks.
    strassen_covered = set()
    for i_off, j_off, k_off in blocks:
        for ii in range(2):
            for jj in range(2):
                for kk in range(2):
                    triple = (i_off + ii, j_off + jj, k_off + kk)
                    if triple in strassen_covered:
                        return None, None  # Overlapping blocks
                    strassen_covered.add(triple)

    # Build the full decomposition.
    all_terms = []

    # Add Strassen terms for each block
    for i_off, j_off, k_off in blocks:
        block_terms = _strassen_terms(i_off, j_off, k_off)
        for U_dict, V_dict, W_dict in block_terms:
            all_terms.append((U_dict, V_dict, W_dict))

    # Add schoolbook terms for all other (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in strassen_covered:
                    U_sb = {(i, j): 1}
                    V_sb = {(j, k): 1}
                    W_sb = {(k, i): 1}
                    all_terms.append((U_sb, V_sb, W_sb))

    rank = len(all_terms)

    # Convert to matrix form
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (U_dict, V_dict, W_dict) in enumerate(all_terms):
        for (i, j), coeff in U_dict.items():
            if coeff != 0:
                U[i * m + j][c] = coeff
        for (j, k), coeff in V_dict.items():
            if coeff != 0:
                V[j * p + k][c] = coeff
        for (k, i), coeff in W_dict.items():
            if coeff != 0:
                W[k * n + i][c] = coeff

    return {"U": U, "V": V, "W": W}, rank


def _find_max_disjoint_blocks():
    """
    Find the maximum set of disjoint 2x2x2 Strassen blocks in the n x m x p grid.
    Uses backtracking search with pruning.
    """
    n, m, p = N, M, P

    # Generate all possible blocks
    all_blocks = []
    for i_off in range(n - 1):
        for j_off in range(m - 1):
            for k_off in range(p - 1):
                if i_off + 1 < n and j_off + 1 < m and k_off + 1 < p:
                    all_blocks.append((i_off, j_off, k_off))

    def get_triples(block):
        i_off, j_off, k_off = block
        return frozenset((i_off + ii, j_off + jj, k_off + kk)
                         for ii in range(2) for jj in range(2) for kk in range(2))

    # Precompute triples for each block
    block_triples_list = [get_triples(b) for b in all_blocks]

    # Precompute conflict matrix: which blocks conflict with which
    num_blocks = len(all_blocks)
    conflicts = [set() for _ in range(num_blocks)]
    for a in range(num_blocks):
        for b in range(a + 1, num_blocks):
            if block_triples_list[a] & block_triples_list[b]:
                conflicts[a].add(b)
                conflicts[b].add(a)

    best = []
    best_set = frozenset()

    # Upper bound: remaining blocks that don't conflict with current selection
    def upper_bound(start, current_indices, covered):
        """Upper bound on total blocks we can still add."""
        # Simple bound: count blocks from start onwards that don't conflict with covered
        count = len(current_indices)
        for idx in range(start, num_blocks):
            if not block_triples_list[idx] & covered:
                count += 1
        return count

    def backtrack(start, current, covered, current_indices):
        nonlocal best, best_set
        if len(current) > len(best):
            best = current[:]
            best_set = frozenset(current_indices)

        # Pruning: if we can't beat best, stop
        if upper_bound(start, current_indices, covered) <= len(best):
            return

        for idx in range(start, num_blocks):
            triples = block_triples_list[idx]
            if not triples & covered:
                new_covered = covered | triples
                current.append(all_blocks[idx])
                current_indices.append(idx)
                backtrack(idx + 1, current, new_covered, current_indices)
                current.pop()
                current_indices.pop()

    backtrack(0, [], frozenset(), [])
    return best


def _try_reduce_rank_by_dependency(result, U, V, W, rank):
    """
    Try to find linear dependencies among rank-1 terms to reduce rank.
    For each pair of terms, check if one is a scalar multiple of the other
    (component-wise across U, V, W). If so, merge them.
    """
    n, m, p = N, M, P
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)

    # Build the full tensor to verify
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We want to check if removing column c still gives correct tensor.

    best_rank = rank
    best_result = result

    for c in range(rank):
        # Try removing column c
        U_test = np.delete(U_mat, c, axis=1)
        V_test = np.delete(V_mat, c, axis=1)
        W_test = np.delete(W_mat, c, axis=1)

        # Check if the decomposition still works
        valid = True
        for i in range(n):
            if not valid:
                break
            for j in range(m):
                if not valid:
                    break
                for k in range(p):
                    s = 0.0
                    for cc in range(rank - 1):
                        s += U_test[i * m + j, cc] * V_test[j * p + k, cc] * W_test[k * n + i, cc]
                    if abs(s - 1.0) > 1e-10:
                        valid = False
                        break

        if valid:
            best_rank = rank - 1
            best_result = {
                "U": U_test.tolist(),
                "V": V_test.tolist(),
                "W": W_test.tolist()
            }
            # Update U, V, W for further removals
            U_mat = U_test
            V_mat = V_test
            W_mat = W_test
            rank = rank - 1

    return best_result, best_rank


def construct():
    n, m, p = N, M, P
    t0 = time.time()
    deadline = t0 + 140  # Safety margin for 150s limit

    # Known good configuration: 4 disjoint Strassen blocks
    known_blocks = [
        (0, 0, 0),  # i={0,1}, j={0,1}, k={0,1}
        (0, 2, 2),  # i={0,1}, j={2,3}, k={2,3}
        (1, 0, 3),  # i={1,2}, j={0,1}, k={3,4}
        (1, 2, 0),  # i={1,2}, j={2,3}, k={0,1}
    ]

    # Try the known configuration first
    result, rank = _build_strassen_decomposition(known_blocks)
    if result is not None and _verify(result["U"], result["V"], result["W"]):
        # Try to reduce rank by finding dependencies
        if time.time() < deadline:
            result, rank = _try_reduce_rank_by_dependency(result, result["U"], result["V"], result["W"], rank)
            if _verify(result["U"], result["V"], result["W"]):
                return result
        return result

    # Search for maximum disjoint blocks
    if time.time() < deadline:
        max_blocks = _find_max_disjoint_blocks()
        if len(max_blocks) > len(known_blocks):
            result, rank = _build_strassen_decomposition(max_blocks)
            if result is not None and _verify(result["U"], result["V"], result["W"]):
                # Try to reduce rank by finding dependencies
                if time.time() < deadline:
                    result, rank = _try_reduce_rank_by_dependency(result, result["U"], result["V"], result["W"], rank)
                    if _verify(result["U"], result["V"], result["W"]):
                        return result
                return result

    # Fallback to schoolbook
    return _schoolbook()
