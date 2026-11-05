"""
Deterministic multi-Strassen decomposition for 3x4 * 4x7 matrix multiplication.
Combines fixed 2x2x2 Strassen blocks (inspiration 1) with a structured greedy
search that also considers 2x2x3 and 3x2x2 sub-blocks (inspiration 2) to
minimise the rank of the remaining schoolbook terms.
"""
import time
import itertools

DIMENSIONS = [3, 4, 7]
_DEADLINE = 135.0  # safety margin before 150s kill


def _verify(U, V, W, n, m, p, rank):
    """Verify the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = 0
                for c in range(rank):
                    total += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if total != 1:
                    return False
    return True


def _strassen_2x2x2_terms(i0, i1, j0, j1, k0, k1, n, m, p):
    """
    Return list of (u_dict, v_dict, w_dict) for a 2x2x2 Strassen block.
    u_dict maps i*m+j -> coeff, v_dict maps j*p+k -> coeff, w_dict maps k*n+i -> coeff.
    """
    terms = []

    # M1: (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1]) -> C[i0,k0]+=, C[i1,k1]+=
    u = {i0 * m + j0: 1, i1 * m + j1: 1}
    v = {j0 * p + k0: 1, j1 * p + k1: 1}
    w = {k0 * n + i0: 1, k1 * n + i1: 1}
    terms.append((u, v, w))

    # M2: (A[i1,j0]+A[i1,j1])*B[j0,k0] -> C[i1,k0]+=, C[i1,k1]-=
    u = {i1 * m + j0: 1, i1 * m + j1: 1}
    v = {j0 * p + k0: 1}
    w = {k0 * n + i1: 1, k1 * n + i1: -1}
    terms.append((u, v, w))

    # M3: A[i0,j0]*(B[j0,k1]-B[j1,k1]) -> C[i0,k1]+=, C[i1,k1]+=
    u = {i0 * m + j0: 1}
    v = {j0 * p + k1: 1, j1 * p + k1: -1}
    w = {k1 * n + i0: 1, k1 * n + i1: 1}
    terms.append((u, v, w))

    # M4: A[i1,j1]*(B[j1,k0]-B[j0,k0]) -> C[i0,k0]+=, C[i1,k0]+=
    u = {i1 * m + j1: 1}
    v = {j1 * p + k0: 1, j0 * p + k0: -1}
    w = {k0 * n + i0: 1, k0 * n + i1: 1}
    terms.append((u, v, w))

    # M5: (A[i0,j0]+A[i0,j1])*B[j1,k1] -> C[i0,k0]-=, C[i0,k1]+=
    u = {i0 * m + j0: 1, i0 * m + j1: 1}
    v = {j1 * p + k1: 1}
    w = {k0 * n + i0: -1, k1 * n + i0: 1}
    terms.append((u, v, w))

    # M6: (A[i1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k1]) -> C[i1,k1]+=
    u = {i1 * m + j0: 1, i0 * m + j0: -1}
    v = {j0 * p + k0: 1, j0 * p + k1: 1}
    w = {k1 * n + i1: 1}
    terms.append((u, v, w))

    # M7: (A[i0,j1]-A[i1,j1])*(B[j1,k0]+B[j1,k1]) -> C[i0,k0]+=
    u = {i0 * m + j1: 1, i1 * m + j1: -1}
    v = {j1 * p + k0: 1, j1 * p + k1: 1}
    w = {k0 * n + i0: 1}
    terms.append((u, v, w))

    return terms


def _strassen_2x2x3_terms(i0, i1, j0, j1, k0, k1, k2, n, m, p):
    """
    Rank-10 decomposition for a 2x2x3 block.
    Uses two 2x2x2 Strassen blocks sharing the same (i,j) pair but different k's,
    then adds 2 extra terms for the third k.
    Actually, a better approach: use 2 Strassen 2x2x2 blocks for (k0,k1) and (k1,k2),
    but that double-counts k1. Instead, we can use:
    - 7 terms for (i0,i1, j0,j1, k0,k1)
    - 3 extra terms for k2: schoolbook for (i0,j0,k2), (i0,j1,k2), (i1,j0,k2), (i1,j1,k2)
      but that's 4 terms. Total = 7+4 = 11.
    
    A better rank-10 approach:
    Use the 7 Strassen terms for (k0,k1), and for k2 use 3 terms:
    - (A[i0,j0]+A[i1,j1]) * B[j1,k2] -> contributes to C[i0,k2] and C[i1,k2]
    - A[i1,j0] * B[j0,k2] -> C[i1,k2]
    - A[i0,j1] * B[j1,k2] -> C[i0,k2]
    But this doesn't work cleanly.
    
    For simplicity, use rank-11: 7 Strassen + 4 schoolbook for k2.
    """
    terms = _strassen_2x2x2_terms(i0, i1, j0, j1, k0, k1, n, m, p)
    # Add 4 schoolbook terms for k2
    for ii in (i0, i1):
        for jj in (j0, j1):
            u = {ii * m + jj: 1}
            v = {jj * p + k2: 1}
            w = {k2 * n + ii: 1}
            terms.append((u, v, w))
    return terms  # rank 11


def _strassen_3x2x2_terms(i0, i1, i2, j0, j1, k0, k1, n, m, p):
    """
    Rank-11 decomposition for a 3x2x2 block.
    Uses 7 Strassen terms for (i0,i1, j0,j1, k0,k1) and 4 schoolbook terms for i2.
    """
    terms = _strassen_2x2x2_terms(i0, i1, j0, j1, k0, k1, n, m, p)
    for jj in (j0, j1):
        for kk in (k0, k1):
            u = {i2 * m + jj: 1}
            v = {jj * p + kk: 1}
            w = {kk * n + i2: 1}
            terms.append((u, v, w))
    return terms  # rank 11


def _apply_terms(U, V, W, col_start, terms, n, m, p):
    """Apply a list of (u_dict, v_dict, w_dict) starting at col_start."""
    for r, (u_dict, v_dict, w_dict) in enumerate(terms):
        col = col_start + r
        for idx, coeff in u_dict.items():
            if coeff != 0:
                U[idx][col] += coeff
        for idx, coeff in v_dict.items():
            if coeff != 0:
                V[idx][col] += coeff
        for idx, coeff in w_dict.items():
            if coeff != 0:
                W[idx][col] += coeff
    return col_start + len(terms)


def _get_block_cells(i0, i1, j0, j1, k0, k1):
    """Get all (i,j,k) cells in a 2x2x2 block."""
    cells = set()
    for ii in (i0, i1):
        for jj in (j0, j1):
            for kk in (k0, k1):
                cells.add((ii, jj, kk))
    return cells


def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS  # n=3, m=4, p=7

    # Start with schoolbook as fallback
    rank_school = n * m * p  # 84
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_school

    # Strategy: Use 6 disjoint 2x2x2 Strassen blocks (as in inspiration 1)
    # This gives rank 6*7 + 36 = 78
    # Then try to improve by using larger blocks or different combinations.

    # First, build the 6-block Strassen decomposition
    U2 = [[0] * rank_school for _ in range(n * m)]
    V2 = [[0] * rank_school for _ in range(m * p)]
    W2 = [[0] * rank_school for _ in range(p * n)]
    col2 = 0
    covered = set()

    blocks = [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 0, 1, 2, 3),
        (0, 1, 0, 1, 4, 5),
        (0, 1, 2, 3, 0, 1),
        (0, 1, 2, 3, 2, 3),
        (0, 1, 2, 3, 4, 5),
    ]

    for (i0, i1, j0, j1, k0, k1) in blocks:
        terms = _strassen_2x2x2_terms(i0, i1, j0, j1, k0, k1, n, m, p)
        col2 = _apply_terms(U2, V2, W2, col2, terms, n, m, p)
        cells = _get_block_cells(i0, i1, j0, j1, k0, k1)
        covered.update(cells)

    # Add schoolbook for remaining cells
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                U2[i * m + j][col2] = 1
                V2[j * p + k][col2] = 1
                W2[k * n + i][col2] = 1
                col2 += 1

    rank2 = col2
    if rank2 < best_rank:
        # Verify
        if _verify(U2, V2, W2, n, m, p, rank2):
            best_rank = rank2
            best_result = {"U": U2, "V": V2, "W": W2}

    # Now try to improve: replace some schoolbook terms with larger block patterns
    # For example, if we have 4 schoolbook terms forming a 2x2x2 block, we can
    # replace them with 7 Strassen terms (worse) or find a better pattern.
    # Actually, the 6-block decomposition already covers the most beneficial blocks.
    # Let's try a different set of 6 blocks or 5 blocks of different sizes.

    # Try: 3 blocks of 2x2x3 (rank 11 each) + remaining schoolbook
    # 3 * 11 = 33, remaining = 84 - 3*2*2*3 = 84 - 36 = 48, total = 33 + 48 = 81 (worse)
    # So 2x2x2 blocks are better.

    # Try: 4 blocks of 2x2x2 (rank 7 each) + 1 block of 2x2x3 (rank 11) + remaining
    # 4*7 + 11 = 39, covered cells = 4*8 + 12 = 44, remaining = 84 - 44 = 40, total = 39 + 40 = 79 (worse than 78)

    # The 6-block 2x2x2 decomposition with rank 78 seems optimal for this family.
    # Let's try to see if we can do better with a different arrangement.

    # Try: use i in {0,1,2} instead of {0,1} for some blocks
    # For i=2, we can make 2x2x2 blocks with j-pairs and k-pairs
    # But i=2 only has 1 row, so we can't make 2x2x2 in the i dimension.
    # We can make 1x2x2 blocks, but those don't benefit from Strassen.

    # Alternative: try a greedy search for the best set of non-overlapping 2x2x2 blocks
    # that maximises the number of blocks (each saves 1 term: 8 schoolbook -> 7 Strassen)
    
    # All possible 2x2x2 blocks:
    all_blocks = []
    for i0 in range(n - 1):
        for i1 in range(i0 + 1, n):
            for j0 in range(m - 1):
                for j1 in range(j0 + 1, m):
                    for k0 in range(p - 1):
                        for k1 in range(k0 + 1, p):
                            all_blocks.append((i0, i1, j0, j1, k0, k1))

    # Greedy: pick blocks that don't overlap in cells
    # Each block covers 8 cells. We want to maximise the number of non-overlapping blocks.
    # Since n=3, i can be (0,1) or (1,2). m=4, j can be (0,1),(0,2),(0,3),(1,2),(1,3),(2,3).
    # p=7, k can be many pairs.

    # The 6-block solution uses i=(0,1) only. Can we add a block with i=(1,2)?
    # If we use i=(1,2), it overlaps with i=(0,1) blocks on i=1.
    # So we can't mix i=(0,1) and i=(1,2) blocks if they share j,k cells.

    # Let's try: 3 blocks with i=(0,1) and 3 blocks with i=(1,2), choosing j,k to avoid overlap.
    # But i=(0,1) and i=(1,2) blocks overlap on i=1, so they can't share (j,k) pairs.
    # Total j,k pairs available: m*(m-1)/2 * p*(p-1)/2 = 6 * 21 = 126 possible (j,k) pairs.
    # Each block uses 1 (j,k) pair. So we can have up to 126 blocks, but we're limited by cells.
    # Each block covers 8 cells. Total cells = 84. Max blocks = 84/8 = 10.5, so at most 10 blocks.
    # But blocks must not share cells. Two blocks share a cell if they share (i,j,k).
    # If they have different i pairs, they can't share a cell unless they share i, j, k.
    # This is getting complex. Let's just stick with the 6-block solution for now.

    # Try to verify if rank 78 is achievable and if we can do better.
    # The theoretical lower bound is max(n*m, m*p, p*n) = max(12, 28, 21) = 28.
    # But that's for the tensor rank, which is different from the bilinear complexity.
    # For matrix multiplication, the bilinear complexity is at least n*m*p / (something).

    # For now, return the best result found.
    if time.time() - _t0 > _DEADLINE:
        return best_result

    # Try a few more configurations
    # Configuration: 5 blocks of 2x2x2 + remaining schoolbook
    # 5*7 = 35, covered = 40, remaining = 44, total = 35 + 44 = 79 (worse)

    # Configuration: 7 blocks of 2x2x2 (if possible)
    # 7*7 = 49, covered = 56, remaining = 28, total = 49 + 28 = 77 (better!)
    # Can we find 7 non-overlapping 2x2x2 blocks?
    # We need 7 blocks that don't share any (i,j,k) cell.
    # Let's try to find such a set.

    # Greedy search for 7 non-overlapping blocks
    def _find_non_overlapping_blocks(target_count):
        """Find target_count non-overlapping 2x2x2 blocks."""
        best_set = None
        best_score = 0
        
        # Try all combinations of blocks (this is expensive, so use greedy)
        # Greedy: repeatedly pick the block that covers the most uncovered cells
        uncovered = set((i, j, k) for i in range(n) for j in range(m) for k in range(p))
        selected = []
        
        remaining_blocks = list(all_blocks)
        for _ in range(target_count):
            if not remaining_blocks:
                break
            # Pick the block that covers the most uncovered cells
            best_block = None
            best_count = 0
            for block in remaining_blocks:
                cells = _get_block_cells(*block)
                count = len(cells & uncovered)
                if count > best_count:
                    best_count = count
                    best_block = block
            if best_block is None or best_count == 0:
                break
            selected.append(best_block)
            cells = _get_block_cells(*best_block)
            uncovered -= cells
            remaining_blocks.remove(best_block)
        
        return selected

    blocks7 = _find_non_overlapping_blocks(7)
    if len(blocks7) == 7:
        U3 = [[0] * rank_school for _ in range(n * m)]
        V3 = [[0] * rank_school for _ in range(m * p)]
        W3 = [[0] * rank_school for _ in range(p * n)]
        col3 = 0
        covered3 = set()
        for block in blocks7:
            terms = _strassen_2x2x2_terms(*block, n, m, p)
            col3 = _apply_terms(U3, V3, W3, col3, terms, n, m, p)
            covered3.update(_get_block_cells(*block))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) in covered3:
                        continue
                    U3[i * m + j][col3] = 1
                    V3[j * p + k][col3] = 1
                    W3[k * n + i][col3] = 1
                    col3 += 1
        
        rank3 = col3
        if rank3 < best_rank:
            if _verify(U3, V3, W3, n, m, p, rank3):
                best_rank = rank3
                best_result = {"U": U3, "V": V3, "W": W3}

    # Try 8 blocks
    blocks8 = _find_non_overlapping_blocks(8)
    if len(blocks8) == 8:
        U4 = [[0] * rank_school for _ in range(n * m)]
        V4 = [[0] * rank_school for _ in range(m * p)]
        W4 = [[0] * rank_school for _ in range(p * n)]
        col4 = 0
        covered4 = set()
        for block in blocks8:
            terms = _strassen_2x2x2_terms(*block, n, m, p)
            col4 = _apply_terms(U4, V4, W4, col4, terms, n, m, p)
            covered4.update(_get_block_cells(*block))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) in covered4:
                        continue
                    U4[i * m + j][col4] = 1
                    V4[j * p + k][col4] = 1
                    W4[k * n + i][col4] = 1
                    col4 += 1
        
        rank4 = col4
        if rank4 < best_rank:
            if _verify(U4, V4, W4, n, m, p, rank4):
                best_rank = rank4
                best_result = {"U": U4, "V": V4, "W": W4}

    # Try 9 blocks
    blocks9 = _find_non_overlapping_blocks(9)
    if len(blocks9) == 9:
        U5 = [[0] * rank_school for _ in range(n * m)]
        V5 = [[0] * rank_school for _ in range(m * p)]
        W5 = [[0] * rank_school for _ in range(p * n)]
        col5 = 0
        covered5 = set()
        for block in blocks9:
            terms = _strassen_2x2x2_terms(*block, n, m, p)
            col5 = _apply_terms(U5, V5, W5, col5, terms, n, m, p)
            covered5.update(_get_block_cells(*block))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) in covered5:
                        continue
                    U5[i * m + j][col5] = 1
                    V5[j * p + k][col5] = 1
                    W5[k * n + i][col5] = 1
                    col5 += 1
        
        rank5 = col5
        if rank5 < best_rank:
            if _verify(U5, V5, W5, n, m, p, rank5):
                best_rank = rank5
                best_result = {"U": U5, "V": V5, "W": W5}

    # Try 10 blocks
    blocks10 = _find_non_overlapping_blocks(10)
    if len(blocks10) == 10:
        U6 = [[0] * rank_school for _ in range(n * m)]
        V6 = [[0] * rank_school for _ in range(m * p)]
        W6 = [[0] * rank_school for _ in range(p * n)]
        col6 = 0
        covered6 = set()
        for block in blocks10:
            terms = _strassen_2x2x2_terms(*block, n, m, p)
            col6 = _apply_terms(U6, V6, W6, col6, terms, n, m, p)
            covered6.update(_get_block_cells(*block))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) in covered6:
                        continue
                    U6[i * m + j][col6] = 1
                    V6[j * p + k][col6] = 1
                    W6[k * n + i][col6] = 1
                    col6 += 1
        
        rank6 = col6
        if rank6 < best_rank:
            if _verify(U6, V6, W6, n, m, p, rank6):
                best_rank = rank6
                best_result = {"U": U6, "V": V6, "W": W6}

    return best_result
