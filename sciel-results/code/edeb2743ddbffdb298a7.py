"""MOLS of order 10 - generalized linear construction over Z_2 x Z_5."""
import time

def _verify_latin(sq, n):
    target = frozenset(range(n))
    for i in range(n):
        if frozenset(sq[i]) != target:
            return False
    for j in range(n):
        if frozenset(sq[i][j] for i in range(n)) != target:
            return False
    return True

def _verify_orthogonal(sq1, sq2, n):
    seen = set()
    for i in range(n):
        r1 = sq1[i]
        r2 = sq2[i]
        for j in range(n):
            p = (r1[j], r2[j])
            if p in seen:
                return False
            seen.add(p)
    return True

def _pair_to_int(a, b):
    return a * 5 + b

def _make_general_linear_sq(A, b):
    """Build a 10x10 Latin square using generalized linear construction over Z_2 x Z_5.
    A is a 2x2 integer matrix [[a00, a01], [a10, a11]]
    b is a tuple (b2, b5) for the Z_2 and Z_5 offsets.
    For position (i, j):
      i2 = i // 5, i5 = i % 5
      j2 = j // 5, j5 = j % 5
      a_out = (A[0][0]*i2 + A[0][1]*j2 + b[0]) % 2
      b_out = (A[1][0]*i5 + A[1][1]*j5 + b[1]) % 5
    """
    a00, a01 = A[0]
    a10, a11 = A[1]
    b2, b5 = b
    sq = []
    for i in range(10):
        i2, i5 = i // 5, i % 5
        row = []
        for j in range(10):
            j2, j5 = j // 5, j % 5
            a_out = (a00 * i2 + a01 * j2 + b2) % 2
            b_out = (a10 * i5 + a11 * j5 + b5) % 5
            row.append(_pair_to_int(a_out, b_out))
        sq.append(row)
    return sq

def construct():
    start = time.time()
    deadline = start + 80.0

    # Generate candidate squares from generalized linear construction.
    # A is a 2x2 matrix over integers (mod 2 for first row, mod 5 for second row).
    # For the square to be Latin, we need:
    #   - The map (i2,j2) -> (A[0][0]*i2 + A[0][1]*j2) mod 2 is bijective for fixed i5,j5
    #   - The map (i5,j5) -> (A[1][0]*i5 + A[1][1]*j5) mod 5 is bijective for fixed i2,j2
    # This requires A[0][1] != 0 mod 2 (i.e., A[0][1] is odd) and A[1][1] != 0 mod 5.
    # More generally, for the full 2D structure to work, we need the 2x2 matrix
    # [[A[0][0], A[0][1]], [A[1][0], A[1][1]]] to have the right properties.
    #
    # Actually, for the Latin property:
    #   Fixing row i (i2, i5): as j varies (j2 in {0,1}, j5 in {0..4}),
    #   a_out = (A[0][0]*i2 + A[0][1]*j2 + b2) mod 2
    #   b_out = (A[1][0]*i5 + A[1][1]*j5 + b5) mod 5
    #   We need all 10 pairs (a_out, b_out) to be distinct.
    #   a_out takes values based on j2: if A[0][1] is odd, j2=0 gives one value, j2=1 gives other.
    #   b_out takes values based on j5: if A[1][1] is nonzero mod 5, j5 maps bijectively.
    #   So for each fixed i, the 10 outputs are: for j2 in {0,1}, j5 in {0..4}:
    #     ( (A[0][0]*i2 + A[0][1]*j2 + b2) % 2, (A[1][0]*i5 + A[1][1]*j5 + b5) % 5 )
    #   This gives 2*5 = 10 distinct pairs iff A[0][1] is odd AND A[1][1] != 0 mod 5.
    #
    # For columns: fixing column j (j2, j5), as i varies (i2 in {0,1}, i5 in {0..4}):
    #   a_out = (A[0][0]*i2 + A[0][1]*j2 + b2) mod 2
    #   b_out = (A[1][0]*i5 + A[1][1]*j5 + b5) mod 5
    #   Similarly need A[0][0] odd and A[1][0] != 0 mod 5 for bijectivity in i.
    #
    # So conditions: A[0][0] odd, A[0][1] odd, A[1][0] != 0 mod 5, A[1][1] != 0 mod 5.

    candidates = []
    seen_patterns = set()

    # A[0][0] in {1} (odd), A[0][1] in {1} (odd) -- only 1 choice mod 2 for each
    # A[1][0] in {1,2,3,4} (nonzero mod 5), A[1][1] in {1,2,3,4} (nonzero mod 5)
    # b2 in {0,1}, b5 in {0,1,2,3,4}
    for a00 in [1]:  # must be odd
        for a01 in [1]:  # must be odd
            for a10 in [1, 2, 3, 4]:
                for a11 in [1, 2, 3, 4]:
                    A = [(a00, a01), (a10, a11)]
                    for b2 in [0, 1]:
                        for b5 in range(5):
                            b = (b2, b5)
                            sq = _make_general_linear_sq(A, b)
                            if _verify_latin(sq, 10):
                                pat = tuple(tuple(row) for row in sq)
                                if pat not in seen_patterns:
                                    seen_patterns.add(pat)
                                    candidates.append(sq)

    # Also try the "swapped" index convention: i = (i5, i2) instead of (i2, i5)
    # This gives a different family of squares
    seen_patterns2 = set()
    for a00 in [1]:
        for a01 in [1]:
            for a10 in [1, 2, 3, 4]:
                for a11 in [1, 2, 3, 4]:
                    A = [(a00, a01), (a10, a11)]
                    for b2 in [0, 1]:
                        for b5 in range(5):
                            b = (b2, b5)
                            sq = []
                            for i in range(10):
                                i5, i2 = i % 5, i // 5
                                row = []
                                for j in range(10):
                                    j5, j2 = j % 5, j // 5
                                    a_out = (a00 * i2 + a01 * j2 + b2) % 2
                                    b_out = (a10 * i5 + a11 * j5 + b5) % 5
                                    row.append(_pair_to_int(a_out, b_out))
                                sq.append(row)
                            if _verify_latin(sq, 10):
                                pat = tuple(tuple(row) for row in sq)
                                if pat not in seen_patterns2:
                                    seen_patterns2.add(pat)
                                    candidates.append(sq)

    # Greedy: find largest mutually orthogonal set
    best_set = []
    for sq in candidates:
        ok = True
        for sq2 in best_set:
            if not _verify_orthogonal(sq, sq2, 10):
                ok = False
                break
        if ok:
            best_set.append(sq)
            if len(best_set) >= 9:
                break

    # Backtracking to try to find a larger set
    if len(best_set) < 9 and time.time() < deadline - 15:
        best_size = len(best_set)
        best_set_copy = best_set[:]

        def backtrack(idx, current_set):
            nonlocal best_size, best_set_copy
            if time.time() > deadline - 5:
                return
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set_copy = current_set[:]
                if best_size >= 9:
                    return
            for i in range(idx, len(candidates)):
                sq = candidates[i]
                ok = True
                for sq2 in current_set:
                    if not _verify_orthogonal(sq, sq2, 10):
                        ok = False
                        break
                if ok:
                    current_set.append(sq)
                    backtrack(i + 1, current_set)
                    current_set.pop()
                    if len(current_set) >= 9:
                        return

        backtrack(0, best_set[:])
        if best_size > len(best_set):
            best_set = best_set_copy

    # Ensure we have at least 1 square
    if len(best_set) == 0:
        best_set = [[(i + j) % 10 for j in range(10)] for i in range(10)]

    # Convert to pure Python lists of ints
    result = [[[int(x) for x in row] for row in sq] for sq in best_set]
    return result
