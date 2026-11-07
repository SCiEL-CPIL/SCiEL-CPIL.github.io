"""MOLS of order 10 - linear construction over Z_2 x Z_5."""
import time
import itertools

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

def _make_linear_sq(c2, c5):
    """Build a 10x10 Latin square using Z_2 x Z_5 linear construction.
    For position (i, j):
      i2 = i // 5, i5 = i % 5
      j2 = j // 5, j5 = j % 5
      a_out = (c2 * i2 + j2) % 2
      b_out = (c5 * i5 + j5) % 5
    """
    sq = []
    for i in range(10):
        i2, i5 = i // 5, i % 5
        row = []
        for j in range(10):
            j2, j5 = j // 5, j % 5
            a_out = (c2 * i2 + j2) % 2
            b_out = (c5 * i5 + j5) % 5
            row.append(_pair_to_int(a_out, b_out))
        sq.append(row)
    return sq

def construct():
    start = time.time()
    deadline = start + 50.0

    # Generate all 10 linear squares (c2 in {0,1}, c5 in {0,1,2,3,4})
    linear_sqs = {}
    for c2 in range(2):
        for c5 in range(5):
            sq = _make_linear_sq(c2, c5)
            if _verify_latin(sq, 10):
                linear_sqs[(c2, c5)] = sq

    # Also try to build squares via Z_5 x Z_2 with different index ordering
    # Alternate: treat i as (i5, i2) and j as (j5, j2)
    linear_sqs_alt = {}
    for c2 in range(2):
        for c5 in range(5):
            sq = []
            for i in range(10):
                i5, i2 = i % 5, i // 5
                row = []
                for j in range(10):
                    j5, j2 = j % 5, j // 5
                    # Now a_out is in Z_2, b_out in Z_5
                    a_out = (c2 * i2 + j2) % 2
                    b_out = (c5 * i5 + j5) % 5
                    row.append(_pair_to_int(a_out, b_out))
                sq.append(row)
            if _verify_latin(sq, 10):
                key = ('alt', c2, c5)
                linear_sqs_alt[key] = sq

    # Combine all candidates
    all_candidates = list(linear_sqs.values()) + list(linear_sqs_alt.values())

    # Deduplicate: two squares are equivalent if they're identical
    unique_sqs = []
    seen_patterns = set()
    for sq in all_candidates:
        pat = tuple(tuple(row) for row in sq)
        if pat not in seen_patterns:
            seen_patterns.add(pat)
            unique_sqs.append(sq)

    # Greedy: find largest mutually orthogonal set
    best_set = []
    for sq in unique_sqs:
        ok = True
        for sq2 in best_set:
            if not _verify_orthogonal(sq, sq2, 10):
                ok = False
                break
        if ok:
            best_set.append(sq)

    # Try to improve: use backtracking to find a larger set
    if len(best_set) < len(unique_sqs):
        # Try all subsets of size 3, 4, ... to see if we can do better
        # For small n this is feasible
        def backtrack(idx, current_set, remaining):
            if time.time() > deadline - 5:
                return current_set
            if len(current_set) >= len(best_set):
                if len(current_set) > len(best_set):
                    best_set[:] = current_set[:]
            for i in range(idx, len(unique_sqs)):
                sq = unique_sqs[i]
                ok = True
                for sq2 in current_set:
                    if not _verify_orthogonal(sq, sq2, 10):
                        ok = False
                        break
                if ok:
                    current_set.append(sq)
                    backtrack(i + 1, current_set, remaining - 1)
                    current_set.pop()
                    if len(current_set) >= 9:  # max possible is 9
                        break

        if time.time() < deadline - 10:
            backtrack(0, [], len(unique_sqs))

    # Ensure we have at least 1 square
    if len(best_set) == 0:
        best_set = [unique_sqs[0]] if unique_sqs else [[(i + j) % 10 for j in range(10)] for i in range(10)]

    # Convert to pure Python lists of ints
    result = [[ [int(x) for x in row] for row in sq] for sq in best_set]
    return result
