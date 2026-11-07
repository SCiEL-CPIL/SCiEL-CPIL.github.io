import time
import random

def construct():
    n = 10
    deadline = time.time() + 55  # safety margin before the 90s kill

    # ---- Deterministic construction of 2 MOLS via GF(5) x GF(2) ----
    def encode(a, b):
        return a * 2 + b

    def decode(x):
        return x // 2, x % 2

    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]

    L2 = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        ai, bi = decode(i)
        for j in range(n):
            aj, bj = decode(j)
            a_part = (ai + 2 * aj) % 5
            b_part = (bi + bj) % 2
            L2[i][j] = encode(a_part, b_part)

    def is_latin(sq):
        for i in range(n):
            if sorted(sq[i]) != list(range(n)):
                return False
        for j in range(n):
            if sorted([sq[i][j] for i in range(n)]) != list(range(n)):
                return False
        return True

    def are_orthogonal(sq1, sq2):
        pairs = set()
        for i in range(n):
            for j in range(n):
                p = (sq1[i][j], sq2[i][j])
                if p in pairs:
                    return False
                pairs.add(p)
        return True

    if not (is_latin(L1) and is_latin(L2) and are_orthogonal(L1, L2)):
        # Should not happen; fall back to a single cyclic square.
        return [[[int((i + j) % n) for j in range(n)] for i in range(n)]]

    base = [L1, L2]
    base_int = [[[int(L1[i][j]) for j in range(n)] for i in range(n)],
                [[int(L2[i][j]) for j in range(n)] for i in range(n)]]

    # ---- Try to find a 3rd square orthogonal to both, via backtracking ----
    # Constraints for L3:
    #   - each row/col is a permutation of 0..9
    #   - pairs (L1[i][j], L3[i][j]) all distinct
    #   - pairs (L2[i][j], L3[i][j]) all distinct
    #   (the L1/L2 pair constraint is already satisfied)

    def search_third(base_sq, rng):
        L3 = [[-1] * n for _ in range(n)]
        row_used = [set() for _ in range(n)]
        col_used = [set() for _ in range(n)]
        pair1 = set()  # (L1[i][j], val)
        pair2 = set()  # (L2[i][j], val)

        # Precompute per-cell candidate values (prune by the two pair constraints
        # against values that would collide) -- simple check is enough.
        def valid(i, j, val):
            if val in row_used[i]:
                return False
            if val in col_used[j]:
                return False
            if (base_sq[0][i][j], val) in pair1:
                return False
            if (base_sq[1][i][j], val) in pair2:
                return False
            return True

        # Fill order: row-major, but choose the most-constrained cell first.
        # For speed, use a static MRV-ish ordering: iterate cells, pick smallest
        # candidate count each step.
        cells = [(i, j) for i in range(n) for j in range(n)]

        def backtrack(remaining):
            if time.time() > deadline:
                return False
            if not remaining:
                return True

            # MRV: pick the cell with fewest valid values.
            best_k = -1
            best_vals = None
            best_cnt = 11
            for k, (i, j) in enumerate(remaining):
                vals = [v for v in range(n) if valid(i, j, v)]
                c = len(vals)
                if c == 0:
                    return False
                if c < best_cnt:
                    best_cnt = c
                    best_k = k
                    best_vals = vals
                    if c == 1:
                        break

            i, j = remaining[best_k]
            new_remaining = remaining[:best_k] + remaining[best_k + 1:]
            rng.shuffle(best_vals)
            for val in best_vals:
                L3[i][j] = val
                row_used[i].add(val)
                col_used[j].add(val)
                pair1.add((base_sq[0][i][j], val))
                pair2.add((base_sq[1][i][j], val))
                if backtrack(new_remaining):
                    return True
                row_used[i].remove(val)
                col_used[j].remove(val)
                pair1.remove((base_sq[0][i][j], val))
                pair2.remove((base_sq[1][i][j], val))
                L3[i][j] = -1
            return False

        if backtrack(cells):
            return L3
        return None

    # A few random restarts with different value-ordering, bounded by deadline.
    found = None
    for attempt in range(40):
        if time.time() > deadline:
            break
        rng = random.Random(12345 + attempt * 7)
        res = search_third(base, rng)
        if res is not None:
            found = res
            break

    if found is not None:
        # Verify before returning.
        if is_latin(found) and are_orthogonal(L1, found) and are_orthogonal(L2, found):
            base_int.append([[int(found[i][j]) for j in range(n)] for i in range(n)])

    # Return between 1 and 9 squares, all Python int, all lists.
    return base_int
