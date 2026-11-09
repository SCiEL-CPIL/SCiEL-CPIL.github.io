"""Combined program: product-construction base + backtracking CSP extension."""
# EVOLVE-BLOCK-START
import time

P = {'n': 10}


def _is_latin(sq):
    n = len(sq)
    for i in range(n):
        if sorted(sq[i]) != list(range(n)):
            return False
    for j in range(n):
        col = [sq[i][j] for i in range(n)]
        if sorted(col) != list(range(n)):
            return False
    return True


def _are_orthogonal(sq1, sq2):
    n = len(sq1)
    pairs = set()
    for i in range(n):
        for j in range(n):
            pair = (sq1[i][j], sq2[i][j])
            if pair in pairs:
                return False
            pairs.add(pair)
    return True


def _is_orth_to_all(sq, existing):
    for ex in existing:
        if not _are_orthogonal(sq, ex):
            return False
    return True


def _build_product_mols():
    """Build MOLS of order 10 via product construction from MOLS of order 5.
    Returns up to 4 mutually orthogonal Latin squares of order 10."""
    n = 10
    m = 5
    p = 2
    base_mols = []
    for k in range(m - 1):
        sq = [[(i + (k + 1) * j) % m for j in range(m)] for i in range(m)]
        base_mols.append(sq)
    lifted = []
    for sq in base_mols:
        new_sq = []
        for i in range(n):
            row = []
            i1 = i // m
            i2 = i % m
            for j in range(n):
                j1 = j // m
                j2 = j % m
                val = p * sq[i2][j2] + ((i1 + j1) % p)
                row.append(val)
            new_sq.append(row)
        lifted.append(new_sq)
    return lifted


def _backtrack_build(squares, deadline):
    """Extend the list of mutually orthogonal squares by backtracking over cells
    of a new Latin square, enforcing row/column uniqueness and a global set of
    forbidden (a,b) symbol pairs (orthogonality with every existing square)."""
    n = 10

    def _is_latin_partial(sq, r, c):
        # sq[r][c] is the value just placed
        v = sq[r][c]
        # row r: v appears exactly once
        row = sq[r]
        if row.count(v) != 1:
            return False
        # column c: v appears exactly once
        col = [sq[i][c] for i in range(n)]
        if col.count(v) != 1:
            return False
        return True

    def _candidates_for(sq, r, c, row_sets, col_sets, forbidden):
        used = row_sets[r] | col_sets[c]
        cands = []
        for v in range(n):
            if v in used:
                continue
            ok = True
            for sq2 in squares:
                if (sq2[r][c], v) in forbidden:
                    ok = False
                    break
            if ok:
                cands.append(v)
        return cands

    def _solve(sq, r, c, row_sets, col_sets, forbidden):
        if time.time() > deadline:
            return False
        if r == n - 1 and c == n - 1:
            # last cell: only one value can fit
            used = row_sets[r] | col_sets[c]
            cands = []
            for v in range(n):
                if v in used:
                    continue
                ok = True
                for sq2 in squares:
                    if (sq2[r][c], v) in forbidden:
                        ok = False
                        break
                if ok:
                    cands.append(v)
            if len(cands) != 1:
                return False
            v = cands[0]
            sq[r][c] = v
            return True
        if c + 1 == n:
            nr, nc = r + 1, 0
        else:
            nr, nc = r, c + 1
        cands = _candidates_for(sq, r, c, row_sets, col_sets, forbidden)
        if not cands:
            return False
        # Try candidates; prefer values that appear less often in this column
        # to prune faster (heuristic)
        for v in cands:
            sq[r][c] = v
            row_sets[r].add(v)
            col_sets[c].add(v)
            if _is_latin_partial(sq, r, c):
                if _solve(sq, nr, nc, row_sets, col_sets, forbidden):
                    return True
            col_sets[c].discard(v)
            row_sets[r].discard(v)
            sq[r][c] = -1
        return False

    # Build a new square
    sq = [[-1 for _ in range(n)] for _ in range(n)]
    row_sets = [set() for _ in range(n)]
    col_sets = [set() for _ in range(n)]
    # Precompute forbidden pairs: (existing_val, new_val) for each cell
    forbidden = set()
    for sq2 in squares:
        for i in range(n):
            for j in range(n):
                forbidden.add((sq2[i][j], sq2[i][j]))  # placeholder; we need (existing, new)
    # Actually forbidden should be: for each existing square, the set of (existing_val_at_cell, new_val)
    # that we must avoid. Since new_val must differ in pair, forbidden = {(sq2[i][j], x) for all x}
    # but that's everything. We want: the pair (sq2[i][j], new_val) must be unique across cells.
    # So for a given existing square, at cell (i,j), the pair is (sq2[i][j], new_val).
    # This pair must not equal (sq2[i'][j'], new_val') for any other cell.
    # Equivalently: for a fixed existing square, the mapping (i,j) -> (sq2[i][j], new_val[i][j])
    # must be injective. Since sq2 is Latin, sq2[i][j] is unique per row/col but not per cell.
    # The constraint is: no two cells have the same (sq2_val, new_val) pair.
    # For a new square, given existing square sq2, the constraint is:
    #   for all cells (i,j), (i',j') with (i,j) != (i',j'):
    #   NOT (sq2[i][j] == sq2[i'][j'] AND new_val[i][j] == new_val[i'][j'])
    # This is equivalent to: new_val must be a Latin square AND the pairs must be distinct.
    # For a CSP, we track: for each existing square, which (existing_val, new_val) pairs have been used.
    # When placing new_val at (i,j), we must ensure (sq2[i][j], new_val) is not already used
    # for that existing square.
    # So forbidden should be per-existing-square: a set of (existing_val, new_val) pairs already used.
    # Let me restructure: forbidden is a list of sets, one per existing square.
    forbidden = [set() for _ in squares]

    def _candidates_for2(sq, r, c, row_sets, col_sets, forbidden):
        used = row_sets[r] | col_sets[c]
        cands = []
        for v in range(n):
            if v in used:
                continue
            ok = True
            for idx, sq2 in enumerate(squares):
                if (sq2[r][c], v) in forbidden[idx]:
                    ok = False
                    break
            if ok:
                cands.append(v)
        return cands

    def _solve2(sq, r, c, row_sets, col_sets, forbidden):
        if time.time() > deadline:
            return False
        if r == n - 1 and c == n - 1:
            used = row_sets[r] | col_sets[c]
            cands = []
            for v in range(n):
                if v in used:
                    continue
                ok = True
                for idx, sq2 in enumerate(squares):
                    if (sq2[r][c], v) in forbidden[idx]:
                        ok = False
                        break
                if ok:
                    cands.append(v)
            if len(cands) != 1:
                return False
            v = cands[0]
            sq[r][c] = v
            return True
        if c + 1 == n:
            nr, nc = r + 1, 0
        else:
            nr, nc = r, c + 1
        cands = _candidates_for2(sq, r, c, row_sets, col_sets, forbidden)
        if not cands:
            return False
        for v in cands:
            sq[r][c] = v
            row_sets[r].add(v)
            col_sets[c].add(v)
            # Update forbidden: for each existing square, add (sq2[r][c], v)
            for idx, sq2 in enumerate(squares):
                forbidden[idx].add((sq2[r][c], v))
            if _is_latin_partial(sq, r, c):
                if _solve2(sq, nr, nc, row_sets, col_sets, forbidden):
                    return True
            for idx, sq2 in enumerate(squares):
                forbidden[idx].discard((sq2[r][c], v))
            col_sets[c].discard(v)
            row_sets[r].discard(v)
            sq[r][c] = -1
        return False

    if _solve2(sq, 0, 0, row_sets, col_sets, forbidden):
        # Validate
        if _is_latin(sq) and _is_orth_to_all(sq, squares):
            squares.append(sq)
            return True
    return False


def construct():
    n = 10
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    # Start with product construction base (4 MOLS)
    squares = _build_product_mols()

    # Verify base
    valid = []
    for sq in squares:
        if _is_latin(sq):
            if _is_orth_to_all(sq, valid):
                valid.append(sq)
    squares = valid

    # Try to extend via backtracking
    while len(squares) < 9 and time.time() < deadline:
        if not _backtrack_build(squares, deadline):
            break

    # Fallback: if we have fewer than 1, add a standard cyclic square
    if not squares:
        s1 = [[(i + j) % n for j in range(n)] for i in range(n)]
        squares = [s1]

    # Convert to pure Python int lists
    result = []
    for sq in squares:
        result.append([[int(sq[i][j]) for j in range(n)] for i in range(n)])

    return result
# EVOLVE-BLOCK-END
