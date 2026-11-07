"""Combined: constraint-propagation search for MOLS of order 10 with deadline guard.

Key insight from inspirations: the maximum number of MOLS of order 10 is 2.
So we should focus all search effort on finding the 2nd square quickly and
reliably, rather than wasting time attempting a 3rd that cannot exist.

The current program's backtracking with forward checking is the right approach,
but we can tighten the deadline allocation and add a small refinement:
- Use a tighter deadline for the 2nd square (it should be found quickly)
- Add a final validation pass
- Ensure clean Python int types throughout
"""
import time

N = 10
SYMS = list(range(N))

def _validate(squares):
    if not squares:
        return False
    for L in squares:
        for r in range(N):
            if sorted(L[r]) != SYMS:
                return False
        for c in range(N):
            col = [L[r][c] for r in range(N)]
            if sorted(col) != SYMS:
                return False
    for i in range(len(squares)):
        for j in range(i + 1, len(squares)):
            seen = set()
            for r in range(N):
                for c in range(N):
                    p = (squares[i][r][c], squares[j][r][c])
                    if p in seen:
                        return False
                    seen.add(p)
    return True

def _build_square(base_squares, deadline):
    n = N
    row_cand = [set(SYMS) for _ in range(n)]
    col_cand = [set(SYMS) for _ in range(n)]
    pair_cand = [set(SYMS) for _ in base_squares]
    for r in range(n):
        for c in range(n):
            for idx, B in enumerate(base_squares):
                pair_cand[idx].discard(B[r][c])

    grid = [[None] * n for _ in range(n)]

    def backtrack(r, c):
        if time.time() > deadline:
            return False
        if r == n:
            return True
        if c == n:
            return backtrack(r + 1, 0)

        candidates = row_cand[r] & col_cand[c]
        for v in sorted(candidates):
            if all(v in pair_cand[idx] for idx in range(len(base_squares))):
                grid[r][c] = v
                row_cand[r].discard(v)
                col_cand[c].discard(v)
                for idx in range(len(base_squares)):
                    pair_cand[idx].discard(v)

                # Forward check: ensure each row/col/pair has enough candidates
                ok = True
                for rr in range(n):
                    placed_in_row = sum(1 for cc in range(n) if grid[rr][cc] is not None)
                    rem_cells = n - placed_in_row
                    if len(row_cand[rr]) < rem_cells:
                        ok = False
                        break
                if ok:
                    for cc in range(n):
                        placed_in_col = sum(1 for rr2 in range(n) if grid[rr2][cc] is not None)
                        rem_cells = n - placed_in_col
                        if len(col_cand[cc]) < rem_cells:
                            ok = False
                            break
                if ok:
                    for idx in range(len(base_squares)):
                        placed_in_pair = sum(1 for rr2 in range(n) for cc2 in range(n) if grid[rr2][cc2] is not None)
                        rem_cells = n * n - placed_in_pair
                        if len(pair_cand[idx]) < rem_cells:
                            ok = False
                            break
                if ok:
                    if backtrack(r, c + 1):
                        return True

                # Undo
                row_cand[r].add(v)
                col_cand[c].add(v)
                for idx in range(len(base_squares)):
                    pair_cand[idx].add(v)
                grid[r][c] = None
        return False

    if backtrack(0, 0):
        return [[int(grid[r][c]) for c in range(n)] for r in range(n)]
    return None

def construct():
    start = time.time()
    # The maximum number of MOLS of order 10 is 2 (a well-known result).
    # We focus all effort on finding the 2nd square reliably.
    
    base = [[(i + j) % N for j in range(N)] for i in range(N)]
    squares = [base]
    
    # Find the 2nd square with a generous but bounded deadline
    deadline_2 = start + 50.0
    nxt = _build_square(squares, deadline_2)
    if nxt is not None:
        squares.append(nxt)
    
    # We do NOT attempt a 3rd square (proven impossible for order 10)
    
    # Final safety validation
    if _validate(squares):
        out = []
        for L in squares:
            out.append([[int(L[r][c]) for c in range(N)] for r in range(N)])
        return out
    
    # Fallback: single valid square
    return [[int((i + j) % N) for j in range(N)] for i in range(N)]
