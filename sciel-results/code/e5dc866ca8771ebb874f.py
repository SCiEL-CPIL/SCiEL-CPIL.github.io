"""Rank-minimal exact tensor decomposition search for 3x3 by 3x4 matrix multiplication."""
import time
import random
import fractions
import numpy as np

DIMENSIONS = [3, 3, 4]
Frac = fractions.Fraction


def _validate(U, V, W, n, m, p, rank):
    """Exact check: sum_c U[i*m+j,c]*V[l*p+k,c]*W[k*n+i,c] == delta_{j,l}"""
    Uf = [[Frac(U[i][c]) for c in range(rank)] for i in range(n * m)]
    Vf = [[Frac(V[j][c]) for c in range(rank)] for j in range(m * p)]
    Wf = [[Frac(W[k][c]) for c in range(rank)] for k in range(p * n)]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for l in range(m):
                    s = Frac(0)
                    for c in range(rank):
                        s += Uf[i * m + j][c] * Vf[l * p + k][c] * Wf[k * n + i][c]
                    expected = Frac(1) if j == l else Frac(0)
                    if s != expected:
                        return False
    return True


def _check_half_integer(W, rank):
    """Check all W entries are integer or half-integer."""
    for row in W:
        for x in row:
            if isinstance(x, Frac):
                if x.denominator not in (1, 2):
                    return False
            elif isinstance(x, (int, float)):
                if isinstance(x, float) and x != int(x) and 2 * x != int(2 * x):
                    return False
    return True


def _solve_w_system_exact(U, V, i, k, n, m, p, rank):
    """Solve 9 equations (j,l pairs) for W[k*n+i]. Returns list of Fractions or None."""
    rows = []
    rhs = []
    for j in range(m):
        for l in range(m):
            row = [Frac(U[i * m + j][c]) * Frac(V[l * p + k][c]) for c in range(rank)]
            rows.append(row)
            rhs.append(Frac(1) if j == l else Frac(0))

    R = len(rows)
    C = rank
    aug = [rows[r][:] + [rhs[r]] for r in range(R)]

    pivots = []
    col = 0
    for r in range(R):
        if col >= C:
            break
        pr = None
        for rr in range(r, R):
            if aug[rr][col] != 0:
                pr = rr
                break
        if pr is None:
            col += 1
            continue
        if pr != r:
            aug[r], aug[pr] = aug[pr], aug[r]
        pivots.append((r, col))
        pv = aug[r][col]
        if pv != 1:
            aug[r] = [x / pv for x in aug[r]]
        for rr in range(R):
            if rr != r and aug[rr][col] != 0:
                f = aug[rr][col]
                aug[rr] = [a - f * b for a, b in zip(aug[rr], aug[r])]
        col += 1

    for rr in range(R):
        coeff_zero = all(aug[rr][c] == 0 for c in range(C))
        if coeff_zero and aug[rr][C] != 0:
            return None

    Wvec = [Frac(0)] * C
    for (r, c) in pivots:
        Wvec[c] = aug[r][C]

    return Wvec


def _fast_consistency_check(U, V, i, k, n, m, p, rank):
    """Fast float check if the 9x rank system is likely consistent."""
    rows = np.zeros((m * m, rank))
    rhs = np.zeros(m * m)
    idx = 0
    for j in range(m):
        for l in range(m):
            for c in range(rank):
                rows[idx, c] = U[i * m + j][c] * V[l * p + k][c]
            rhs[idx] = 1.0 if j == l else 0.0
            idx += 1
    
    # Check if rhs is in column space of rows
    try:
        rank_mat = np.linalg.matrix_rank(rows)
        aug = np.column_stack([rows, rhs])
        rank_aug = np.linalg.matrix_rank(aug)
        if rank_mat != rank_aug:
            return False
    except:
        return False
    return True


def _try_structured_search(rank, n, m, p, deadline, rng, max_iter=5000):
    """Try structured U,V patterns more likely to yield consistent W systems."""
    nm = n * m
    mp = m * p
    pn = p * n

    it = 0
    while time.time() < deadline and it < max_iter:
        it += 1

        # Pattern: use small integers with structure
        # Try different sparsity levels
        sparsity = rng.choice([0.3, 0.4, 0.5, 0.6, 0.7])
        
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]

        for idx in range(nm):
            nnz = max(1, int(rank * sparsity * rng.uniform(0.5, 1.5)))
            nnz = min(nnz, rank)
            cols = rng.sample(range(rank), nnz)
            for c in cols:
                U[idx][c] = rng.choice([-2, -1, 1, 2])

        for idx in range(mp):
            nnz = max(1, int(rank * sparsity * rng.uniform(0.5, 1.5)))
            nnz = min(nnz, rank)
            cols = rng.sample(range(rank), nnz)
            for c in cols:
                V[idx][c] = rng.choice([-2, -1, 1, 2])

        if any(all(x == 0 for x in row) for row in U):
            continue
        if any(all(x == 0 for x in row) for row in V):
            continue

        # Fast consistency check before exact solve
        consistent = True
        for i in range(n):
            for k in range(p):
                if not _fast_consistency_check(U, V, i, k, n, m, p, rank):
                    consistent = False
                    break
            if not consistent:
                break

        if not consistent:
            continue

        # Exact solve for W
        W = [[0] * rank for _ in range(pn)]
        ok = True
        for i in range(n):
            for k in range(p):
                Wvec = _solve_w_system_exact(U, V, i, k, n, m, p, rank)
                if Wvec is None:
                    ok = False
                    break
                W[k * n + i] = [int(x) if x.denominator == 1 else x for x in Wvec]
            if not ok:
                break

        if not ok:
            continue

        if not _check_half_integer(W, rank):
            continue

        if _validate(U, V, W, n, m, p, rank):
            return U, V, W

    return None


def _try_block_diagonal(rank, n, m, p, deadline, rng, max_iter=3000):
    """Try block-diagonal inspired structures."""
    nm = n * m
    mp = m * p
    pn = p * n

    it = 0
    while time.time() < deadline and it < max_iter:
        it += 1

        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]

        # Assign primary column based on index
        for i in range(n):
            for j in range(m):
                idx = i * m + j
                primary = idx % rank
                U[idx][primary] = rng.choice([1, -1, 2, -2])
                if rank > 2 and rng.random() < 0.3:
                    secondary = (primary + rng.randint(1, rank - 1)) % rank
                    if secondary != primary:
                        U[idx][secondary] = rng.choice([-1, 1])

        for l in range(m):
            for k in range(p):
                idx = l * p + k
                primary = idx % rank
                V[idx][primary] = rng.choice([1, -1, 2, -2])
                if rank > 2 and rng.random() < 0.3:
                    secondary = (primary + rng.randint(1, rank - 1)) % rank
                    if secondary != primary:
                        V[idx][secondary] = rng.choice([-1, 1])

        if any(all(x == 0 for x in row) for row in U):
            continue
        if any(all(x == 0 for x in row) for row in V):
            continue

        # Fast consistency check
        consistent = True
        for i in range(n):
            for k in range(p):
                if not _fast_consistency_check(U, V, i, k, n, m, p, rank):
                    consistent = False
                    break
            if not consistent:
                break

        if not consistent:
            continue

        W = [[0] * rank for _ in range(pn)]
        ok = True
        for i in range(n):
            for k in range(p):
                Wvec = _solve_w_system_exact(U, V, i, k, n, m, p, rank)
                if Wvec is None:
                    ok = False
                    break
                W[k * n + i] = [int(x) if x.denominator == 1 else x for x in Wvec]
            if not ok:
                break

        if not ok:
            continue

        if not _check_half_integer(W, rank):
            continue

        if _validate(U, V, W, n, m, p, rank):
            return U, V, W

    return None


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _try_systematic_ranks(n, m, p, deadline):
    """Try ranks from low to high with time allocation."""
    # Lower bound: max(n*m, m*p, p*n) = max(9, 12, 12) = 12
    # But matrix mult rank is higher. Start from reasonable range.
    ranks_to_try = list(range(19, 33))

    for rank in ranks_to_try:
        if time.time() > deadline:
            break

        # Allocate time based on rank
        if rank <= 22:
            max_iter1 = 6000
            max_iter2 = 4000
        elif rank <= 26:
            max_iter1 = 4000
            max_iter2 = 2500
        else:
            max_iter1 = 2000
            max_iter2 = 1500

        # Strategy 1: Structured search
        rng = random.Random(rank * 1000 + 42)
        result = _try_structured_search(rank, n, m, p, deadline, rng, max_iter=max_iter1)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}

        # Strategy 2: Block diagonal
        rng = random.Random(rank * 7777 + 123)
        result = _try_block_diagonal(rank, n, m, p, deadline, rng, max_iter=max_iter2)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}

    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit

    # Start with schoolbook as fallback
    best_result = _schoolbook(n, m, p)

    # Try systematic search for lower rank
    try:
        result = _try_systematic_ranks(n, m, p, deadline)
        if result is not None:
            best_result = result
    except Exception:
        pass

    return best_result
