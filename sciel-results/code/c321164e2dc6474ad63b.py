"""Matrix multiplication 4x5 by 5x6 — structured decomposition via Strassen blocks.

Decomposes 4x5x6 as:
  4x4x4 (Strassen, rank 56) + 4x4x2 (Strassen, rank 28) + 4x1x6 (schoolbook, rank 24)
  Total initial rank = 108, then prune redundant columns.
"""
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

def _time_left(deadline):
    return deadline - time.time()


def construct():
    deadline = time.time() + 138
    n, m, p = DIMENSIONS  # 4, 5, 6

    # Strassen 2x2x2 terms: (u[(i,j)], v[(j,k)], w[(k,i)])
    strassen_2x2_terms = [
        ({(0,0):1,(0,1):1,(1,0):1,(1,1):1},
         {(0,0):1,(0,1):1,(1,0):1,(1,1):1},
         {(0,0):1,(0,1):1,(1,0):1,(1,1):1}),
        ({(0,0):1,(0,1):1,(1,0):-1,(1,1):1},
         {(0,0):1,(0,1):-1,(1,0):1,(1,1):1},
         {(0,0):1,(0,1):1,(1,0):1,(1,1):0}),
        ({(0,0):1,(0,1):-1,(1,0):1,(1,1):1},
         {(0,0):1,(0,1):1,(1,0):1,(1,1):-1},
         {(0,0):0,(0,1):1,(1,0):1,(1,1):1}),
        ({(0,0):1,(0,1):0,(1,0):0,(1,1):1},
         {(0,0):1,(0,1):0,(1,0):0,(1,1):1},
         {(0,0):1,(0,1):0,(1,0):0,(1,1):1}),
        ({(0,0):0,(0,1):1,(1,0):1,(1,1):0},
         {(0,0):0,(0,1):1,(1,0):1,(1,1):0},
         {(0,0):1,(0,1):0,(1,0):0,(1,1):1}),
        ({(0,0):1,(0,1):1,(1,0):0,(1,1):0},
         {(0,0):1,(0,1):-1,(1,0):0,(1,1):0},
         {(0,0):0,(0,1):1,(1,0):0,(1,1):0}),
        ({(0,0):0,(0,1):0,(1,0):1,(1,1):1},
         {(0,0):0,(0,1):0,(1,0):1,(1,1):-1},
         {(0,0):0,(0,1):0,(1,0):1,(1,1):0}),
    ]

    U_full = {}
    V_full = {}
    W_full = {}
    next_term = 0

    # Part 1: 4x4x4 Strassen (rows 0-3, inner 0-3, cols 0-3)
    for i0 in range(2):
        for j0 in range(2):
            for k0 in range(2):
                for t, (u, v, w) in enumerate(strassen_2x2_terms):
                    for (i, j), val in u.items():
                        U_full.setdefault((i0*2+i, j0*2+j), {})[next_term] = Fraction(val)
                    for (j, k), val in v.items():
                        V_full.setdefault((j0*2+j, k0*2+k), {})[next_term] = Fraction(val)
                    for (k, i), val in w.items():
                        W_full.setdefault((k0*2+k, i0*2+i), {})[next_term] = Fraction(val)
                    next_term += 1

    # Part 2: 4x4x2 Strassen (rows 0-3, inner 0-3, cols 4-5)
    for i0 in range(2):
        for j0 in range(2):
            for t, (u, v, w) in enumerate(strassen_2x2_terms):
                for (i, j), val in u.items():
                    U_full.setdefault((i0*2+i, j0*2+j), {})[next_term] = Fraction(val)
                for (j, k), val in v.items():
                    V_full.setdefault((j0*2+j, 4+k), {})[next_term] = Fraction(val)
                for (k, i), val in w.items():
                    W_full.setdefault((4+k, i0*2+i), {})[next_term] = Fraction(val)
                next_term += 1

    # Part 3: 4x1x6 schoolbook (rows 0-3, inner 4, cols 0-5)
    for i in range(4):
        for k in range(6):
            U_full.setdefault((i, 4), {})[next_term] = Fraction(1)
            V_full.setdefault((4, k), {})[next_term] = Fraction(1)
            W_full.setdefault((k, i), {})[next_term] = Fraction(1)
            next_term += 1

    rank = next_term

    # Convert to dense
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]

    for (i, j), vals in U_full.items():
        for r, val in vals.items():
            U[i * m + j][r] = val
    for (j, k), vals in V_full.items():
        for r, val in vals.items():
            V[j * p + k][r] = val
    for (k, i), vals in W_full.items():
        for r, val in vals.items():
            W[k * n + i][r] = val

    # Verify
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if total != 1:
                        return False
        return True

    if not verify(U, V, W, rank):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[Fraction(0)] * rank for _ in range(n * m)]
        V = [[Fraction(0)] * rank for _ in range(m * p)]
        W = [[Fraction(0)] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = Fraction(1)
                    V[j * p + k][col] = Fraction(1)
                    W[k * n + i][col] = Fraction(1)
                    col += 1

    # Prune redundant columns
    if _time_left(deadline) > 15:
        def try_remove_column(U, V, W, rank, col_to_remove):
            if rank <= 1:
                return False
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = Fraction(0)
                        for r in range(rank):
                            if r == col_to_remove:
                                continue
                            total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                        if total != 1:
                            return False
            return True

        for iteration in range(50):
            if _time_left(deadline) < 10:
                break
            current_rank = len(U[0]) if U else 0
            removed = False
            for c in range(current_rank):
                if _time_left(deadline) < 5:
                    break
                if try_remove_column(U, V, W, current_rank, c):
                    for i in range(n * m):
                        U[i].pop(c)
                    for j in range(m * p):
                        V[j].pop(c)
                    for k in range(p * n):
                        W[k].pop(c)
                    removed = True
                    break
            if not removed:
                break

    def frac_to_val(f):
        if f.denominator == 1:
            return int(f.numerator)
        elif f.denominator == 2:
            return f.numerator / 2.0
        else:
            return float(f)

    U_out = [[frac_to_val(val) for val in row] for row in U]
    V_out = [[frac_to_val(val) for val in row] for row in V]
    W_out = [[frac_to_val(val) for val in row] for row in W]

    return {"U": U_out, "V": V_out, "W": W_out}
