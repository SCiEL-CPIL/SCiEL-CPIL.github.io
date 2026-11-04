"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0  # safety margin before 150s wall-clock kill
    n, m, p = DIMENSIONS
    N, M, P = n * m, m * p, p * n

    # Schoolbook rank-1 terms: (i,j,k)
    base = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    rank0 = len(base)

    def to_uv(t):
        i, j, k = t
        u = [0] * N
        v = [0] * M
        u[i * m + j] = 1
        v[j * p + k] = 1
        return u, v

    def verify(terms):
        # Build the 3-way tensor from terms and compare to the multiplication tensor.
        T = [[0] * P for _ in range(N)]
        for (i, j, k), s in terms:
            a = i * m + j
            b = j * p + k
            c = k * n + i
            if s == 1:
                T[a][b] += 1
            elif s == -1:
                T[a][b] -= 1
            elif s == 2:
                T[a][b] += 2
            elif s == -2:
                T[a][b] -= 2
            elif s == Fraction(1, 2):
                T[a][b] += Fraction(1, 2)
            elif s == Fraction(-1, 2):
                T[a][b] -= Fraction(1, 2)
            elif s == Fraction(1, 4):
                T[a][b] += Fraction(1, 4)
            elif s == Fraction(-1, 4):
                T[a][b] -= Fraction(1, 4)
            elif s == Fraction(1, 8):
                T[a][b] += Fraction(1, 8)
            elif s == Fraction(-1, 8):
                T[a][b] -= Fraction(1, 8)
            else:
                T[a][b] += s
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    if T[a][b] != (1 if c == k * n + i else 0):
                        return False
        return True

    def build_uv(terms):
        U = [[0] * len(terms) for _ in range(N)]
        V = [[0] * len(terms) for _ in range(M)]
        W = [[0] * len(terms) for _ in range(P)]
        for c, ((i, j, k), s) in enumerate(terms):
            U[i * m + j][c] = s
            V[j * p + k][c] = 1
            W[k * n + i][c] = 1
        return U, V, W

    # --- Greedy pairwise merge search ---
    def greedy(terms, seed):
        rng = random.Random(seed)
        terms = list(terms)
        # index of each term's (i,j,k)
        while True:
            if time.time() > deadline:
                return terms
            # find a merge: pair (a,b) with signs (+,-) such that combined tensor is exact
            found = False
            order = list(range(len(terms)))
            rng.shuffle(order)
            # try all pairs in shuffled order (bounded by time)
            for xa in order:
                if time.time() > deadline:
                    return terms
                ta = terms[xa]
                for xb in order:
                    if xb == xa:
                        continue
                    tb = terms[xb]
                    sa, sb = ta[1], tb[1]
                    if sa == sb:
                        continue  # only opposite signs can cancel
                    # try s = sa (keep ta's magnitude), -sa, and possibly half
                    for s in (sa, -sa):
                        # new term: keep the (i,j,k) of the surviving sign's larger-magnitude? 
                        # We'll keep ta's (i,j,k) with sign s, and drop both a and b.
                        # But the surviving term must be one that still contributes.
                        # Strategy: new term uses ta's (i,j,k) with sign s; we remove a and b.
                        # Check: does replacing {a,b} with {(i,j,k)_a, s} keep tensor exact?
                        new_terms = []
                        for idx, t in enumerate(terms):
                            if idx == xa or idx == xb:
                                continue
                            new_terms.append(t)
                        new_terms.append((ta[0], s))
                        if verify(new_terms):
                            terms = new_terms
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                return terms

    # Run greedy from several randomizations
    best = base
    for seed in range(200):
        cur = greedy(base, seed)
        if len(cur) < len(best):
            best = cur
        if time.time() > deadline:
            break

    U, V, W = build_uv(best)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
