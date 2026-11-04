"""Rank-minimal exact tensor decomposition for 2x4 x 4x4 matrix multiplication."""
import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [2, 4, 4]
START = time.time()
DEADLINE = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2,4,4
    # Bilinear rank lower bound: max(n*p, m*p, n*m)
    rank_lb = max(n * p, m * p, n * m)  # max(8,16,8)=16

    # Precompute the multiplication tensor T[i,j,k] = 1 for the (i,j,k) entry.
    # We need U (n*m x r), V (m*p x r), W (p*n x r) such that
    #   T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # with entries in {0, 1/2, 1, -1/2, -1} etc. (integer/half-integer).

    # Schoolbook (rank = n*m*p) as the guaranteed fallback.
    def schoolbook():
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    # Exact verification: check T[i,j,k] == sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    def verify(U, V, W):
        r = len(U[0])
        # Use Fraction for exactness
        Uf = [[Fraction(x) for x in row] for row in U]
        Vf = [[Fraction(x) for x in row] for row in V]
        Wf = [[Fraction(x) for x in row] for row in W]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(r):
                        s += Uf[i * m + j][c] * Vf[j * p + k][c] * Wf[k * n + i][c]
                    if s != 1:
                        return False
        # Also check that non-(i,j,k) products are 0.
        # The tensor is supported only on (i,j,k) with the given structure.
        # Actually the full tensor T[a,b,c] where a in [0,n*m), b in [0,m*p), c in [0,p*n).
        # T[a,b,c] = 1 iff a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k, else 0.
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    # determine if (a,b,c) is a valid triple
                    i = a // m
                    j = a % m
                    jj = b // p
                    k = b % p
                    kk = c // n
                    ii = c % n
                    valid = (i == ii and j == jj and k == kk)
                    s = Fraction(0)
                    for cc in range(r):
                        s += Uf[a][cc] * Vf[b][cc] * Wf[c][cc]
                    expected = Fraction(1) if valid else Fraction(0)
                    if s != expected:
                        return False
        return True

    # Search strategy: try to find a low-rank decomposition.
    # Use a randomized/evolutionary approach with small integer/half-integer entries.
    # We'll attempt ranks from rank_lb upward.

    # Known: the bilinear rank of 2x4 x 4x4. Let's search.
    # Strategy: parameterize and use random restarts with local improvement,
    # verifying exactness. Since entries are half-integers, scale by 2 to work in integers.

    def try_find_rank(r, time_budget=8.0):
        """Try to find a rank-r decomposition. Returns (U,V,W) or None."""
        deadline = time.time() + time_budget
        # Work in doubled integers: entries in {0,1,2,-1,-2} correspond to {0,0.5,1,-0.5,-1}
        # But to keep it simple, allow a small set of values.
        vals = [0, 1, -1, 2, -2, 0.5, -0.5]

        # We need to solve: for each (a,b,c) in support, sum_c U[a,c]V[b,c]W[c,c'] = 1
        # and for all others = 0.
        # This is a system of polynomial equations. Use random init + gradient-free search.

        # Simpler: use the structure. Try to build a decomposition by assigning
        # random small values and then solving linear constraints per column.
        # Actually, a cleaner approach: fix U and V randomly, solve for W linearly.
        # For each column c, W[:,c] must satisfy: for all (a,b) with (a,b,c) in support...
        # This is underdetermined in general.

        # Let's use a direct search: random small-integer assignments, verify.
        # To make it efficient, use the fact that the support has n*m*p = 32 ones.

        for attempt in range(200000):
            if time.time() > deadline:
                return None
            # Random assignment of small values
            U = [[random.choice(vals) for _ in range(r)] for _ in range(n * m)]
            V = [[random.choice(vals) for _ in range(r)] for _ in range(m * p)]
            W = [[random.choice(vals) for _ in range(r)] for _ in range(p * n)]
            if verify(U, V, W):
                return U, V, W
        return None

    # Try increasing ranks
    for r in range(rank_lb, best_rank + 1):
        if time.time() > DEADLINE - 5:
            break
        result = try_find_rank(r, time_budget=min(6.0, max(2.0, (DEADLINE - 5 - time.time()))))
        if result is not None:
            U, V, W = result
            if verify(U, V, W):
                best = {"U": U, "V": V, "W": W}
                best_rank = r
                break  # found minimal rank

    return best
# EVOLVE-BLOCK-END
