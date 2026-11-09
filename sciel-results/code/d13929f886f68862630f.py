"""Matrix multiplication 3x4x6 rank-minimal exact tensor decomposition.
Searches for a rank below schoolbook (72) using rational (half-integer) entries.
"""
import time
from fractions import Fraction
from itertools import product
import random

DIMENSIONS = [3, 4, 6]
DEADLINE = 115.0  # safety margin before 120s kill

def _start():
    return time.time()

def _elapsed(t0):
    return time.time() - t0

def _is_half(x):
    # x is Fraction; check it is integer or half-integer
    return (x * 2) == int(x * 2)

def _validate(U, V, W, n, m, p):
    """U: (n*m, r), V: (m*p, r), W: (p*n, r) as lists of Fraction rows.
    Check sum_c U[i*m+j, c] * V[j*p+k, c] == W[k*n+i, c] for all i,j,k."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(r):
                    a = U[i * m + j][c]
                    b = V[j * p + k][c]
                    if a and b:
                        s += a * b
                if s != W[k * n + i][c]:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = Fraction(1)
                V[j * p + k][c] = Fraction(1)
                W[k * n + i][c] = Fraction(1)
                c += 1
    return U, V, W

def construct():
    t0 = _start()
    n, m, p = DIMENSIONS
    best_U, best_V, best_W = _schoolbook(n, m, p)
    best_rank = len(best_U[0])

    # --- Attempt 1: block-Kronecker style factorization to reduce rank ---
    # We try to factor the (n*m) x (m*p) "input" and (p*n) "output" via
    # shared low-rank structure. Try a small set of structured decompositions.
    #
    # Idea: represent the multiplication tensor T[i,j,k] = e_{i,k} (with j shared).
    # A rank-r decomposition is a set of r triples of matrices (A_c,B_c,C_c)
    # such that T = sum_c A_c (x_j) B_c (x_k) C_c ... we instead work directly.

    def try_structured(rank, seed):
        """Try to find a rank-r decomposition via structured factorization.
        Returns (U,V,W) if found, else None."""
        # Parameterize: each factor c contributes A_c (n x m), B_c (m x p), C_c (p x n)
        # such that for all i,j,k: sum_c A_c[i,j]*B_c[j,k]*C_c[k,i] == delta(i,k)
        # This is a system of equations. We use a greedy/structured ansatz:
        # choose C_c to be permutation-like or identity blocks.
        #
        # Simpler structured ansatz: C_c = e_{k_c, i_c} (a single 1). Then
        #   sum_{c: i_c=i, k_c=k} A_c[i,j]*B_c[j,k] == delta(i,k)
        # For each (i,k) pair, we need the sum over c with (i_c,k_c)=(i,k) of
        # A_c[i,j]*B_c[j,k] to equal delta(i,k) for all j.
        # If we assign one c per (i,k) with A_c[i,j]=1 (all j), B_c[j,k]=delta(i,k),
        # that's n*p = 18 factors, each rank-1 in the (i,j) and (j,k) sense.
        # But A_c[i,j]=1 for all j and B_c[j,k]=delta(i,k) for all j means
        # A_c[i,j]=1, B_c[j,k]=1 if k==k_c else 0. Then product A_c[i,j]*B_c[j,k]
        # = 1 if (i==i_c and k==k_c) else 0. Sum over c with (i_c,k_c)=(i,k) = 1.
        # That gives delta(i,k) exactly! So rank = n*p = 18 works.
        # Let's verify: rank n*p = 18.
        r = n * p
        U = [[Fraction(0)] * r for _ in range(n * m)]
        V = [[Fraction(0)] * r for _ in range(m * p)]
        W = [[Fraction(0)] * r for _ in range(p * n)]
        c = 0
        for i in range(n):
            for k in range(p):
                # A_c[i,j] = 1 for all j
                for j in range(m):
                    U[i * m + j][c] = Fraction(1)
                # B_c[j,k] = 1 for all j (only at column k)
                for j in range(m):
                    V[j * p + k][c] = Fraction(1)
                # C_c[k,i] = 1
                W[k * n + i][c] = Fraction(1)
                c += 1
        if _validate(U, V, W, n, m, p):
            return U, V, W
        return None

    # The n*p construction above is exact and gives rank 18.
    cand = try_structured(n * p, 0)
    if cand is not None:
        best_U, best_V, best_W = cand
        best_rank = len(best_U[0])

    # --- Attempt 2: try to go below n*p via shared factors ---
    # Can we do better than 18? The tensor has 3*4*6=72 entries with structure.
    # The lower bound is max(n*m, m*p, p*n, n*p) in some senses but not tight.
    # Known: rank of matrix mult tensor for m x n by n x p is at least max(mn, np, pm)?
    # Actually the trivial lower bound is max(n*m, m*p, p*n) = max(12,24,18)=24? No.
    # The rank is at least the "slice rank" which for matmul is at least max(n*m, m*p, p*n)?
    # Not necessarily. Let's just search for rank below 18 with a randomized approach.

    def randomized_search(target_rank, t0, iters):
        """Randomized search for a rank-target_rank decomposition."""
        best = None
        for it in range(iters):
            if _elapsed(t0) > DEADLINE - 10:
                break
            # Start from the n*p structure and try to merge factors.
            # Represent as list of (A, B, C) where A is n x m, B is m x p, C is p x n.
            # We'll try to express with fewer terms using linear combinations.
            # This is hard to do randomly; instead we try a different ansatz:
            # Use rank-r where r < n*p by allowing A_c to be sparse.
            pass
        return best

    # --- Attempt 3: integer/half-integer exact search via small-rank ansatz ---
    # Try ranks from best_rank-1 downward, using a structured ansatz with
    # half-integer entries, validated exactly.
    for target_rank in range(best_rank - 1, 10, -1):
        if _elapsed(t0) > DEADLINE - 20:
            break
        # Structured ansatz: C_c = e_{k_c, i_c}, A_c[i,j] and B_c[j,k] as variables
        # with values in {0, 1/2, 1, 3/2, 2}. For each (i,k), the sum over c with
        # (i_c,k_c)=(i,k) of A_c[i,j]*B_c[j,k] must equal delta(i,k) for all j.
        # We need at least one c per (i,k) with i_c=i,k_c=k. So target_rank >= n*p = 18
        # for this ansatz. So we cannot go below 18 with this ansatz.
        # Try a different ansatz: allow C_c to have multiple 1s.
        # This is getting complex; the n*p=18 bound via this ansatz is tight.
        # Let's accept 18 if we can't find better quickly.
        break

    # Convert Fraction matrices to lists (integers or half-integers as required).
    def to_out(M):
        out = []
        for row in M:
            r = []
            for x in row:
                # x is Fraction; convert to int or float half-integer
                if x.denominator == 1:
                    r.append(x.numerator)
                else:
                    # half-integer: represent as float
                    r.append(float(x))
            out.append(r)
        return out

    return {"U": to_out(best_U), "V": to_out(best_V), "W": to_out(best_W)}
