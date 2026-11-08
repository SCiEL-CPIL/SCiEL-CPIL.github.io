"""
Search for a rank-minimal exact tensor decomposition for 4x5 x 5x6 matrix multiplication.

Uses a time-limited local search with integer/half-integer entries to try to find
a rank lower than the schoolbook rank of 120.
"""

import time
import random
from fractions import Fraction

DIMENSIONS = [4, 5, 6]
N, M, P = DIMENSIONS
SCHOOLBOOK_RANK = N * M * P
DEADLINE = 135.0  # 15s safety margin before the 150s limit
START_TIME = time.time()


def _elapsed():
    return time.time() - START_TIME


def _deadline_exceeded():
    return _elapsed() > DEADLINE


def _build_schoolbook():
    """Construct the standard schoolbook rank-120 decomposition."""
    rank = SCHOOLBOOK_RANK
    U = [[0] * rank for _ in range(N * M)]
    V = [[0] * rank for _ in range(M * P)]
    W = [[0] * rank for _ in range(P * N)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][col] = Fraction(1)
                V[j * P + k][col] = Fraction(1)
                W[k * N + i][col] = Fraction(1)
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W):
    """
    Check whether the tensor decomposition is exact.
    U: (N*M, rank), V: (M*P, rank), W: (P*N, rank)
    The multiplication tensor T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    must equal 1 if (i,j,k) corresponds to a valid triple (i in 0..N-1, j in 0..M-1, k in 0..P-1),
    and the convention is W[k*N+i, c].
    """
    rank = len(U[0])
    for i in range(N):
        for j in range(M):
            for k in range(P):
                total = Fraction(0)
                for c in range(rank):
                    total += U[i * M + j][c] * V[j * P + k][c] * W[k * N + i][c]
                if total != 1:
                    return False
    return True


def _random_half_integer():
    """Generate a random half-integer in [-2, 2]."""
    choices = [Fraction(-2), Fraction(-3, 2), Fraction(-1), Fraction(-1, 2), Fraction(0),
               Fraction(1, 2), Fraction(1), Fraction(3, 2), Fraction(2)]
    return random.choice(choices)


def _local_search(start_rank, max_iterations=5000):
    """
    Try to find a valid decomposition of the given rank using local search.
    Start with random half-integer entries and try to fix errors.
    """
    rank = start_rank
    best = None

    for attempt in range(3):
        if _deadline_exceeded():
            return best

        U = [[_random_half_integer() for _ in range(rank)] for _ in range(N * M)]
        V = [[_random_half_integer() for _ in range(rank)] for _ in range(M * P)]
        W = [[_random_half_integer() for _ in range(rank)] for _ in range(P * N)]

        # Try to fix: for each (i,j,k), we need sum_c U[i*M+j][c]*V[j*P+k][c]*W[k*N+i][c] == 1
        # This is very hard. Let's use a different approach:
        # Start with schoolbook and try to reduce rank by zeroing out some columns
        # and adjusting others.

        if attempt == 0:
            # Start from schoolbook, try to zero out one column at a time
            U = [[0] * rank for _ in range(N * M)]
            V = [[0] * rank for _ in range(M * P)]
            W = [[0] * rank for _ in range(P * N)]
            # Fill first 'rank' columns with schoolbook pattern (if rank == 120, all columns)
            col = 0
            for i in range(N):
                for j in range(M):
                    for k in range(P):
                        if col < rank:
                            U[i * M + j][col] = Fraction(1)
                            V[j * P + k][col] = Fraction(1)
                            W[k * N + i][col] = Fraction(1)
                        col += 1

        # Local search: try to flip signs or zero out entries
        for it in range(max_iterations):
            if _deadline_exceeded():
                return best
            if _verify(U, V, W):
                best = {"U": U, "V": V, "W": W}
                return best

            # Random perturbation
            if random.random() < 0.5:
                # Zero out a random entry
                mat = random.choice([U, V, W])
                rows = N * M if mat is U else (M * P if mat is V else P * N)
                r = random.randint(0, rows - 1)
                c = random.randint(0, rank - 1)
                mat[r][c] = Fraction(0)
            else:
                # Flip sign of a random entry
                mat = random.choice([U, V, W])
                rows = N * M if mat is U else (M * P if mat is V else P * N)
                r = random.randint(0, rows - 1)
                c = random.randint(0, rank - 1)
                mat[r][c] = -mat[r][c]

    return best


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x5 x 5x6 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    global START_TIME
    START_TIME = time.time()

    # Start with schoolbook
    best = _build_schoolbook()
    best_rank = SCHOOLBOOK_RANK

    # Try ranks from 119 down to 110 (limited search)
    for rank in range(best_rank - 1, 109, -1):
        if _deadline_exceeded():
            break
        result = _local_search(rank, max_iterations=2000)
        if result is not None:
            best = result
            best_rank = rank
            break  # Found a lower rank, stop

    # Convert Fractions to a format the evaluator accepts
    # Use float for half-integers, int for integers
    def convert(mat):
        out = []
        for row in mat:
            new_row = []
            for val in row:
                if val.denominator == 1:
                    new_row.append(val.numerator)
                else:
                    new_row.append(float(val))
            out.append(new_row)
        return out

    return {"U": convert(best["U"]), "V": convert(best["V"]), "W": convert(best["W"])}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
