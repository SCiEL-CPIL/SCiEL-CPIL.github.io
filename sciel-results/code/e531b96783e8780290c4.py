from fractions import Fraction

def solve_linear_system(A, b):
    # A: list of lists of Fraction (m x n), b: list of Fraction (m,)
    # Returns x (n,) if consistent, else None
    m = len(A)
    n = len(A[0])
    # Augmented matrix
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    # Gaussian elimination
    rank = 0
    for col in range(n):
        # Find pivot
        pivot = None
        for row in range(rank, m):
            if M[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            continue
        # Swap
        M[rank], M[pivot] = M[pivot], M[rank]
        # Normalize
        piv_val = M[rank][col]
        for j in range(col, n+1):
            M[rank][j] = M[rank][j] / piv_val
        # Eliminate
        for row in range(m):
            if row != rank and M[row][col] != 0:
                factor = M[row][col]
                for j in range(col, n+1):
                    M[row][j] = M[row][j] - factor * M[rank][j]
        rank += 1
    # Check consistency: any row that is [0,0,...,0, b] with b != 0
    for row in range(m):
        all_zero = all(M[row][j] == 0 for j in range(n))
        if all_zero and M[row][n] != 0:
            return None
    # Extract solution (free variables set to 0)
    x = [Fraction(0)] * n
    for i in range(rank):
        # find the pivot column for row i
        for col in range(n):
            if M[i][col] == 1:
                x[col] = M[i][n]
                break
    return x
