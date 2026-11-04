import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 5, 5]


def _solve_rational_system(A, b):
    """Solve Ax = b over rationals using Gaussian elimination.
    A is a 2D list of Fractions, b is a 1D list of Fractions.
    Returns a list of Fractions (one solution) or None if inconsistent.
    """
    m = len(A)
    if m == 0:
        return [Fraction(0) for _ in range(len(b))]
    n = len(A[0])
    # Augmented matrix
    M = [[A[i][j] for j in range(n)] + [b[i]] for i in range(m)]
    rows = m
    cols = n + 1
    rank = 0
    pivot_cols = []
    
    for col in range(n):
        # Find pivot
        pivot = None
        for row in range(rank, rows):
            if M[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            continue
        # Swap
        if pivot != rank:
            M[rank], M[pivot] = M[pivot], M[rank]
        # Normalize
        pv = M[rank][col]
        M[rank] = [x / pv for x in M[rank]]
        # Eliminate
        for row in range(rows):
            if row != rank and M[row][col] != 0:
                factor = M[row][col]
                M[row] = [M[row][j] - factor * M[rank][j] for j in range(cols)]
        pivot_cols.append(col)
        rank += 1
    
    # Check consistency
    for row in range(rank, rows):
        if M[row][n] != 0:
            return None
    
    # Build solution: free variables = 0
    sol = [Fraction(0) for _ in range(n)]
    for i in range(rank):
        sol[pivot_cols[i]] = M[i][n]
    return sol


def _build_system(n, m, p, r):
    """Build the linear system for tensor decomposition.
    Unknowns: U[i*m+j, t], V[j*p+k, t], W[k*n+i, t] for t in [0, r)
    Each triple (i,j,k) gives: sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
    
    This is not linear in all variables. But we can fix W and solve linearly for U and V,
    or use the fact that it's a system we can attack iteratively.
    
    For the minimal rank search, we use a different approach:
    We note that for fixed W, the system is bilinear in (U, V).
    
    Alternative: treat as a matrix factorization problem.
    The tensor T_{ijk} = sum_t U_{ij,t} V_{jk,t} W_{ki,t}
    
    We can reshape: for each t, U_t (n*m x 1), V_t (m*p x 1), W_t (p*n x 1)
    T = sum_t outer products.
    
    Since T_{ijk} = 1 for all i,j,k, we need:
    sum_t U_{ij,t} * V_{jk,t} * W_{ki,t} = 1
    
    This is still nonlinear. But we can use the following strategy:
    Fix W to be simple (e.g., W_{ki,t} = delta_{k,0} * delta_{i,0} for some pattern)
    and solve the resulting linear system for U and V.
    
    Actually, a cleaner approach: since the tensor is all-ones, we can use
    the rank-1 decomposition: T_{ijk} = sum_t a_t(i) b_t(j) c_t(k)
    where a_t(i)*b_t(j)*c_t(k) sums to 1.
    
    For the all-ones tensor of size n x m x p, the tensor rank is at most n*m*p
    but can be much less. For all-ones tensor, rank = max(n, m, p) is NOT correct.
    
    Actually, the all-ones tensor T_{ijk} = 1 has tensor rank:
    We can write 1 = sum_t f_t(i) g_t(j) h_t(k)
    If we let f_t(i) = 1 for all i, then we need sum_t g_t(j) h_t(k) = 1
    This is the rank-1 decomposition of the all-ones matrix of size m x p, which has rank 1.
    So: f_1(i)=1, g_1(j)=1, h_1(k)=1 gives T_{ijk}=1 with rank 1!
    
    Wait, that's just rank 1. Let me verify:
    U_{ij,1} = 1, V_{jk,1} = 1, W_{ki,1} = 1
    Then sum_t U_{ij,t} V_{jk,t} W_{ki,t} = 1*1*1 = 1. Yes!
    
    So the all-ones tensor has rank 1!
    """
    pass


def construct():
    n, m, p = DIMENSIONS
    
    # The tensor T_{ijk} = 1 for all i,j,k
    # Rank-1 decomposition: U_{ij,1}=1, V_{jk,1}=1, W_{ki,1}=1
    # This gives sum_t=1: 1*1*1 = 1 = T_{ijk}
    
    rank = 1
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            U[i * m + j][0] = 1
    for j in range(m):
        for k in range(p):
            V[j * p + k][0] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][0] = 1
    
    return {"U": U, "V": V, "W": W}
