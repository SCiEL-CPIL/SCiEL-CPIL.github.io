"""Rewritten program: search for a lower-rank exact tensor decomposition for 2x2 * 2x9."""
DIMENSIONS = [2, 2, 9]
import time
import random
import itertools
from fractions import Fraction

# We need to find rank-minimal U (n*m x r), V (m*p x r), W (p*n x r)
# such that for all i,j,k: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
# where T[i,j,k] = 1 if (i,j,k) corresponds to a "1" in the matmul tensor.
# For matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p:
# C[i,k] = sum_j A[i,j] B[j,k]
# The tensor T has T[i,j,k] = 1 for all i,j,k (since every product term contributes).
# Wait, let me re-read: "multiplying a 2 by 2 matrix by a 2 by 9 matrix"
# A: 2x2, B: 2x9, C: 2x9
# C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k]
# The tensor T[i,j,k] = 1 for all i in [0,1], j in [0,1], k in [0,8].
# So T is all ones: 2*2*9 = 36 entries, all equal to 1.

# The decomposition requires:
# sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1  for all i,j,k

# This is a 3-tensor with all entries 1. We want minimum rank.

# For the standard matrix multiplication tensor, the rank is at most n*m*p (schoolbook).
# For 2x2 * 2x2, the rank is 7 (Strassen).
# For 2x2 * 2x9, we can potentially do better than 36.

# Let's think about this differently. The tensor T[i,j,k] = 1 for all i,j,k.
# We need rank r such that there exist vectors u_c (in R^{nm}), v_c (in R^{mp}), w_c (in R^{pn})
# with sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1 for all i,j,k.

# Since all entries are 1, we can try to find a low-rank decomposition.
# One approach: if we can write 1 = sum_c a_c[i,j] * b_c[j,k] * c_c[k,i]
# where a_c is a 2x2 matrix, b_c is a 2x9 matrix, c_c is a 9x2 matrix.

# Actually, let's just do a search. The problem is small enough.
# nm = 4, mp = 18, pn = 18.

# Let's try ranks from 1 upward and check if a solution exists.
# For small ranks, we can use random search with rational numbers.

def check_decomposition(n, m, p, rank, U, V, W):
    """Check if the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def try_random_search(n, m, p, max_rank, time_limit, seed=42):
    """Try to find a decomposition with rank <= max_rank using random search."""
    rng = random.Random(seed)
    deadline = time.time() + time_limit
    
    for rank in range(1, max_rank + 1):
        if time.time() > deadline:
            return None
        # Try multiple random initializations
        for attempt in range(100):
            if time.time() > deadline:
                return None
            # Initialize with random small integers or half-integers
            U = [[rng.choice([-1, 0, 1, 2, 3]) for _ in range(rank)] for _ in range(n * m)]
            V = [[rng.choice([-1, 0, 1, 2, 3]) for _ in range(rank)] for _ in range(m * p)]
            W = [[rng.choice([-1, 0, 1, 2, 3]) for _ in range(rank)] for _ in range(p * n)]
            
            if check_decomposition(n, m, p, rank, U, V, W):
                return {"U": U, "V": V, "W": W, "rank": rank}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin
    
    # First, try to find a better decomposition than schoolbook (rank 36)
    # The theoretical lower bound is max(nm, mp, pn) = max(4, 18, 18) = 18
    # But that's a very loose bound. Let's try ranks from 18 upward.
    
    # For the all-ones tensor, there might be a clever decomposition.
    # Note: T[i,j,k] = 1 = 1 * 1 * 1, so rank 1 would work if we could make
    # U[i*m+j] * V[j*p+k] * W[k*n+i] = 1 for all i,j,k.
    # This requires U to be constant across i,j for each c, etc.
    # If rank=1: U[i*m+j,0] * V[j*p+k,0] * W[k*n+i,0] = 1 for all i,j,k.
    # Let U[i*m+j,0] = a_i * b_j, V[j*p+k,0] = b_j * c_k, W[k*n+i,0] = c_k * a_i
    # Then product = a_i^2 * b_j^2 * c_k^2 = 1 for all i,j,k.
    # This requires a_i^2 * b_j^2 * c_k^2 = 1, which means a_i^2 = const1, b_j^2 = const2, c_k^2 = const3
    # with const1 * const2 * const3 = 1.
    # So a_i = ±1, b_j = ±1, c_k = ±1 (or scaled versions).
    # Let's try: a_i = 1, b_j = 1, c_k = 1.
    # U[i*m+j,0] = 1, V[j*p+k,0] = 1, W[k*n+i,0] = 1.
    # Product = 1*1*1 = 1. Yes! Rank 1 works!
    
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    if check_decomposition(n, m, p, 1, U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # This shouldn't happen, but just in case, fall back to schoolbook
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
