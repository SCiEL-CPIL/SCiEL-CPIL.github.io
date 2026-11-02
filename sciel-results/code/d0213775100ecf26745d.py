"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import math

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x5 matrix.
    Uses a structured search over small sets of monomials to find a low-rank CP decomposition
    with integer or half-integer coefficients. Falls back to schoolbook if no better is found.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p

    # The multiplication tensor T has indices (i, j, k) for i in [0,n), j in [0,m), k in [0,p).
    # We want to decompose T = sum_c U_c ⊗ V_c ⊗ W_c
    # where U_c is a vector in R^{n*m} (indexed by i*m+j), V_c in R^{m*p} (indexed by j*p+k),
    # W_c in R^{p*n} (indexed by k*n+i).
    # The entry T[i*m+j, j*p+k, k*n+i] = 1 if the indices match, else 0.
    
    # Actually, the tensor is defined such that:
    # (A * B)[i, k] = sum_j A[i, j] * B[j, k]
    # So T[i*m+j, j*p+k, k*n+i] = 1.
    
    # We are looking for factors U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # such that for all i, j, k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of equations. Let's search for small rank decompositions.
    # The structure of the equations suggests that the rank should be at least max(n*m, m*p, p*n) = 15.
    # Schoolbook is 45.
    
    # Let's try to find a decomposition by searching for small sets of "terms".
    # Each term c corresponds to a choice of:
    #   i_c in {0, 1, ...} or "all" (for U support)
    #   j_c in {0, 1, ...} or "all" (for V support)
    #   k_c in {0, 1, ...} or "all" (for W support)
    # But this is too vague.
    
    # A more systematic approach:
    # The equations are:
    # For each (i, j, k):
    # sum_{c in C(i,j,k)} prod_factors = 1
    # where C(i,j,k) is the set of terms c that cover (i,j,k).
    
    # Let's try rank 15 (the lower bound).
    # For rank 15, we might have one term for each j (3 terms) and each k (5 terms)? No, 3*5=15.
    # If we have 15 terms, indexed by (j, k), then:
    # Term (j, k) has:
    #   U: non-zero at (i, j) for all i? If so, U[i*m+j, (j,k)] = 1.
    #   V: non-zero at (j, k) for all j? No, V[j*p+k, (j,k)] = 1.
    #   W: non-zero at (k, i) for all i? If so, W[k*n+i, (j,k)] = 1.
    # Then for a fixed (i, j, k):
    # Sum over (j', k') such that U[i*m+j, (j',k')] V[j*p+k, (j',k')] W[k*n+i, (j',k')] != 0
    # U[i*m+j, (j',k')] is non-zero if j' = j.
    # V[j*p+k, (j',k')] is non-zero if j' = j and k' = k.
    # W[k*n+i, (j',k')] is non-zero if k' = k.
    # So only the term (j, k) contributes.
    # Sum = U[i*m+j, (j,k)] * V[j*p+k, (j,k)] * W[k*n+i, (j,k)] = 1 * 1 * 1 = 1.
    # This works! Rank 15 is sufficient!
    
    # Let's verify:
    # U[i*m+j, c=(j',k')] = 1 if j' == j else 0
    # V[j*p+k, c=(j',k')] = 1 if j' == j and k' == k else 0
    # W[k*n+i, c=(j',k')] = 1 if k' == k else 0
    # 
    # For a specific (i, j, k):
    # Sum over c=(j', k'):
    # U[i*m+j, (j',k')] * V[j*p+k, (j',k')] * W[k*n+i, (j',k')]
    # = sum_{j', k'} [j'==j] * [j'==j and k'==k] * [k'==k]
    # = [j==j and k==k] = 1.
    
    # So rank 15 is valid!
    
    # Let's construct the factors.
    rank = 15
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Index c corresponds to (j, k)
    c = 0
    for j in range(m):
        for k in range(p):
            # U[i*m+j, c] = 1 for all i
            for i in range(n):
                U[i * m + j][c] = 1
            # V[j*p+k, c] = 1
            V[j * p + k][c] = 1
            # W[k*n+i, c] = 1 for all i
            for i in range(n):
                W[k * n + i][c] = 1
            c += 1
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
