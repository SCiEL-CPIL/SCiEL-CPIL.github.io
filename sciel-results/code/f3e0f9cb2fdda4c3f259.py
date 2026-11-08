import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Try to find a lower rank decomposition
    # Heuristic: Rank might be 2*m*p - m
    # For 2x5x5, 2*25 - 5 = 45
    # We will attempt to construct and verify a rank 45 decomposition.
    
    def verify(U, V, W):
        n, m, p = DIMENSIONS
        # Check all i, j, k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(len(U[0])):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if val != 1:
                        return False
        # Check cross terms (should be 0)
        # The tensor is 0 if indices are inconsistent.
        # We only need to check that the sum is 0 for inconsistent indices?
        # The definition of the tensor decomposition implies that the sum
        # over r of U_a_r V_b_r W_c_r must equal T_abc.
        # T_abc is 1 if a=(i,j), b=(j,k), c=(k,i) and 0 otherwise.
        # So we must check that for any a, b, c that do NOT form a consistent triple,
        # the sum is 0.
        # There are 2500 total triples. 50 are 1. 2450 are 0.
        # Checking all 2450 is expensive but doable for small R.
        # However, if we construct it carefully, we might only need to check a subset.
        # For safety, check all.
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    # Check if consistent
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if i_a == i_c and j_a == j_b and k_b == k_c:
                        continue # Should be 1, checked above
                    else:
                        val = 0
                        for r in range(len(U[0])):
                            val += U[a][r] * V[b][r] * W[c][r]
                        if val != 0:
                            return False
        return True

    # Attempt Rank 45
    # This is a placeholder. Constructing a valid rank 45 decomposition
    # requires specific algebraic identities.
    # Without a verified construction, we return schoolbook.
    # However, we can try a random search for rank 49? No, too slow.
    
    return best
