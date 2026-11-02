"""Matrix multiplication 3,3,8 — rank-minimal exact tensor decomposition."""
import time
import math

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s wall-clock

    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 72

    # Build the multiplication tensor T[i, j, k] = delta(i, k)
    # T is n x m x p
    # We seek a decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]

    # Try to find a lower rank by exploiting structure.
    # For m=3, each "fiber" T[:, j, :] is n x p.
    # T[:, j, :] = I_n (n x p identity embedded, i.e., first n columns of I_p, but p=8, n=3)
    # Actually T[i,j,k] = 1 if i==k else 0, for all j.
    # So T[:, j, :] is the same for all j: it's a 3x8 matrix with 1s at (0,0),(1,1),(2,2) and 0s elsewhere.

    # The tensor can be written as: T = sum_j sum_{i=0}^{n-1} e_i (in U) * e_j (in V) * e_i (in W)
    # Wait, more carefully: T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    # If we set U[i*m+j,c] = 1 when c = i*m+j (diagonal), V[j*p+k,c] = 1 when c = j*p+k (diagonal),
    # W[k*n+i,c] = 1 when c = k*n+i (diagonal), then for the term c = i*m+j = j*p+k = k*n+i to be nonzero,
    # we need i*m+j = j*p+k AND k*n+i = j*p+k.
    # This is exactly the schoolbook: each (i,j,k) triple with i==k gives one term.
    # Number of such triples: for each j (0..m-1), for each i (0..n-1), k=i, so n*m = 9 terms? No wait.
    # The schoolbook has n*m*p = 72 terms, one per (i,j,k). But T[i,j,k] = delta(i,k), so only n*m = 9 terms are nonzero in T.
    # However, the decomposition must reproduce ALL entries, including zeros. The schoolbook uses 72 rank-1 tensors,
    # but only 9 of them contribute nonzero values. The rest contribute to positions where T=0 but the product U*V*W is 0
    # because the indices don't match.

    # Key insight: T[i,j,k] = delta(i,k) is independent of j. So we can factor out the j-dependence.
    # T[i,j,k] = delta(i,k) * 1 = delta(i,k) * sum_j' e_{j'} * (1/m * m) ... no.
    # Actually: delta(i,k) * 1 = delta(i,k) * (1/m) * sum_{j'} 1
    # But V has entries V[j*p+k, c]. If we use a rank-1 tensor with V[j*p+k, c] = 1 for all j,
    # that would work for any j.

    # Let's try: rank = n * p = 3*8 = 24? Or n = 3?
    # Consider rank-1 tensors of the form: U[i*m+j, c] = f(i,j), V[j*p+k, c] = g(j,k), W[k*n+i, c] = h(k,i)
    # We need sum_c f(i,j) g(j,k) h(k,i) = delta(i,k) for all i,j,k.

    # Simplest: use n rank-1 tensors (one per i=k value).
    # For c = i (0,1,2): U[i*m+j, c] = 1 if i==c else 0, V[j*p+k, c] = 1 if k==c else 0, W[k*n+i, c] = 1 if (k==c and i==c) else 0
    # Then sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = sum_c [i==c][k==c][k==c][i==c] = sum_c [i==c][k==c] = delta(i,k).
    # Yes! This works with rank = n = 3.

    rank = n
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c in range(n):
        for i in range(n):
            for j in range(m):
                if i == c:
                    U[i * m + j][c] = 1
        for j in range(m):
            for k in range(p):
                if k == c:
                    V[j * p + k][c] = 1
        for k in range(p):
            for i in range(n):
                if k == c and i == c:
                    W[k * n + i][c] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
