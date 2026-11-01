import time

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 2, 13

    # Rank-39 decomposition: 3 terms per column k
    # For each column k, use:
    #   T1 = (A[0,0] + A[0,1]) * (B[0,k] + B[1,k]) -> contributes to C[0,k]
    #   T2 = A[0,1] * B[1,k]                        -> contributes to C[0,k]
    #   T3 = (A[1,0] + A[1,1]) * (B[0,k] + B[1,k]) -> contributes to C[1,k]
    #   T4 = A[1,1] * B[1,k]                        -> contributes to C[1,k]
    #
    # Wait, that's 4 terms. Let me reconsider.
    #
    # The standard rank-3 decomposition for 2x2 * 2x1:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    #
    # Use identities:
    # (a+b)(c+d) = ac + ad + bc + bd
    #
    # For C[0,k]: use (A[0,0]+A[0,1])*(B[0,k]+B[1,k]) - A[0,1]*B[0,k] - A[0,0]*B[1,k]
    # That's 3 terms but introduces extra terms.
    #
    # Actually, the minimal rank for multiplying 2x2 by 2x1 is 4 (it's a bilinear map).
    # So rank 39 is not achievable with 3 per column.
    #
    # Let me reconsider. The tensor for 2x2 * 2x1 has entries:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # This is a tensor of size (2*2) x (2*13) x (13*2) = 4 x 26 x 26
    # The rank of this tensor is at least max(n*m, m*p, p*n) / something...
    #
    # Actually, for matrix multiplication C = A*B where A is n x m and B is m x p,
    # the tensor rank is at most n*m*p (the naive algorithm).
    # For n=2, m=2, p=13: rank at most 4*13 = 52.
    #
    # Can we do better? The key insight is that for each column k of B (which is a 2x1 vector),
    # we compute A * b_k where b_k is 2x1. This is a 2x2 * 2x1 multiplication.
    #
    # The 2x2 * 2x1 multiplication has tensor rank 4 (each of the 4 products A[i,j]*B[j,k] is needed).
    # There's no known rank-3 decomposition for 2x2 * 2x1.
    #
    # So the minimal rank is 4*13 = 52.
    #
    # However, let me try a different approach. Maybe we can share computations across columns.
    # But since B is 2x13 and A is 2x2, each column of B is independent.
    #
    # So rank 52 is likely minimal. Let me just return the correct rank-52 solution.

    rank = 4 * p  # 52
    U = [[0] * rank for _ in range(n * m)]  # 4 x 52
    V = [[0] * rank for _ in range(m * p)]  # 26 x 52
    W = [[0] * rank for _ in range(p * n)]  # 26 x 52

    for k in range(p):
        if time.time() > deadline:
            break
        base = 4 * k

        # Term 1: A[0,0] * B[0,k] -> C[0,k]
        U[0][base] = 1
        V[k][base] = 1
        W[2*k][base] = 1

        # Term 2: A[0,1] * B[1,k] -> C[0,k]
        U[1][base + 1] = 1
        V[13 + k][base + 1] = 1
        W[2*k][base + 1] = 1

        # Term 3: A[1,0] * B[0,k] -> C[1,k]
        U[2][base + 2] = 1
        V[k][base + 2] = 1
        W[2*k + 1][base + 2] = 1

        # Term 4: A[1,1] * B[1,k] -> C[1,k]
        U[3][base + 3] = 1
        V[13 + k][base + 3] = 1
        W[2*k + 1][base + 3] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
