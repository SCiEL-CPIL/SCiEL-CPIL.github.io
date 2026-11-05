"""Rewritten program: search for a rank-minimal exact tensor decomposition for 3x5 * 5x7."""
import time
import itertools
import random
import math

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Tensor T[i, j, k] = 1 if the output C[i,k] uses A[i,j]*B[j,k]
    # In the decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # W index is k*n + i (trace convention W[k*n+i, r])

    # Schoolbook rank
    schoolbook_rank = n * m * p  # 3*5*7 = 105

    # Try to find a lower rank decomposition
    # For small matrices, we can try various approaches:
    # 1. Check if rank = n*p works (rank of C matrix)
    # 2. Try incremental rank search with randomized methods

    # The tensor T has entries T[i,j,k] = 1 for all i,j,k (since every product term contributes)
    # Wait, that's not right. Let me reconsider.
    #
    # Matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] = 1 for all valid i,j,k.
    # A rank-r decomposition means: T[i,j,k] = sum_{r=1}^{R} U[i,j,r] * V[j,k,r] * W[i,k,r]
    # where W uses the convention W[k*n+i, r].
    #
    # So we need: 1 = sum_{r=1}^{R} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all i,j,k.

    # The minimum rank for matrix multiplication is known to be at most n*m*p (schoolbook).
    # For 3x5 and 5x7, the exact minimum rank is not known in general, but we can try to find good decompositions.

    # Strategy: Try to find a decomposition with rank < schoolbook using randomized search.
    # For small dimensions, we can try:
    # - Rank n*p = 21: This would mean each output entry is a single product (impossible in general)
    # - Rank between n*p and n*m*p

    # Actually, let's think about this differently. The tensor T is the all-ones tensor of size 3x5x7.
    # We want to factor it as a sum of rank-1 tensors.

    # A rank-1 tensor is U[i,j] * V[j,k] * W[i,k] where U depends only on (i,j), V only on (j,k), W only on (i,k).

    # For the all-ones tensor, one trivial decomposition is the schoolbook one with rank 105.

    # Can we do better? Let's try a structured approach.
    # If we set U[i,j,r] = a_r(i,j), V[j,k,r] = b_r(j,k), W[i,k,r] = c_r(i,k),
    # then we need sum_r a_r(i,j) * b_r(j,k) * c_r(i,k) = 1 for all i,j,k.

    # This is a system of equations. For small ranks, we can try to solve it.

    # Let's try rank = n*m*p first as fallback, then try to reduce.

    # Approach: Use the fact that for the all-ones tensor, we can use:
    # - One term per (i,k) pair with U[i,j,r] depending on j, etc.
    # - Or use a more clever construction.

    # Actually, the minimum rank for matrix multiplication C_{n,p} = A_{n,m} * B_{m,p}
    # is at least max(n*p, m) and at most n*m*p.
    # For n=3, m=5, p=7: lower bound is max(21, 5) = 21, upper bound is 105.

    # Let's try to find a decomposition with rank as low as possible.
    # We'll use a randomized approach: start with random rank-r tensors and try to fit.

    # For the all-ones tensor, a simple construction:
    # Use rank = n*p = 21:
    # For each (i,k) pair, set one rank-1 term.
    # Term (i,k): U[i,j,r] = 1/m, V[j,k,r] = 1, W[i,k,r] = m
    # Wait, that gives sum over j of (1/m)*1*m = m, not 1.
    # Let me redo: We need sum_j A[i,j]B[j,k] to be represented.
    #
    # Actually, the tensor is T[i,j,k] = 1 for all i,j,k.
    # A rank-1 decomposition term is u(i,j) * v(j,k) * w(i,k).
    # We need sum_r u_r(i,j) * v_r(j,k) * w_r(i,k) = 1.

    # Simple construction with rank = m:
    # For r = 0..m-1 (corresponding to j=r):
    # u_r(i,j) = 1 if j=r else 0
    # v_r(j,k) = 1 if j=r else 0
    # w_r(i,k) = 1
    # Then sum_r u_r(i,j)*v_r(j,k)*w_r(i,k) = 1 if j is in [0,m) which is always true, but
    # this gives 1 only when... wait, this sums to 1 for each (i,j,k) since exactly one r matches j.
    # So rank m = 5 works! But wait, is that correct?

    # Let's verify: For a given (i,j,k), the sum is sum_{r=0}^{4} [j==r] * [j==r] * 1 = [j==r] for r=j = 1.
    # Yes! So rank m = 5 should work for the all-ones tensor.

    # But wait, is the tensor really all-ones? Let me re-examine the problem.
    # The matrix multiplication tensor T satisfies:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The bilinear form is: C[i,k] = sum_{j} A[i,j] * B[j,k]
    # As a trilinear tensor, T[i,j,k] = 1 for all i,j,k.
    # Yes, it's the all-ones tensor.

    # So a rank-m decomposition exists! Let's construct it.
    # Rank = m = 5:
    # For r in 0..m-1:
    #   U[i*m + j, r] = 1 if j == r else 0
    #   V[j*p + k, r] = 1 if j == r else 0
    #   W[k*n + i, r] = 1

    # Let's verify: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # = sum_r [j==r] * [j==r] * 1
    # = [j==r] where r=j
    # = 1. Correct!

    # So the minimum rank is at most 5. Can we do even better?
    # The lower bound is max(n*p, m) = max(21, 5) = 21? No, that lower bound is for general matrix multiplication
    # where the tensor entries are not all 1. For the all-ones tensor, the lower bound is different.

    # Actually, for the all-ones tensor of size n x m x p, the minimum rank is min(n, m, p) if we can
    # factor it appropriately. Let me think again.

    # We showed rank m works. By symmetry, rank n and rank p also work:
    # Rank n = 3:
    # For r in 0..n-1:
    #   U[i*m+j, r] = 1 if i == r else 0
    #   V[j*p+k, r] = 1
    #   W[k*n+i, r] = 1 if i == r else 0
    # Sum: sum_r [i==r] * 1 * [i==r] = [i==r] where r=i = 1. Correct!

    # Rank p = 7:
    # For r in 0..p-1:
    #   U[i*m+j, r] = 1
    #   V[j*p+k, r] = 1 if k == r else 0
    #   W[k*n+i, r] = 1 if k == r else 0
    # Sum: sum_r 1 * [k==r] * [k==r] = [k==r] where r=k = 1. Correct!

    # So the minimum rank is min(n, m, p) = min(3, 5, 7) = 3.

    # Let's construct the rank-3 decomposition:
    rank = min(n, m, p)  # = 3

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i == r:
                        U[i * m + j][r] = 1
                        V[j * p + k][r] = 1
                        W[k * n + i][r] = 1

    # Verify the decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    # Fallback to schoolbook if verification fails
                    rank = n * m * p
                    U = [[0] * rank for _ in range(n * m)]
                    V = [[0] * rank for _ in range(m * p)]
                    W = [[0] * rank for _ in range(p * n)]
                    column = 0
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                U[i2 * m + j2][column] = 1
                                V[j2 * p + k2][column] = 1
                                W[k2 * n + i2][column] = 1
                                column += 1
                    return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
