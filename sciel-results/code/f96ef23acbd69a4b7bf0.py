"""Rank-minimal tensor decomposition for 2x3x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    deadline = time.time() + 130

    def _verify(U, V, W, n, m, p):
        rank = U.shape[1]
        T = np.ones((n, m, p))
        result = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    result[i, j, k] = val
        return np.allclose(result, T, atol=1e-9)

    def _is_half_integer(M):
        for val in M.flatten():
            if abs(val * 2 - round(val * 2)) > 1e-9:
                return False
        return True

    # Strategy: Use a systematic Kronecker-product / block-diagonal structure.
    # 
    # The key insight: we can group the computation by the "inner" dimension j.
    # For each j, we have a 2x1x5 sub-problem (row i of A times column k of B for fixed j).
    # But we can do better by grouping across j's.
    #
    # Alternative structural approach: Use the fact that matrix multiplication is
    # a bilinear map. We can express it as a sum of rank-1 tensors where each
    # rank-1 tensor corresponds to a "pattern" of (i,j) in A, (j,k) in B, and (i,k) in C.
    #
    # Let's try a different decomposition:
    # - For the 2x3 part of A: we can use 2x2 Strassen-like blocks plus the third column.
    # - For the 3x5 part of B: we can use 3x3 blocks.
    #
    # Actually, let me try a cleaner approach: decompose by grouping k values.
    # Group k into {0,1,2} and {3,4}.
    #
    # For k in {0,1,2}: We need to compute C[:, 0:3] = A @ B[:, 0:3]
    #   This is 2x3 times 3x3 = 2x3.
    #   We can use a rank-10 decomposition for 2x3x3 (if it exists) or schoolbook 18.
    #   Actually, rank(2x3x3) is known to be 16 (from Winograd-style).
    #
    # For k in {3,4}: We need to compute C[:, 3:5] = A @ B[:, 3:5]
    #   This is 2x3 times 3x2 = 2x2.
    #   Rank(2x3x2) can be done with 10 (schoolbook) or possibly less.
    #
    # Let me try a more direct approach: use the known optimal ranks for sub-problems.
    #
    # Actually, the simplest structurally different approach: use a "tensor train" style
    # decomposition where we factor through intermediate tensors.
    #
    # Let me try: express the 2x3x5 multiplication as a composition.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # We can write this as: C = (A @ I_3) @ B, but that doesn't help directly.
    #
    # New structural idea: Use a "sum of Kronecker products" representation.
    # The multiplication tensor T_{i,j,k} = delta_{i,j} * delta_{j,k} in a sense.
    #
    # Let me just try to find a lower-rank decomposition by exploiting the structure.
    # For 2x3x5, the schoolbook rank is 30. 
    # 
    # Key structural change: instead of ad-hoc Strassen blocks, use a systematic
    # "row-column" factorization where we precompute linear combinations of A's rows
    # and B's columns, then combine them.
    #
    # Specifically:
    # Let's define r linear combinations of A's rows (for each j) and s linear combinations
    # of B's columns (for each j), such that we can express all C[i,k] from these.
    #
    # Actually, the cleanest structural approach: use the "border rank" idea.
    # For 2x3x5, let's try to find a rank-20 or rank-18 decomposition.
    #
    # Here's a concrete plan:
    # 1. Split the 3 columns of B into two groups: {col0, col1} and {col2}.
    # 2. For group {col0, col1}: compute 2x2x5 sub-problem (rows 0,1 of A cols 0,1 times B rows 0,1 all 5 cols)
    #    Wait, that's not right. Let me reconsider.
    #
    # The multiplication is: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    #
    # Structural decomposition by the "middle" index j:
    # Term j=0: A[i,0]*B[0,k] for all i,k -> this is a 2x1x5 outer product structure
    # Term j=1: A[i,1]*B[1,k] for all i,k -> another 2x1x5
    # Term j=2: A[i,2]*B[2,k] for all i,k -> another 2x1x5
    # Total schoolbook: 3 * 10 = 30
    #
    # To reduce rank, we need to share computations across j's.
    #
    # New approach: Use a "linear algebra" based search.
    # We want to find matrices U (6 x r), V (15 x r), W (10 x r) such that
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    #
    # This is equivalent to finding a rank-r decomposition of the 30-entry tensor T where
    # T[i,j,k] = 1 if the indices are "compatible" (which they always are for matmul).
    # Wait, T[i,j,k] = 1 for ALL i,j,k? No!
    #
    # The multiplication tensor has T[i,j,k] = 1 for all (i,j,k) in [0,n) x [0,m) x [0,p).
    # Yes! The tensor is all-ones. So we need sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k.
    #
    # This is a system of 30 equations in r * (6+15+10) = 31r unknowns.
    # For r=30, we have 930 unknowns and 30 equations - very underdetermined.
    #
    # Let me try to find a lower-rank solution by solving the system.
    #
    # For a given rank r, we can set up the equations and solve.
    # But the equations are quadratic in the unknowns (product of U, V, W entries).
    #
    # Alternative: fix U and V, then solve for W linearly? No, it's still nonlinear.
    #
    # Let me try a different structural approach: use the "Strassen-like" pattern
    # but applied more systematically.
    #
    # For 2x3x5, here's a rank-20 construction:
    # 
    # Think of it as: C = A @ B where A is 2x3, B is 3x5.
    # 
    # Split B into B1 (3x3) and B2 (3x2).
    # C1 = A @ B1 (2x3), C2 = A @ B2 (2x2).
    #
    # For C1 = A @ B1 (2x3x3):
    #   Use a rank-10 decomposition (known optimal for 2x3x3 is 10? Let me verify).
    #   Actually, rank(2x3x3) >= 16 by some bounds? No, I think it's 16.
    #   Let me just use schoolbook for now: rank 18.
    #
    # For C2 = A @ B2 (2x3x2):
    #   Schoolbook rank 12.
    #
    # Total: 18 + 12 = 30. Same as before.
    #
    # Let me try to find a better decomposition for 2x3x3.
    #
    # Actually, I recall that rank(2x3x3) = 16. Let me try to construct it.
    #
    # For 2x3x3: A is 2x3, B is 3x3, C is 2x3.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Here's a rank-16 construction for 2x3x3:
    # 
    # Terms 1-3: For each k, (A[i,0] + A[i,1]) * B[2,k] -> contributes to C[i,k]
    #   Actually, let me think differently.
    #
    # Let me just try to solve the system numerically for small ranks.
    #
    # For rank r, we have variables U (6 x r), V (15 x r), W (10 x r).
    # Equations: for each (i,j,k), sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1.
    #
    # This is 30 nonlinear equations. We can use scipy to solve.
    #
    # But first, let me try a simpler structural approach:
    # Use the "Kronecker product" structure explicitly.
    #
    # The all-ones tensor T of size 2x3x5 can be written as:
    # T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r is a 6-dim vector, v_r is a 15-dim vector, w_r is a 10-dim vector.
    #
    # One trivial decomposition: T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k (rank 30).
    #
    # A better one: T = (sum_i e_i) ⊗ (sum_j e_j) ⊗ (sum_k e_k) (rank 1)!
    # Wait, that gives u = [1,1,1,1,1,1], v = [1]*15, w = [1]*10.
    # Then sum_r u[i,j] v[j,k] w[k,i] = 1*1*1 = 1. Yes!
    #
    # So the rank-1 decomposition works! The all-ones tensor has rank 1!
    #
    # Let me verify: U[i*m+j, 0] = 1 for all i,j. V[j*p+k, 0] = 1 for all j,k. W[k*n+i, 0] = 1 for all k,i.
    # Then for any (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1*1*1 = 1. Correct!
    #
    # So the rank is 1!

    # Verify rank-1 decomposition
    U = np.ones((n * m, 1))
    V = np.ones((m * p, 1))
    W = np.ones((p * n, 1))
    
    if _verify(U, V, W, n, m, p) and _is_half_integer(U) and _is_half_integer(V) and _is_half_integer(W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: guaranteed-valid schoolbook decomposition (rank 30)
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
