"""Rewritten program for 3x4x4 matrix multiplication tensor decomposition."""
DIMENSIONS = [3, 4, 4]

import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Start with schoolbook (rank 48)
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

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank

    # Try to find a lower-rank decomposition using structured methods
    # For 3x4x4, we can try various approaches:
    # 1. Direct enumeration of small rank decompositions
    # 2. Strassen-like block decomposition
    # 3. Randomized search with verification

    # Verification function
    def verify(U_mat, V_mat, W_mat, n, m, p, tol=1e-9):
        r = U_mat.shape[1]
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    s = 0.0
                    for rr in range(r):
                        s += U_mat[a, rr] * V_mat[b, rr] * W_mat[c, rr]
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    is_valid = (i_a == i_c) and (j_a == j_b) and (k_b == k_c)
                    if is_valid:
                        if abs(s - 1.0) > tol:
                            return False
                    else:
                        if abs(s) > tol:
                            return False
        return True

    # Try rank 47 down to some lower bound
    # For small ranks, we can try structured decompositions
    
    # One approach: try to find decompositions with integer/half-integer entries
    # by solving linear systems or using known algebraic identities

    # For 3x4x4, the bilinear complexity is at most 48 (schoolbook)
    # There may exist decompositions of rank < 48

    # Let's try a systematic search for ranks from 47 down to 40
    # using a greedy or structured approach

    # Simple structured approach: try to combine schoolbook terms
    # For each pair of terms that share structure, try to combine them

    # Actually, let's try a more direct approach:
    # Initialize with random values and check if they satisfy the constraints
    # This is unlikely to work directly, but we can use it as a starting point

    # Better approach: use the fact that the tensor has a specific structure
    # and try to find low-rank approximations

    # For now, let's try a few specific structured decompositions

    # Try rank 44: see if we can save 4 terms
    # One idea: for the last row of A (i=2), the computation is:
    # C[2,:] = A[2,:] @ B
    # This is a 1x4 @ 4x4 multiplication, which needs 16 terms in schoolbook
    # But maybe we can share some computations

    # Let's try a random search for lower ranks
    np.random.seed(12345)

    for target_rank in range(best_rank - 1, max(best_rank - 15, 0), -1):
        if time.time() > deadline - 10:
            break

        # Try to find a decomposition with target_rank
        # Use random initialization and check
        # This is very unlikely to work, but let's try a few attempts

        attempts = 5
        found = False
        for _ in range(attempts):
            U_rand = np.random.randn(n * m, target_rank)
            V_rand = np.random.randn(m * p, target_rank)
            W_rand = np.random.randn(p * n, target_rank)

            if verify(U_rand, V_rand, W_rand, n, m, p):
                best_U, best_V, best_W = U_rand, V_rand, W_rand
                best_rank = target_rank
                found = True
                break

        if found:
            break

    # Return the best decomposition found
    return {
        'U': best_U.tolist(),
        'V': best_V.tolist(),
        'W': best_W.tolist()
    }
