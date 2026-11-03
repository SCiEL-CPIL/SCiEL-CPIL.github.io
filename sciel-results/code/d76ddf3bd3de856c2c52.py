"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x6 matrix.
    Uses a combination of known low-rank constructions and local search to minimize rank.
    """
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 5, 6

    # The tensor T is defined by T[i, j, k] = 1 if we're computing C[i,k] += A[i,j]*B[j,k]
    # We need to find U (n*m x rank), V (m*p x rank), W (p*n x rank) such that:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k
    # Note: W index is k*n + i (trace convention W[k*n+i, r])

    def schoolbook():
        """Standard rank n*m*p = 90 decomposition."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """Check if the decomposition is exact. Returns True if valid."""
        rank = len(U[0])
        # Check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Start with schoolbook
    best = schoolbook()
    best_rank = n * m * p  # 90

    # Try to find a better decomposition using known techniques
    # For small matrices, we can try random low-rank constructions

    # Strategy: Try ranks below 90 and see if we can find a valid decomposition
    # We'll use a greedy/constructive approach

    # First, let's try a simple optimization: can we merge some terms?
    # The schoolbook decomposition has rank 90. Let's see if we can do better.

    # For 3x5x6, the lower bound is max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # But that's just the dimension of the largest matrix.

    # Let's try a different approach: use the fact that we can sometimes
    # combine terms when there's structure.

    # Try rank from 30 up to 90, and for each rank try to construct a valid decomposition
    # using a randomized search

    def try_rank(rank, max_iter=10000):
        """Try to find a decomposition of the given rank using random search."""
        if time.time() > DEADLINE:
            return None

        # Initialize with random small values
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # This is a system of n*m*p = 90 equations in rank*15 + rank*30 + rank*18 = 63*rank unknowns

        # For rank >= ceil(90/63) = 2, we have more unknowns than equations
        # But the constraints are nonlinear (trilinear)

        # Let's try a constructive approach:
        # For each (i,j,k), we need at least one r where U,V,W are non-zero
        # and their product sums to 1.

        # Simplest: for each (i,j,k), assign a unique r (this is schoolbook, rank=90)
        # To reduce rank, we need to share r's across different (i,j,k) tuples.

        # Let's try: for a given rank, randomly assign each (i,j,k) to a "group" r
        # and then solve for the values.

        # Actually, let's try a more direct approach:
        # Fix U and V randomly, then solve for W.
        # Or fix two of them and solve for the third.

        # Let's try: fix U and V as random, then for each (i,j,k), we need
        # W[k*n+i, :] such that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # This is: (U[i*m+j,:] * V[j*p+k,:]) dot W[k*n+i,:] = 1
        # where * is element-wise product.

        # So if we define M[i,j,k,:] = U[i*m+j,:] * V[j*p+k,:] (element-wise),
        # then we need M[i,j,k,:] dot W[k*n+i,:] = 1 for all i,j,k.

        # This gives 90 equations in 18*rank unknowns (W has 18 rows, rank columns).
        # For rank >= ceil(90/18) = 5, this is overdetermined but might have solutions.

        # Let's try this approach:
        for attempt in range(max_iter):
            if time.time() > DEADLINE:
                return None

            # Random U and V
            U_rand = np.random.randn(n * m, rank)
            V_rand = np.random.randn(m * p, rank)

            # Build the linear system for W
            # For each (i,j,k), we have: sum_r U_rand[i*m+j,r] * V_rand[j*p+k,r] * W[k*n+i,r] = 1
            # Let A be a 90 x (18*rank) matrix, b be a 90-vector of 1s
            A = np.zeros((n * m * p, p * n * rank))
            b = np.ones(n * m * p)

            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(rank):
                            A[idx, (k * n + i) * rank + r] = U_rand[i * m + j, r] * V_rand[j * p + k, r]
                        idx += 1

            # Solve the linear system A * W_flat = b
            try:
                W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
                W_rand = W_flat.reshape(p * n, rank)
            except:
                continue

            # Check if the solution is exact
            # Reconstruct and verify
            U_list = U_rand.tolist()
            V_list = V_rand.tolist()
            W_list = W_rand.tolist()

            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U_list[i * m + j][r] * V_list[j * p + k][r] * W_list[k * n + i][r]
                        if abs(s - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break

            if valid:
                return {"U": U_list, "V": V_list, "W": W_list}

        return None

    # Try ranks from low to high
    for rank in range(30, best_rank):
        if time.time() > DEADLINE:
            break

        result = try_rank(rank, max_iter=5000)
        if result is not None:
            best = result
            best_rank = rank
            break  # Found the minimum rank

    # If no better rank found, return schoolbook
    return best
# EVOLVE-BLOCK-END
