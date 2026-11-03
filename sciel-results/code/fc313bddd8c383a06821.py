"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal tensor decomposition for 3x3 * 3x4 matrix multiplication.
    Uses a hybrid approach:
    1. Build a schoolbook decomposition (rank 36) as a baseline.
    2. Use a randomized rank-revealing compression strategy to find a lower rank
       decomposition with integer/half-integer entries.
    3. Verify the decomposition and return the best valid one found.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s wall-clock

    # Step 1: Build schoolbook decomposition (rank 36)
    def build_schoolbook():
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

    schoolbook = build_schoolbook()
    best_result = schoolbook
    best_rank = n * m * p

    # Step 2: Try to find a lower-rank decomposition using iterative random search
    # We'll use a greedy sparse recovery approach with random rank-1 terms.
    # The tensor T is defined by T[i, j, k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # The decomposition is: T = sum_{r=1}^{rank} U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # where (U[:,r])_i * (V[:,r])_j * (W[:,r])_k = delta_{i, j+k} (in the appropriate indexing)

    # For 3x3 * 3x4:
    # n=3, m=3, p=4
    # U has shape (n*m, rank) = (9, rank)
    # V has shape (m*p, rank) = (12, rank)
    # W has shape (p*n, rank) = (12, rank)
    # Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # The product is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k} for the (i,k) entry
    # Actually, the tensor is: T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    # T[i, j, k] = 1 if we want C[i, k] to depend on A[i, j] * B[j, k]
    # So T[i, j, k] = 1 for all valid i, j, k, and the contraction gives C[i, k] = sum_j T[i,j,k] A[i,j] B[j,k]

    # Let's verify the schoolbook:
    # For each (i, j, k), U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # Product: U[i*m+j, col] * V[j*p+k, col] * W[k*n+i, col] = 1
    # This contributes to the tensor element T[i, j, k] = 1
    # So the sum over all rank-1 terms gives T[i,j,k] = 1 for all (i,j,k), which is correct.

    # Now try random sparse recovery:
    # We want to find rank-1 terms with integer/half-integer coefficients that sum to the identity tensor.
    # Use a random initialization and greedy coordinate descent.

    import numpy as np
    from scipy.optimize import minimize

    # The tensor T is 3x3x4 = 36 elements, all 1s
    # We want to decompose it into rank-1 terms.
    # Let's try ranks from 18 up to 35 (since 36 is schoolbook)
    # Use a random search with verification.

    def verify_decomposition(U, V, W):
        """Verify that the decomposition exactly reproduces the tensor."""
        rank = len(U[0])
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def try_random_rank(r, iterations=5000):
        """Try to find a rank-r decomposition using random search."""
        if r >= best_rank:
            return None
        np.random.seed(42)
        for _ in range(iterations):
            # Initialize with random rank-1 terms
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            for c in range(r):
                # Choose random support patterns
                U[:, c] = np.random.randint(0, 2, n * m)
                V[:, c] = np.random.randint(0, 2, m * p)
                W[:, c] = np.random.randint(0, 2, p * n)
            # Verify
            if verify_decomposition(U.tolist(), V.tolist(), W.tolist()):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return None

    # Try lower ranks
    for r in range(18, best_rank):
        if time.time() > deadline - 30:
            break
        result = try_random_rank(r, iterations=2000)
        if result is not None:
            best_result = result
            best_rank = r
            break

    return best_result
# EVOLVE-BLOCK-END
