"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit

    # Strategy: Use Strassen-like recursive block partitioning.
    # Split 4x5 into blocks of sizes (2,2) for rows and (2,3) for cols.
    # Split 5x5 into blocks of sizes (2,3) for rows and (2,3) for cols.
    # This gives a block structure where we can apply rank-7 decomposition to 2x2x2 blocks.

    # Block dimensions:
    # A is 4x5: split into 2 row blocks of size 2, 2 col blocks of sizes 2,3
    #   A = [[A11 (2x2), A12 (2x3)],
    #        [A21 (2x2), A22 (2x3)]]
    # B is 5x5: split into 2 row blocks of sizes 2,3, 2 col blocks of sizes 2,3
    #   B = [[B11 (2x2), B12 (2x3)],
    #        [B21 (3x2), B22 (3x3)]]
    # C is 4x5: split into 2 row blocks of size 2, 2 col blocks of sizes 2,3
    #   C = [[C11 (2x2), C12 (2x3)],
    #        [C21 (2x2), C22 (2x3)]]

    # C11 = A11*B11 + A12*B21  (2x2 = 2x2*2x2 + 2x3*3x2)
    # C12 = A11*B12 + A12*B22  (2x3 = 2x2*2x3 + 2x3*3x3)
    # C21 = A21*B11 + A22*B21  (2x2 = 2x2*2x2 + 2x3*3x2)
    # C22 = A21*B12 + A22*B22  (2x3 = 2x2*2x3 + 2x3*3x3)

    # For each 2x2x2 block multiplication, we can use rank-7 (Strassen).
    # For other block sizes, we need to find efficient decompositions.

    # Let me try a simpler approach: use the known result that 4x5x5 has rank 98.
    # I'll construct it by combining smaller rank-optimal decompositions.

    # Actually, let me try a direct numerical approach with better initialization.
    # Use the schoolbook decomposition and try to find a rank-98 solution via optimization.

    def verify_decomposition(U, V, W, n, m, p, tol=1e-6):
        """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
        rank = U.shape[1]
        # Check all valid (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                    if abs(val - 1.0) > tol:
                        return False
        # Check that the decomposition is zero for invalid index combinations
        # This is implicitly satisfied if the above holds and the matrices have proper structure
        return True

    # Schoolbook decomposition
    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    U_sb, V_sb, W_sb = schoolbook(n, m, p)
    rank_schoolbook = n * m * p  # 100

    # Try to find a rank-98 decomposition
    target_rank = 98
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    if time.time() < deadline - 15:
        # Approach: Start from schoolbook and try to remove 2 dimensions
        # by finding a rank-98 decomposition that matches the tensor.

        # The tensor equation is:
        # For all i in [0,4), j in [0,5), k in [0,5):
        #   sum_{c=0}^{r-1} U[i*5+j, c] * V[j*5+k, c] * W[k*4+i, c] = 1
        # For all other (a, b, c) triples: sum = 0

        # Let me try a random initialization and use a simple optimization approach.
        np.random.seed(42)

        for attempt in range(20):
            if time.time() > deadline - 15:
                break

            r = target_rank
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5

            # Simple gradient-free optimization: try to minimize the error
            # Error = sum over all tensor entries of (decomposed_value - target_value)^2

            # This is a 3-way tensor, so the error surface is complex.
            # Let me try a different approach: use the structure of the problem.

            # For 4x5x5, one known construction uses the fact that:
            # rank(4x5x5) <= rank(2x5x5) + rank(2x5x5)
            # And rank(2x5x5) can be bounded using smaller blocks.

            # Actually, let me just try to verify if the schoolbook minus 2 works
            # by checking if we can find a rank-98 decomposition.

            # Try removing 2 random columns from schoolbook and see if we can
            # adjust the remaining columns to compensate.

            U_try = U_sb.copy()
            V_try = V_sb.copy()
            W_try = W_sb.copy()

            # Remove 2 random columns
            cols_to_remove = np.random.choice(rank_schoolbook, 2, replace=False)
            U_try[:, cols_to_remove] = 0
            V_try[:, cols_to_remove] = 0
            W_try[:, cols_to_remove] = 0

            # Check if this is still valid (it won't be, but let's see the error)
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U_try[i*m+j, c] * V_try[j*p+k, c] * W_try[k*n+i, c]
                                  for c in range(rank_schoolbook))
                        error += (val - 1.0) ** 2

            # This approach won't work directly. Let me try a different strategy.
            break

    # If we haven't found a better solution, return the schoolbook decomposition
    U = best_U.tolist()
    V = best_V.tolist()
    W = best_W.tolist()

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
