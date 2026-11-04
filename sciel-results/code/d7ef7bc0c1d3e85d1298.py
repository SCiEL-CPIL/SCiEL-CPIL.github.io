import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomp(U, V, W):
        """Check if sum_r U[:,r] V[:,r] W[:,r] == T"""
        rank = U.shape[1]
        for r in range(rank):
            pass
        # Efficient check using einsum
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=1e-9)

    def get_schoolbook():
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
        return U, V, W

    def get_rank50():
        """Construct a known valid rank-50 decomposition using Strassen blocks."""
        rank = 50
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            # M1: (a+d)(e+h)
            U[i0 * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M2: (c+d)e
            U[(i0 + 1) * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] -= 1
            col += 1

            # M3: a(f-h)
            U[i0 * m + j0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M4: d(g-e)
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + k0, col] += 1
            V[j0 * p + k0, col] -= 1
            W[k0 * n + i0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            col += 1

            # M5: (a+b)h
            U[i0 * m + j0, col] += 1
            U[i0 * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            col += 1

            # M6: (c-a)(e+f)
            U[(i0 + 1) * m + j0, col] += 1
            U[i0 * m + j0, col] -= 1
            V[j0 * p + k0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M7: (b-d)(g+h)
            U[i0 * m + (j0 + 1), col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] -= 1
            V[(j0 + 1) * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            col += 1

        # Tile 2x4 x 4x7:
        # A is 2 rows, 4 cols. B is 4 rows, 7 cols. C is 2 rows, 7 cols.
        # Split A cols into pairs: (0,1), (2,3)
        # Split B rows into pairs: (0,1), (2,3)
        # Split B cols: (0,1), (2,3), (4,5), and col 6 separately
        # Split C cols: (0,1), (2,3), (4,5), and col 6 separately

        # For each pair of A-cols and B-rows, and each pair of B-cols/C-cols:
        # Use Strassen for 2x2 x 2x2 blocks
        # A block rows: always both rows (i0=0)
        # A block cols: j0 in {0, 2}
        # B block rows: j0 in {0, 2} (same as A cols)
        # B block cols: k0 in {0, 2, 4}

        # Strassen blocks for 2x2 x 2x2:
        # j0 = 0,2 (A col pairs / B row pairs)
        # k0 = 0,2,4 (B col pairs / C col pairs)
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                add_strassen(0, j0, k0)

        # Handle column 6 of B (and C) separately with rank-1 terms
        # A[i,j] * B[j,6] for i=0,1; j=0,1,2,3
        for j in range(m):
            U[0 * m + j, col] = 1  # A[0,j]
            V[j * p + 6, col] = 1  # B[j,6]
            W[6 * n + 0, col] = 1  # C[0,6]
            col += 1
            U[1 * m + j, col] = 1  # A[1,j]
            V[j * p + 6, col] = 1  # B[j,6]
            W[6 * n + 1, col] = 1  # C[1,6]
            col += 1

        # Should be exactly rank 50
        assert col == rank, f"Expected rank 50, got {col}"
        return U, V, W

    # Get the rank-50 decomposition
    U50, V50, W50 = get_rank50()

    # Verify it's correct
    if not verify_decomp(U50, V50, W50):
        # Fallback to schoolbook
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Now try to reduce the rank using linear algebra
    # Strategy: check if any column is a linear combination of others
    # For each column r, check if the vector (U[:,r] ⊗ V[:,r]) is in the span
    # of {U[:,s] ⊗ V[:,s] for s != r}, with W[:,r] providing the output

    # More precisely: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # We can try to remove column r if there exist coefficients alpha_s such that:
    # U[:,r] ⊗ V[:,r] ⊗ W[:,r] = sum_{s != r} alpha_s * U[:,s] ⊗ V[:,s] ⊗ W[:,s]
    # This is hard in general. Instead, use a simpler heuristic:
    # Check if removing a column and re-solving still works.

    # Alternative approach: use the matricization rank as a lower bound
    T_mat1 = T.reshape(n * m, -1)
    T_mat2 = T.reshape(m * p, -1)
    T_mat3 = T.reshape(p * n, -1)

    rank_lower1 = np.linalg.matrix_rank(T_mat1)
    rank_lower2 = np.linalg.matrix_rank(T_mat2)
    rank_lower3 = np.linalg.matrix_rank(T_mat3)
    rank_lower = max(rank_lower1, rank_lower2, rank_lower3)

    # Try to reduce from 50 down to rank_lower
    # Use a greedy approach: try removing each column and check if the rest can compensate
    # This is expensive, so we'll use a simpler heuristic

    # For now, try a few specific reductions
    # Check if we can merge columns that have similar structure

    # Actually, let's try a more systematic approach:
    # For each pair of columns (r, s), check if we can eliminate one
    # by expressing it as a linear combination

    # Given time constraints, let's just try to see if rank < 50 is achievable
    # by checking the matricization rank

    if time.time() > deadline - 20:
        return {"U": U50.tolist(), "V": V50.tolist(), "W": W50.tolist()}

    # Try rank reduction: for each column, try to see if it's redundant
    # by checking if the tensor can be reconstructed without it
    best_U, best_V, best_W = U50.copy(), V50.copy(), W50.copy()
    best_rank = 50

    # Try removing each column and see if the remaining columns can be adjusted
    # This is a nonlinear problem, so we'll use a simple heuristic:
    # Check if the column is exactly zero (it's not) or if it's a linear combination

    # For a more practical approach, let's try specific known reductions
    # For 2x4x7, the optimal rank might be 48 or lower

    # Try to find if any two columns can be merged
    # Check linear dependence of the Kronecker products
    for r in range(50):
        if time.time() > deadline - 10:
            break
        # Check if column r is redundant
        # Remove it and check if the residual can be expressed
        U_test = np.delete(U50, r, axis=1)
        V_test = np.delete(V50, r, axis=1)
        W_test = np.delete(W50, r, axis=1)

        # Check if T - T_without_r is rank 1 in some sense
        # This is expensive, so skip for now

    # Return the best decomposition found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
