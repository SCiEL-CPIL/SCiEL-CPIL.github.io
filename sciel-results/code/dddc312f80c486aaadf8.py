"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110s limit for construct()

    def make_seed():
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

    def validate(U, V, W):
        """Check if U, V, W form a valid tensor decomposition."""
        r = len(U[0])
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)

        # Check that entries are integer or half-integer
        Ua2 = Ua * 2
        Va2 = Va * 2
        Wa2 = Wa * 2
        if not np.allclose(Ua2, np.round(Ua2), atol=1e-6):
            return False
        if not np.allclose(Va2, np.round(Va2), atol=1e-6):
            return False
        if not np.allclose(Wa2, np.round(Wa2), atol=1e-6):
            return False

        # Build the multiplication tensor T[i, j, k] = 1 if (i,j,k) is a valid product term
        # The tensor we're decomposing is the matrix multiplication tensor:
        # T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor of shape (n,m,p))
        # Actually, the matmul tensor has T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p].
        # The decomposition says: sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = T[i,j,k]
        # So we need: for all (i,j,k), sum_l U[i*m+j,l] * V[j*p+k,l] * W[k*n+i,l] = 1

        # Compute via einsum: T[i,j,k] = sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
        # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
        Ur = Ua.reshape(n, m, r)
        Vr = Va.reshape(m, p, r)
        Wr = Wa.reshape(p, n, r)

        # T[i,j,k] = sum_l Ur[i,j,l] * Vr[j,k,l] * Wr[k,i,l]
        # This is a 3-way contraction. Compute as:
        # First compute A[i,j,k] = sum_l Ur[i,j,l] * Vr[j,k,l]  -> shape (n, m, p)
        A = np.einsum('ijl,jkl->ijk', Ur, Vr)
        # Then T[i,j,k] = sum_l A[i,j,l] * Wr[k,i,l]  -> but wait, that's not right.
        # Actually T[i,j,k] = sum_l Ur[i,j,l] * Vr[j,k,l] * Wr[k,i,l]
        # = sum_l (Ur[i,j,l] * Vr[j,k,l]) * Wr[k,i,l]
        # But A[i,j,k] already summed over l, so we can't just multiply by Wr.
        # We need the full 3-index contraction.
        # T[i,j,k] = sum_l Ur[i,j,l] * Vr[j,k,l] * Wr[k,i,l]
        T = np.einsum('ijl,jkl,kin->ijk', Ur, Vr, Wr)

        # T should be all ones (n x m x p)
        if not np.allclose(T, np.ones((n, m, p)), atol=1e-4):
            return False
        return True

    # First, return the seed as a fallback
    seed = make_seed()

    # Try to find a lower-rank decomposition
    # Known bounds: for 3x5x6 matmul, the tensor rank is at most 90.
    # The known lower bound is around 89 or so.
    # Let's try random searches for ranks below 90.

    best_result = seed
    best_rank = n * m * p

    # Try random initialization with small integer/half-integer values
    # Use values from {-1, -0.5, 0, 0.5, 1}
    values = [-1.0, -0.5, 0.0, 0.5, 1.0]

    # Try ranks from 89 down to maybe 85, with limited attempts each
    rng = np.random.default_rng(42)

    for rank in range(89, 84, -1):
        if time.time() > deadline - 20:
            break

        attempts = 0
        max_attempts = 200  # limited attempts per rank

        while attempts < max_attempts:
            attempts += 1
            if time.time() > deadline - 10:
                break

            # Random initialization
            U = np.random.choice(values, size=(n * m, rank), p=[0.15, 0.2, 0.3, 0.2, 0.15])
            V = np.random.choice(values, size=(m * p, rank), p=[0.15, 0.2, 0.3, 0.2, 0.15])
            W = np.random.choice(values, size=(p * n, rank), p=[0.15, 0.2, 0.3, 0.2, 0.15])

            # Quick check: compute the tensor contraction
            Ur = U.reshape(n, m, rank)
            Vr = V.reshape(m, p, rank)
            Wr = W.reshape(p, n, rank)
            T = np.einsum('ijl,jkl,kin->ijk', Ur, Vr, Wr)

            if np.allclose(T, np.ones((n, m, p)), atol=1e-4):
                # Found a valid decomposition!
                # Convert to lists with exact half-integer values
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]

                # Verify once more
                if validate(U_list, V_list, W_list):
                    result = {"U": U_list, "V": V_list, "W": W_list}
                    best_result = result
                    best_rank = rank
                    break
        # If we found one at this rank, stop (we want minimal rank)
        if best_rank < n * m * p:
            break

    return best_result
# EVOLVE-BLOCK-END
