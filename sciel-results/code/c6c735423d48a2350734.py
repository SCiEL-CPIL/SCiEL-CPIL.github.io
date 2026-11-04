import numpy as np
import time

def _verify_decomposition(U, V, W):
    """
    Verifies if U, V, W exactly decompose the 2x3x6 multiplication tensor.
    Returns True if valid.
    """
    n, m, p = 2, 3, 6
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0

    rank = U.shape[1]
    reconstructed = np.zeros_like(target)
    for r in range(rank):
        reconstructed += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])

    return np.allclose(reconstructed, target, atol=1e-9)


def construct():
    n, m, p = 2, 3, 6
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # Target tensor T[i*m+j, j*p+k, k*n+i] = 1 for all (i,j,k), 0 otherwise
    # Dimensions: (6, 18, 12)
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0

    def get_rank_33_solution():
        """Known rank-33 solution using hybrid schoolbook + Strassen blocks."""
        rank = 33
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        col = 0
        # Part 1: Schoolbook for A[:, 0] * B[0, :] (6 terms)
        for k in range(p):
            for i in range(n):
                U[i * m + 0, col] = 1.0
                V[0 * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

        # Part 2: Strassen-like for 2x2 blocks on columns (1,2) for each pair
        for t in range(3):
            k0 = 2 * t
            k1 = 2 * t + 1

            # M1
            U[0 * m + 1, col] += 1.0
            U[1 * m + 2, col] += 1.0
            V[1 * p + k0, col] += 1.0
            V[2 * p + k1, col] += 1.0
            W[k0 * n + 0, col] += 1.0
            W[k1 * n + 1, col] += 1.0
            col += 1

            # M2
            U[1 * m + 1, col] += 1.0
            U[1 * m + 2, col] += 1.0
            V[1 * p + k0, col] += 1.0
            W[k0 * n + 1, col] += 1.0
            W[k1 * n + 1, col] += -1.0
            col += 1

            # M3
            U[0 * m + 1, col] += 1.0
            V[1 * p + k1, col] += 1.0
            V[2 * p + k1, col] += -1.0
            W[k1 * n + 0, col] += 1.0
            W[k1 * n + 1, col] += 1.0
            col += 1

            # M4
            U[1 * m + 2, col] += 1.0
            V[2 * p + k0, col] += 1.0
            V[1 * p + k0, col] += -1.0
            W[k0 * n + 0, col] += 1.0
            W[k0 * n + 1, col] += 1.0
            col += 1

            # M5
            U[0 * m + 1, col] += 1.0
            U[0 * m + 2, col] += 1.0
            V[2 * p + k1, col] += 1.0
            W[k0 * n + 0, col] += -1.0
            W[k1 * n + 0, col] += 1.0
            col += 1

            # M6
            U[1 * m + 1, col] += 1.0
            U[0 * m + 1, col] += -1.0
            V[1 * p + k0, col] += 1.0
            V[1 * p + k1, col] += 1.0
            W[k1 * n + 1, col] += 1.0
            col += 1

            # M7
            U[0 * m + 2, col] += 1.0
            U[1 * m + 2, col] += -1.0
            V[2 * p + k0, col] += 1.0
            V[2 * p + k1, col] += 1.0
            W[k0 * n + 0, col] += 1.0
            col += 1

        return U, V, W

    def try_rank_32_structured():
        """
        Try to find a rank-32 solution by modifying the rank-33 solution.
        The idea: can we merge or eliminate one term?
        """
        if time.time() > deadline - 5:
            return None
        
        # Start from rank-33 and try to see if one term can be absorbed
        U33, V33, W33 = get_rank_33_solution()
        
        # Try removing each term and see if the remaining 32 can still span
        # This is a linear algebra question: is the 33rd vector in the span of the other 32?
        # If yes, we have rank 32.
        
        # Build the tensor as a matrix: flatten (m*p, p*n) -> (m*p*p*n,)
        # Each term r contributes u_r (6,) outer (v_r outer w_r) (18*12=216,)
        # So we have a 6 x 216 matrix for each term, and the sum equals target reshaped
        
        T_mat = target.reshape(n * m, m * p * p * n)
        
        # For each term r, the contribution is U[:,r] (6,1) * (V[:,r] outer W[:,r]) (1,216)
        # So T_mat = sum_r U[:,r] * X_r where X_r = (V[:,r] outer W[:,r]).T (216,1) -> actually X_r is (216,)
        
        # Let's form the matrix A where A[:, r] = V[:,r] outer W[:,r] (216,)
        # And B where B[:, r] = U[:,r] (6,)
        # Then T_mat = B @ A.T (6 x 216)
        
        A = np.zeros((m * p * p * n, 33))
        B = np.zeros((n * m, 33))
        
        for r in range(33):
            A[:, r] = np.kron(V33[:, r], W33[:, r])
            B[:, r] = U33[:, r]
        
        # Check if T_mat = B @ A.T
        # And check if any column of B is in the span of the others (given A)
        
        # Actually, let's check the rank of the combined system
        # The tensor rank is the minimum r such that T = sum_{r=1}^r u_r ⊗ v_r ⊗ w_r
        
        # Let's try a different approach: use the known structure and see if we can
        # find a rank-32 by solving a system of equations
        
        # For now, let's just verify the rank-33 solution is correct
        if _verify_decomposition(U33, V33, W33):
            pass  # Good, rank 33 works
        
        # Try to find rank 32 via random search with constraints
        # This is hard, so let's just return None for now
        return None

    # First, verify the rank-33 solution
    U33, V33, W33 = get_rank_33_solution()
    if not _verify_decomposition(U33, V33, W33):
        # Fallback: schoolbook rank 36
        rank = 36
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to find rank 32
    result_32 = try_rank_32_structured()
    if result_32 is not None:
        U, V, W = result_32
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Return rank 33
    return {"U": U33.tolist(), "V": V33.tolist(), "W": W33.tolist()}
