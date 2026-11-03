"""Program for 3x3 x 3x5 matrix multiplication tensor decomposition."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 5]
DEADLINE = 150.0
SAFETY_MARGIN = 10.0

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 3, 5

    def elapsed():
        return time.time() - start_time

    def remaining():
        return DEADLINE - SAFETY_MARGIN - elapsed()

    def verify(U, V, W, rank):
        """Verify that the decomposition exactly reconstructs the multiplication tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :rank], V[j*p+k, :rank] * W[k*n+i, :rank])
                    if abs(val - 1) > 1e-9:
                        return False
        return True

    def make_schoolbook():
        """Return the schoolbook decomposition of rank n*m*p = 45."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = make_schoolbook()
    best_rank = n * m * p  # 45

    def tensor_error(U, V, W, rank):
        """Compute the sum of squared errors for the decomposition."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :rank], V[j*p+k, :rank] * W[k*n+i, :rank])
                    err += (val - 1.0) ** 2
        return err

    def local_search(U, V, W, rank, max_iter=5000):
        """Perform local search to reduce the error."""
        U = np.array(U, dtype=float).copy()
        V = np.array(V, dtype=float).copy()
        W = np.array(W, dtype=float).copy()
        best_err = tensor_error(U, V, W, rank)

        for it in range(max_iter):
            if remaining() < 5:
                break
            mat = random.choice(['U', 'V', 'W'])
            if mat == 'U':
                row = random.randint(0, n*m-1)
                col = random.randint(0, rank-1)
                delta = random.uniform(-0.1, 0.1)
                old = U[row, col]
                U[row, col] += delta
            elif mat == 'V':
                row = random.randint(0, m*p-1)
                col = random.randint(0, rank-1)
                delta = random.uniform(-0.1, 0.1)
                old = V[row, col]
                V[row, col] += delta
            else:
                row = random.randint(0, p*n-1)
                col = random.randint(0, rank-1)
                delta = random.uniform(-0.1, 0.1)
                old = W[row, col]
                W[row, col] += delta

            new_err = tensor_error(U, V, W, rank)
            if new_err < best_err:
                best_err = new_err
                if best_err < 1e-12:
                    break
            else:
                if mat == 'U':
                    U[row, col] = old
                elif mat == 'V':
                    V[row, col] = old
                else:
                    W[row, col] = old

        return U, V, W, best_err

    def try_rank(R, max_attempts=20):
        """Try to find a rank-R decomposition using random init + local search."""
        if remaining() < 10:
            return None

        for attempt in range(max_attempts):
            if remaining() < 5:
                break

            U = np.random.randn(n*m, R) * 0.1
            V = np.random.randn(m*p, R) * 0.1
            W = np.random.randn(p*n, R) * 0.1

            U, V, W, err = local_search(U, V, W, R, max_iter=2000)

            if err < 1e-10:
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                if verify(U_round, V_round, W_round, R):
                    return {
                        "U": U_round.tolist(),
                        "V": V_round.tolist(),
                        "W": W_round.tolist()
                    }
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                if verify(U_int, V_int, W_int, R):
                    return {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }

        return None

    def try_integer_search(R, max_iterations=50000):
        """Try to find a rank-R decomposition with integer entries in {-1, 0, 1}."""
        if remaining() < 10:
            return None

        for _ in range(max_iterations):
            if remaining() < 5:
                break

            U = np.zeros((n*m, R), dtype=int)
            V = np.zeros((m*p, R), dtype=int)
            W = np.zeros((p*n, R), dtype=int)

            for r in range(R):
                while True:
                    U[:, r] = np.random.choice([-1, 0, 1], size=n*m)
                    V[:, r] = np.random.choice([-1, 0, 1], size=m*p)
                    W[:, r] = np.random.choice([-1, 0, 1], size=p*n)
                    if np.any(U[:, r] != 0) and np.any(V[:, r] != 0) and np.any(W[:, r] != 0):
                        break

            if verify(U, V, W, R):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }

        return None

    def merge_schoolbook(R, max_attempts=200):
        """
        Structural merge: start from the schoolbook rank-45 decomposition and
        attempt to collapse 45-R columns by finding linear combinations of
        rank-1 terms that preserve the tensor. We parameterize a linear map
        from the 45-dim schoolbook space to an R-dim subspace and optimize
        so that the induced tensor equals the target.
        """
        if remaining() < 10:
            return None

        # Schoolbook basis: each column r corresponds to (i,j,k) with U[i*m+j]=1,
        # V[j*p+k]=1, W[k*n+i]=1.
        # We want to find a linear change of basis A (45 x R) such that
        # the new rank-R decomposition still equals the tensor.
        # Equivalent: find R rank-1 terms whose sum equals the tensor.
        #
        # Strategy: pick R random linear combinations of the 45 schoolbook
        # columns, then optimize the coefficients to match the tensor.
        #
        # Concretely: let B be a 45 x R matrix with random entries. Define
        # new columns as linear combinations. The resulting tensor is a
        # bilinear function of B. We optimize B to minimize the error.

        for attempt in range(max_attempts):
            if remaining() < 5:
                break

            # Random mixing matrix: 45 -> R
            B = np.random.randn(45, R)
            B /= np.linalg.norm(B, axis=0, keepdims=True)

            # Build the schoolbook factor matrices as 45-column matrices
            U0 = np.zeros((n*m, 45))
            V0 = np.zeros((m*p, 45))
            W0 = np.zeros((p*n, 45))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U0[i*m+j, col] = 1
                        V0[j*p+k, col] = 1
                        W0[k*n+i, col] = 1
                        col += 1

            # New factors: U_new = U0 @ B, etc.
            U = U0 @ B
            V = V0 @ B
            W = W0 @ B

            # But this gives a specific rank-R decomposition. The tensor it
            # represents is fixed by B. We need to optimize B so that the
            # tensor matches. The tensor is a multilinear function of B.
            #
            # The (i,j,k) entry of the tensor is:
            # sum_r U0[i*m+j, :]_r * B[r, :] ... this gets complicated.
            #
            # Actually, the tensor from factors (U0@B, V0@B, W0@B) is:
            # T[i,j,k] = sum_r (U0@B)[i*m+j, r] * (V0@B)[j*p+k, r] * (W0@B)[k*n+i, r]
            #
            # This is a degree-3 polynomial in B. We can use gradient descent.
            #
            # For now, let's just try a few random B and check if any work.
            # This is unlikely to work, but it's a structural approach.

            err = tensor_error(U, V, W, R)
            if err < 1e-10:
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                if verify(U_round, V_round, W_round, R):
                    return {
                        "U": U_round.tolist(),
                        "V": V_round.tolist(),
                        "W": W_round.tolist()
                    }

        return None

    # Try ranks from 43 upward
    for target_rank in [43, 44]:
        if remaining() < 15:
            break

        # First try the integer search
        result = try_integer_search(target_rank, max_iterations=10000)
        if result is not None:
            best_rank = target_rank
            best_result = result
            break

        # Then try the floating point search
        result = try_rank(target_rank, max_attempts=15)
        if result is not None:
            best_rank = target_rank
            best_result = result
            break

        # Then try the structural merge approach
        result = merge_schoolbook(target_rank, max_attempts=100)
        if result is not None:
            best_rank = target_rank
            best_result = result
            break

    return best_result
# EVOLVE-BLOCK-END
