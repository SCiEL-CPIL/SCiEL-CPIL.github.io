"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    Uses structured search and randomized linear algebra to find a low-rank decomposition.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0  # Safety margin before 120s limit

    def verify(U, V, W, rank):
        """Verify that the decomposition exactly reconstructs the multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    def _to_half_int(x):
        """Convert to nearest half-integer if close, otherwise round to integer."""
        half = round(x * 2) / 2.0
        if abs(x - half) < 1e-9:
            return half
        return round(x)

    def try_random_decomposition(rank, num_trials=50):
        """Try to find a valid decomposition of given rank using random initialization."""
        import numpy as np

        for trial in range(num_trials):
            if time.time() > deadline:
                break

            if rank < m:
                continue

            U = np.random.uniform(-1, 1, (n * m, rank))
            V = np.random.uniform(-1, 1, (m * p, rank))

            W = np.zeros((p * n, rank))
            valid = True

            for i in range(n):
                for k in range(p):
                    A = np.array([[U[i * m + j, r] * V[j * p + k, r] for r in range(rank)] for j in range(m)])
                    b = np.ones(m)

                    try:
                        if rank == m:
                            w = np.linalg.solve(A, b)
                        else:
                            w, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                            if np.linalg.norm(A @ w - b) > 1e-8:
                                valid = False
                                break
                        W[k * n + i] = w
                    except Exception:
                        valid = False
                        break

                if not valid:
                    break

            if valid:
                U_out = [[_to_half_int(x) for x in row] for row in U]
                V_out = [[_to_half_int(x) for x in row] for row in V]
                W_out = [[_to_half_int(x) for x in row] for row in W]
                if verify(U_out, V_out, W_out, rank):
                    return {"U": U_out, "V": V_out, "W": W_out}

        return None

    def try_structured_search(rank, num_trials=30):
        """Try structured patterns for low-rank decompositions."""
        import numpy as np

        for trial in range(num_trials):
            if time.time() > deadline:
                break

            if rank < m:
                continue

            # Use sparse random initialization
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))

            for r in range(rank):
                # Randomly assign non-zero entries
                for idx in range(n * m):
                    if random.random() < 0.3:
                        U[idx, r] = random.choice([-1, 1])
                for idx in range(m * p):
                    if random.random() < 0.3:
                        V[idx, r] = random.choice([-1, 1])
                for idx in range(p * n):
                    if random.random() < 0.3:
                        W[idx, r] = random.choice([-1, 1])

            if verify(U.tolist(), V.tolist(), W.tolist(), rank):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

        return None

    # Search from lower bound to schoolbook
    lower_bound = max(n * m, m * p, p * n)
    schoolbook_rank = n * m * p

    # First try structured search for ranks near the lower bound
    for rank in range(lower_bound, min(lower_bound + 5, schoolbook_rank + 1)):
        if time.time() > deadline:
            break
        result = try_structured_search(rank, num_trials=20)
        if result is not None:
            return result

    # Then try random search
    for rank in range(lower_bound, schoolbook_rank + 1):
        if time.time() > deadline:
            break
        result = try_random_decomposition(rank, num_trials=30)
        if result is not None:
            return result

    # Fallback: schoolbook decomposition
    rank = schoolbook_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
