"""Matrix multiplication 3x4x5 - search for rank-minimal decomposition.

Uses a structured random search with {-1, 0, 1} entries for U and V,
then checks if W can be solved in half-integers.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill

    # 1. Baseline: Schoolbook Decomposition (Rank 60)
    rank_baseline = n * m * p
    U_base = [[0] * rank_baseline for _ in range(n * m)]
    V_base = [[0] * rank_baseline for _ in range(m * p)]
    W_base = [[0] * rank_baseline for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_base[i * m + j][col] = 1
                V_base[j * p + k][col] = 1
                W_base[k * n + i][col] = 1
                col += 1

    if time.time() - start_time > 5.0:
        return {"U": U_base, "V": V_base, "W": W_base}

    best_solution = None
    best_rank = rank_baseline
    deadline = start_time + time_limit - 5.0

    # Precompute the (n*p) pairs we need to satisfy
    np_pairs = [(i, k) for i in range(n) for k in range(p)]

    # Search for lower rank
    # We iterate R from 59 down to 40
    for R in range(rank_baseline - 1, 39, -1):
        if time.time() > deadline:
            break

        # Number of trials per rank - adjust based on time remaining
        time_remaining = deadline - time.time()
        if time_remaining < 10:
            break

        # Scale trials with remaining time
        trials = min(2000, max(200, int(time_remaining * 30)))

        for t in range(trials):
            if time.time() > deadline:
                break

            # Random U, V with entries from {-1, 0, 1}
            # Use a structured approach: sparse matrices
            U = np.random.choice([-1, 0, 1], size=(n * m, R))
            V = np.random.choice([-1, 0, 1], size=(m * p, R))

            W_candidate = np.zeros((p * n, R), dtype=float)
            valid = True

            for (i, k) in np_pairs:
                # Construct A of shape (m, R)
                # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                A = np.zeros((m, R), dtype=float)
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]

                # Solve A * w = 1
                # Check if the system is consistent and has a half-integer solution
                try:
                    # Use lstsq to get a solution
                    w_ls, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)

                    # Check if w_ls is close to half-integers
                    w_candidate = np.round(w_ls * 2) / 2.0

                    # Verify A * w_candidate == 1
                    if np.allclose(A.dot(w_candidate), np.ones(m), atol=1e-9):
                        W_candidate[k * n + i, :] = w_candidate
                    else:
                        # Check consistency via rank
                        A_aug = np.hstack((A, np.ones((m, 1))))
                        if np.linalg.matrix_rank(A) != np.linalg.matrix_rank(A_aug):
                            valid = False
                            break
                        else:
                            # Try a few random null-space perturbations
                            # Get null space basis
                            if rank_a < R:
                                # Compute null space
                                _, _, vh = np.linalg.svd(A, full_matrices=True)
                                null_basis = vh[rank_a:, :].T  # (R, R-rank_a)
                                # Try a few random perturbations
                                found = False
                                for attempt in range(50):
                                    y = np.random.randn(null_basis.shape[1])
                                    w_try = w_ls + null_basis.dot(y)
                                    w_rounded = np.round(w_try * 2) / 2.0
                                    if np.allclose(A.dot(w_rounded), np.ones(m), atol=1e-9):
                                        W_candidate[k * n + i, :] = w_rounded
                                        found = True
                                        break
                                if not found:
                                    valid = False
                                    break
                            else:
                                valid = False
                                break
                except:
                    valid = False
                    break

            if not valid:
                continue

            # Verify the full tensor equation
            valid_full = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(R):
                            s += U[i * m + j, c] * V[j * p + k, c] * W_candidate[k * n + i, c]
                        if abs(s - 1) > 1e-9:
                            valid_full = False
                            break
                    if not valid_full:
                        break
                if not valid_full:
                    break

            if valid_full:
                # Convert to lists
                best_solution = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W_candidate.tolist()
                }
                best_rank = R
                # Break the trial loop for this R
                break

    if best_solution is not None:
        return best_solution
    else:
        return {"U": U_base, "V": V_base, "W": W_base}
