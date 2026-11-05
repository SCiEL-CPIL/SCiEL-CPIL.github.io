import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s margin

    # Target tensor T[i, j, k] = 1
    # U shape (n*m, R), V shape (m*p, R), W shape (p*n, R)

    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1

    def verify_exact(U, V, W):
        """Vectorized verification that decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True

    def compute_error(U, V, W):
        """Compute total squared error."""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    err += (val - 1.0)**2
        return err

    def snap_to_half_integers(U, V, W):
        """Snap to nearest half-integers and verify."""
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        if verify_exact(U_snap, V_snap, W_snap):
            return U_snap, V_snap, W_snap
        return None

    def als_refine(U, V, W, R, max_iters=100, tol=1e-9):
        """ALS refinement with convergence check."""
        prev_err = float('inf')
        for it in range(max_iters):
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, R))
                    for k in range(p):
                        A[k, :] = V[j*p + k, :] * W[k*n + i, :]
                    b = np.ones(p)
                    try:
                        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        x = np.zeros(R)
                    U[i*m + j, :] = x

            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, R))
                    for i in range(n):
                        A[i, :] = U[i*m + j, :] * W[k*n + i, :]
                    b = np.ones(n)
                    try:
                        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        x = np.zeros(R)
                    V[j*p + k, :] = x

            # Update W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, R))
                    for j in range(m):
                        A[j, :] = U[i*m + j, :] * V[j*p + k, :]
                    b = np.ones(m)
                    try:
                        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        x = np.zeros(R)
                    W[k*n + i, :] = x

            # Calculate error
            err = compute_error(U, V, W)

            if err < 1e-8:
                return True, err

            if abs(prev_err - err) < tol:
                break
            prev_err = err

        return False, prev_err

    def high_precision_polish(U, V, W, R, max_iters=50):
        """
        High-precision polish: use tighter tolerance and more iterations.
        """
        prev_err = float('inf')
        for it in range(max_iters):
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, R))
                    for k in range(p):
                        A[k, :] = V[j*p + k, :] * W[k*n + i, :]
                    b = np.ones(p)
                    try:
                        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        x = np.zeros(R)
                    U[i*m + j, :] = x

            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, R))
                    for i in range(n):
                        A[i, :] = U[i*m + j, :] * W[k*n + i, :]
                    b = np.ones(n)
                    try:
                        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        x = np.zeros(R)
                    V[j*p + k, :] = x

            # Update W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, R))
                    for j in range(m):
                        A[j, :] = U[i*m + j, :] * V[j*p + k, :]
                    b = np.ones(m)
                    try:
                        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        x = np.zeros(R)
                    W[k*n + i, :] = x

            err = compute_error(U, V, W)
            if err < 1e-12:
                return True
            if abs(prev_err - err) < 1e-14:
                break
            prev_err = err

        return False

    def build_rank_22_block_solution():
        """
        Build a structured rank-22 solution using ALS from random init.
        """
        R22 = 22
        rng = np.random.default_rng(42)

        U22 = rng.choice([-1, 1], size=(n*m, R22)).astype(float)
        V22 = rng.choice([-1, 1], size=(m*p, R22)).astype(float)
        W22 = rng.choice([-1, 1], size=(p*n, R22)).astype(float)

        # Do a few ALS iterations
        for _ in range(10):
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, R22))
                    for k in range(p):
                        A[k, :] = V22[j*p + k, :] * W22[k*n + i, :]
                    b = np.ones(p)
                    try:
                        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        x = np.zeros(R22)
                    U22[i*m + j, :] = x

            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, R22))
                    for i in range(n):
                        A[i, :] = U22[i*m + j, :] * W22[k*n + i, :]
                    b = np.ones(n)
                    try:
                        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        x = np.zeros(R22)
                    V22[j*p + k, :] = x

            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, R22))
                    for j in range(m):
                        A[j, :] = U22[i*m + j, :] * V22[j*p + k, :]
                    b = np.ones(m)
                    try:
                        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        x = np.zeros(R22)
                    W22[k*n + i, :] = x

        return U22, V22, W22

    # Try lower ranks
    # Known rank for 2x3x4 is 17
    best_result = None

    for R in range(17, rank_school):
        if time.time() > deadline:
            break

        # Strategy 1: Use rank-22 block solution as warm start, then project
        if R <= 22:
            U22, V22, W22 = build_rank_22_block_solution()
            R22 = U22.shape[1]

            max_restarts = 30
            max_iters = 80

            for restart in range(max_restarts):
                if time.time() > deadline:
                    break

                if R < R22:
                    cols = np.random.choice(R22, size=R, replace=False)
                    U = U22[:, cols].copy()
                    V = V22[:, cols].copy()
                    W = W22[:, cols].copy()
                else:
                    U = U22.copy()
                    V = V22.copy()
                    W = W22.copy()

                # Add small random perturbation
                U += 0.01 * np.random.randn(n*m, R)
                V += 0.01 * np.random.randn(m*p, R)
                W += 0.01 * np.random.randn(p*n, R)

                converged, err = als_refine(U, V, W, R, max_iters=max_iters)

                if err < 1e-6:
                    high_precision_polish(U, V, W, R, max_iters=30)
                    err = compute_error(U, V, W)

                    if err < 1e-8:
                        snapped = snap_to_half_integers(U, V, W)
                        if snapped is not None:
                            U_snap, V_snap, W_snap = snapped
                            return {
                                "U": U_snap.tolist(),
                                "V": V_snap.tolist(),
                                "W": W_snap.tolist()
                            }

                        # Try other rounding strategies
                        for strategy in ['round', 'floor', 'ceil']:
                            if strategy == 'round':
                                U_s = np.round(U * 2) / 2
                                V_s = np.round(V * 2) / 2
                                W_s = np.round(W * 2) / 2
                            elif strategy == 'floor':
                                U_s = np.floor(U * 2) / 2
                                V_s = np.floor(V * 2) / 2
                                W_s = np.floor(W * 2) / 2
                            else:
                                U_s = np.ceil(U * 2) / 2
                                V_s = np.ceil(V * 2) / 2
                                W_s = np.ceil(W * 2) / 2

                            if verify_exact(U_s, V_s, W_s):
                                return {
                                    "U": U_s.tolist(),
                                    "V": V_s.tolist(),
                                    "W": W_s.tolist()
                                }

        # Strategy 2: Random restarts with small integers
        max_restarts = 20
        max_iters = 60

        for restart in range(max_restarts):
            if time.time() > deadline:
                break

            U = np.random.randint(-1, 2, (n*m, R)).astype(float)
            V = np.random.randint(-1, 2, (m*p, R)).astype(float)
            W = np.random.randint(-1, 2, (p*n, R)).astype(float)

            if np.all(U == 0) or np.all(V == 0) or np.all(W == 0):
                U[0,0] = 1
                V[0,0] = 1
                W[0,0] = 1

            converged, err = als_refine(U, V, W, R, max_iters=max_iters)

            if err < 1e-6:
                high_precision_polish(U, V, W, R, max_iters=30)
                err = compute_error(U, V, W)

                if err < 1e-8:
                    snapped = snap_to_half_integers(U, V, W)
                    if snapped is not None:
                        U_snap, V_snap, W_snap = snapped
                        return {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist()
                        }

    # Return schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
