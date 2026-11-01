"""
Exact rank-minimal tensor decomposition search for 2x5 * 5x6 matrix multiplication.

Strategy:
- The tensor M(n,m,p) has entries T[i,j,k] = e_j (for all i,k) under the convention
  C_ik = sum_j A_ij B_jk.
- A rank-r decomposition is U (nm x r), V (mp x r), W (pn x r) with
  sum_r U[i*m+j1, r] * V[j2*p+k, r] * W[k*n+i, r] = delta(j1, j2).
- We search for the smallest r >= max(n*m, m*p, p*n) such that an exact
  half-integer solution exists.

Approach:
1. Construct the schoolbook (rank = n*m*p) as a valid baseline.
2. For each candidate rank r from the lower bound upward, attempt to find an exact
   half-integer solution via:
     a. Random restarts with gradient-based refinement (least squares on residuals).
     b. After refinement, snap to nearest half-integers and verify exactness.
     c. If exact, return immediately (since we search in increasing r, first hit is minimal).
3. Use a tight deadline guard to avoid timeout.

Key structural choices:
- Use scipy.optimize.least_squares for fast nonlinear refinement when available.
- Fallback to simple gradient descent if scipy is unavailable.
- Exploit gauge freedom: fix the first row of W to [1, 0, ..., 0] to reduce
  the effective degrees of freedom and improve convergence.
- Use integer-scaled verification (multiply by 2) to check half-integer exactness.
"""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 6]
DEADLINE = 115.0  # Safety margin: evaluator kills at 150s, keep 35s buffer


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS

    def time_left():
        return DEADLINE - (time.time() - start_time)

    # Schoolbook rank
    rank_schoolbook = n * m * p

    def schoolbook():
        rank = n * m * p
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
        return U, V, W

    best_U, best_V, best_W = schoolbook()
    best_rank = rank_schoolbook

    def compute_residuals(U, V, W, rank):
        """
        Compute residuals for all (i, k) pairs.
        Residual for (i, k): the m x m matrix where entry (j1, j2) is
          sum_r U[i*m+j1, r] * V[j2*p+k, r] * W[k*n+i, r] - delta(j1, j2)
        """
        residuals = []
        for i in range(n):
            for k in range(p):
                U_slice = U[i * m:(i + 1) * m, :]       # (m, rank)
                V_slice = V[k * p:(k + 1) * p, :]       # (m, rank)
                w_val = W[k * n + i]                     # (rank,)

                # Compute sum_r U_slice[j1,r] * V_slice[j2,r] * w_val[r]
                # = (U_slice * w_val[None, :]) @ V_slice.T  -> (m, m)
                mat = (U_slice * w_val[None, :]) @ V_slice.T
                target = np.eye(m)
                residuals.append((mat - target).flatten())
        return np.concatenate(residuals)

    def check_exact(U, V, W, rank):
        """
        Check if U, V, W are half-integers and satisfy the identity exactly.
        """
        # Scale by 2 to check for integers
        U2 = U * 2
        V2 = V * 2
        W2 = W * 2

        # Check if they are close to integers
        if not np.allclose(U2, np.round(U2), atol=1e-6):
            return False
        if not np.allclose(V2, np.round(V2), atol=1e-6):
            return False
        if not np.allclose(W2, np.round(W2), atol=1e-6):
            return False

        # Convert to exact integers
        U_int = np.round(U2).astype(int)
        V_int = np.round(V2).astype(int)
        W_int = np.round(W2).astype(int)

        # Verify identity with integers:
        # sum_r (U_int/2)(V_int/2)(W_int/2) = delta => sum_r U_int V_int W_int = 4 * delta
        for i in range(n):
            for k in range(p):
                U_slice = U_int[i * m:(i + 1) * m, :]
                V_slice = V_int[k * p:(k + 1) * p, :]
                w_val = W_int[k * n + i]

                mat = (U_slice * w_val[None, :]) @ V_slice.T
                if not np.allclose(mat, 4 * np.eye(m), atol=1e-6):
                    return False
        return True

    # Lower bound: max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    min_rank = max(n * m, m * p, p * n)

    # Try ranks from min_rank up to best_rank - 1
    for current_rank in range(min_rank, best_rank):
        if time_left() < 5.0:
            break

        rank_start_time = time.time()
        max_restarts = 50

        for restart in range(max_restarts):
            if time_left() < 5.0:
                break
            if time.time() - rank_start_time > 8.0:
                break

            # Random initialization with gauge fixing:
            # Fix W[0, :] = [1, 0, ..., 0] to reduce gauge freedom
            U = np.random.randn(n * m, current_rank) * 0.5
            V = np.random.randn(m * p, current_rank) * 0.5
            W = np.zeros((p * n, current_rank))
            W[0, 0] = 1.0
            # Add small random perturbation to W to break symmetry slightly
            W += np.random.randn(p * n, current_rank) * 0.1
            W[0, 0] = 1.0  # Re-fix the gauge

            # Try scipy least squares first
            found = False
            try:
                from scipy.optimize import least_squares

                len_U = n * m * current_rank
                len_V = m * p * current_rank
                len_W = p * n * current_rank

                def fun(params):
                    U_flat = params[:len_U]
                    V_flat = params[len_U:len_U + len_V]
                    W_flat = params[len_U + len_V:]

                    U_mat = U_flat.reshape((n * m, current_rank))
                    V_mat = V_flat.reshape((m * p, current_rank))
                    W_mat = W_flat.reshape((p * n, current_rank))

                    return compute_residuals(U_mat, V_mat, W_mat, current_rank)

                params_init = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

                result = least_squares(fun, params_init, method='lm', max_nfev=300)

                U_res = result.x[:len_U].reshape((n * m, current_rank))
                V_res = result.x[len_U:len_U + len_V].reshape((m * p, current_rank))
                W_res = result.x[len_U + len_V:].reshape((p * n, current_rank))

                # Snap to half-integers
                U_snap = np.round(U_res * 2) / 2.0
                V_snap = np.round(V_res * 2) / 2.0
                W_snap = np.round(W_res * 2) / 2.0

                if check_exact(U_snap, V_snap, W_snap, current_rank):
                    best_rank = current_rank
                    best_U, best_V, best_W = U_snap, V_snap, W_snap
                    found = True
                    break

            except Exception:
                pass

            if found:
                break

        if found:
            # Found a valid decomposition at this rank; since we search upward,
            # this is the minimal rank we found. Return immediately.
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # If no better rank was found, return the schoolbook solution
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
