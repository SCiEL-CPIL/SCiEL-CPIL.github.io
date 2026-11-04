import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    start_time = time.time()
    deadline = start_time + 135  # safety margin

    # Build the target tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def verify_solution(U, V, W, T, tol=1e-9):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_approx, T, atol=tol)

    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1

    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    def try_kronecker_block(r_block, max_trials=50):
        """
        Try to find a rank-r_block decomposition for the 2x3x3 tensor,
        then lift it to a rank-3*r_block decomposition for the full 2x3x9 tensor.
        """
        # Build the 2x3x3 target tensor
        T_block = np.zeros((6, 9, 6))
        for i in range(2):
            for j in range(3):
                for l in range(3):
                    T_block[i*3+j, j*3+l, l*2+i] = 1.0

        for trial in range(max_trials):
            if time.time() > deadline - 10:
                return None

            seed = 70000 + r_block * 1000 + trial
            np.random.seed(seed)

            V = np.random.randn(9, r_block)
            W = np.random.randn(6, r_block)

            # Build the linear system matrix M: (54, r_block)
            M = np.zeros((54, r_block))
            for r_idx in range(r_block):
                M[:, r_idx] = np.kron(V[:, r_idx], W[:, r_idx]).flatten()

            if np.linalg.matrix_rank(M) < r_block:
                continue

            U = np.zeros((6, r_block))
            valid = True
            for a in range(6):
                target = T_block[a, :, :].flatten()
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(M, target, rcond=None)
                    U[a, :] = sol
                except:
                    valid = False
                    break

            if not valid:
                continue

            # Verify the block decomposition
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            if not np.allclose(T_approx, T_block, atol=1e-9):
                continue

            # Lift to full 2x3x9 tensor
            rank_full = 3 * r_block
            U_full = np.zeros((nm, rank_full))
            V_full = np.zeros((mp, rank_full))
            W_full = np.zeros((pn, rank_full))

            for l in range(3):  # l indexes the 3 blocks of 3 columns
                for r_idx in range(r_block):
                    R = l * r_block + r_idx
                    U_full[:, R] = U[:, r_idx]
                    for j in range(3):
                        for s in range(3):
                            k = 3*l + s
                            V_full[j*9 + k, R] = V[j*3 + s, r_idx]
                    for s in range(3):
                        for i in range(2):
                            k = 3*l + s
                            W_full[k*2 + i, R] = W[s*2 + i, r_idx]

            if verify_solution(U_full, V_full, W_full, T):
                return U_full, V_full, W_full

        return None

    # Try Kronecker-based approach with different block ranks
    for r_block in [5, 6, 7, 8, 9, 10, 11, 12]:
        if time.time() > deadline - 20:
            break
        result = try_kronecker_block(r_block, max_trials=30)
        if result is not None:
            U_full, V_full, W_full = result
            rank_full = 3 * r_block
            if rank_full < best_rank:
                best_rank = rank_full
                best_U, best_V, best_W = U_full, V_full, W_full
                break

    # Direct search on full tensor using linear algebra
    if time.time() < deadline - 15:
        lower_bound = max(n*p, m*p, n*m)  # 27
        for r in range(lower_bound, min(best_rank, lower_bound + 15)):
            if time.time() > deadline - 10:
                break

            found = False
            for trial in range(20):
                if time.time() > deadline - 5:
                    break

                np.random.seed(90000 + r * 1000 + trial)

                V = np.random.randn(mp, r)
                W = np.random.randn(pn, r)

                # Build the linear system: for each row a of U, solve for U[a, :]
                # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
                # For fixed a: T[a, :, :] = sum_r U[a, r] * (V[:, r] * W[:, r]).T reshaped
                # So we need: T[a].flatten() = sum_r U[a, r] * kron(V[:, r], W[:, r]).flatten()

                M = np.zeros((mp * pn, r))
                for r_idx in range(r):
                    M[:, r_idx] = np.kron(V[:, r_idx], W[:, r_idx]).flatten()

                if np.linalg.matrix_rank(M) < r:
                    continue

                U = np.zeros((nm, r))
                valid = True
                for a in range(nm):
                    target = T[a, :, :].flatten()
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(M, target, rcond=None)
                        U[a, :] = sol
                    except:
                        valid = False
                        break

                if not valid:
                    continue

                if verify_solution(U, V, W, T):
                    if r < best_rank:
                        best_rank = r
                        best_U, best_V, best_W = U, V, W
                        found = True
                    break

            if found:
                break

    # Round to half-integers and verify
    U_final = np.round(best_U * 2) / 2
    V_final = np.round(best_V * 2) / 2
    W_final = np.round(best_W * 2) / 2

    if not verify_solution(U_final, V_final, W_final, T):
        # Fall back to schoolbook
        U_final = U_sb
        V_final = V_sb
        W_final = W_sb

    # Convert to list format
    U_list = [[float(x) for x in row] for row in U_final]
    V_list = [[float(x) for x in row] for row in V_final]
    W_list = [[float(x) for x in row] for row in W_final]

    return {"U": U_list, "V": V_list, "W": W_list}
