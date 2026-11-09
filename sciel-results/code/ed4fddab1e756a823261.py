import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS

    # Build the matrix multiplication tensor
    # M[i*m+j, j*p+k, k*n+i] = 1
    M = np.zeros((n*m, m*p, p*n))
    nz_idx = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx1 = i*m + j
                idx2 = j*p + k
                idx3 = k*n + i
                M[idx1, idx2, idx3] = 1.0
                nz_idx.append((idx1, idx2, idx3))
    nz_idx = np.array(nz_idx)

    # Precompute flattened views for ALS
    M_flat = M.reshape(n*m, m*p * p*n)
    M_T_flat = M.T.reshape(m*p, n*m*p*n)
    M_P_flat = M.reshape(n*m*m*p, p*n).T

    # Lower bound: for matrix multiplication (n,m,p), rank >= max(n*p, m*n, m*p)
    # Also rank >= ceil(n*m*p / something) but the standard lower bound is max(n*p, m*n, m*p)
    # For 3x4x6: max(18, 12, 24) = 24
    lower_bound = max(n*p, m*n, m*p)

    # The schoolbook rank is n*m*p = 72
    # We want to find the minimum rank
    # Try ranks from lower_bound up to 72

    # For the ALS approach, we need to be careful about the tensor structure.
    # The key insight is that for matrix multiplication, the tensor has a special structure.
    # Let's try a different approach: use the known structure to guide the search.

    # Actually, let's try a smarter approach. For matrix multiplication C = AB where
    # A is n x m, B is m x p, C is n x p.
    # The tensor is T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r] where
    # i indexes rows of A (0..n-1), j indexes columns of A / rows of B (0..m-1),
    # k indexes columns of B (0..p-1).
    #
    # But our indexing is different:
    # M[idx1, idx2, idx3] = 1 where
    # idx1 = i*m + j (i in 0..n-1, j in 0..m-1)
    # idx2 = j*p + k (j in 0..m-1, k in 0..p-1)
    # idx3 = k*n + i (k in 0..p-1, i in 0..n-1)
    #
    # So U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r should equal 1 when
    # the indices are consistent (i,j,k form a valid triple) and 0 otherwise.
    #
    # The "otherwise" part: for any (idx1, idx2, idx3) not in nz_idx, the sum should be 0.

    best_result = None
    best_rank = None

    # Try ranks from lower_bound to 72
    for rank in range(lower_bound, 73):
        if time.time() > deadline - 15:
            break

        found = False
        # More restarts for smaller ranks
        max_restarts = 30 if rank <= 30 else 15 if rank <= 40 else 8

        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break

            seed = rank * 10000 + restart * 137 + 42
            np.random.seed(seed)

            # Initialize with random matrices
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)

            # Scale to avoid overflow/underflow
            U /= np.sqrt(rank)
            V /= np.sqrt(rank)
            W /= np.sqrt(rank)

            converged = False
            prev_residual = np.inf
            plateau_count = 0

            for iteration in range(80):
                if time.time() > deadline - 5:
                    break

                # ALS: update U given V, W
                # U[i, r] * sum_r' V[j, r'] * W[k, r'] ... this is not quite right.
                # Actually, for each (idx1, idx2, idx3) in the support:
                # sum_r U[idx1, r] * V[idx2, r] * W[idx3, r] = 1
                # For each idx1, we want to solve for U[idx1, :] such that
                # for all (idx2, idx3) paired with idx1, the constraint is satisfied.
                #
                # Let's do proper ALS:
                # Fix V and W. For each row idx1 of U, we have constraints:
                # sum_r U[idx1, r] * (V[idx2, r] * W[idx3, r]) = 1 for each (idx2, idx3)
                # Let G[idx2, idx3, r] = V[idx2, r] * W[idx3, r]
                # Then for a fixed idx1, we need U[idx1, :] * G[idx2, idx3, :] = 1
                # for all valid (idx2, idx3) paired with idx1.
                #
                # This is a linear system. Let's build it properly.

                # For U update:
                # For each idx1, collect all (idx2, idx3) such that M[idx1, idx2, idx3] != 0
                # But we also need to enforce that for (idx2, idx3) where M[idx1, idx2, idx3] == 0,
                # the sum should be 0. However, since we're doing least squares, we only use the
                # constraints where M != 0.

                # Actually, the standard approach is to use all non-zero entries.
                # For U: fix V, W. For each idx1, the constraints are:
                # sum_r U[idx1, r] * V[idx2, r] * W[idx3, r] = 1 for each (idx2, idx3) in support(idx1)

                # Let's build the system for all idx1 at once.
                # For each idx1, let S(idx1) = {(idx2, idx3) : M[idx1, idx2, idx3] != 0}
                # Then we need U[idx1, :] to satisfy:
                # sum_r U[idx1, r] * V[idx2, r] * W[idx3, r] = 1 for all (idx2, idx3) in S(idx1)

                # This is a linear system in U[idx1, :].
                # Let A[idx1, (idx2,idx3), r] = V[idx2, r] * W[idx3, r]
                # Then A[idx1] * U[idx1, :] = 1

                # Build A as a matrix of shape (total_constraints, rank)
                # where total_constraints = sum over idx1 of |S(idx1)| = n*m*p (each triple counted once)

                # Actually, each (idx1, idx2, idx3) in nz_idx gives one constraint.
                # For U update, fix V and W. The constraint is:
                # sum_r U[idx1, r] * V[idx2, r] * W[idx3, r] = 1
                # This is linear in U[idx1, :].

                # Let's build the matrix for U:
                # For each constraint (idx1, idx2, idx3), the coefficient of U[idx1, r] is V[idx2, r] * W[idx3, r]
                # So we have a block diagonal structure.

                # For efficiency, let's solve each block separately.
                i_vals = nz_idx[:, 0]
                j_vals = nz_idx[:, 1]
                k_vals = nz_idx[:, 2]

                # For U update:
                U_new = np.zeros((n*m, rank))
                for idx1 in range(n*m):
                    mask = (i_vals == idx1)
                    if not np.any(mask):
                        continue
                    j_sub = j_vals[mask]
                    k_sub = k_vals[mask]
                    # A_sub[r, c] = V[j_sub[c], r] * W[k_sub[c], r]
                    A_sub = V[j_sub, :] * W[k_sub, :]  # shape (n_sub, rank)
                    # Solve A_sub.T @ u = 1 (least squares)
                    try:
                        u, _, _, _ = np.linalg.lstsq(A_sub.T, np.ones(len(j_sub)), rcond=None)
                        U_new[idx1, :] = u
                    except:
                        pass

                # For V update:
                V_new = np.zeros((m*p, rank))
                for idx2 in range(m*p):
                    mask = (j_vals == idx2)
                    if not np.any(mask):
                        continue
                    i_sub = i_vals[mask]
                    k_sub = k_vals[mask]
                    A_sub = U[i_sub, :] * W[k_sub, :]
                    try:
                        v, _, _, _ = np.linalg.lstsq(A_sub.T, np.ones(len(i_sub)), rcond=None)
                        V_new[idx2, :] = v
                    except:
                        pass

                # For W update:
                W_new = np.zeros((p*n, rank))
                for idx3 in range(p*n):
                    mask = (k_vals == idx3)
                    if not np.any(mask):
                        continue
                    i_sub = i_vals[mask]
                    j_sub = j_vals[mask]
                    A_sub = U[i_sub, :] * V[j_sub, :]
                    try:
                        w, _, _, _ = np.linalg.lstsq(A_sub.T, np.ones(len(i_sub)), rcond=None)
                        W_new[idx3, :] = w
                    except:
                        pass

                U, V, W = U_new, V_new, W_new

                # Check residual on the support
                U_vals = U[i_vals, :]
                V_vals = V[j_vals, :]
                W_vals = W[k_vals, :]
                tensor_vals = np.sum(U_vals * V_vals * W_vals, axis=1)
                residual = np.abs(tensor_vals - 1.0).max()

                if residual < 1e-10:
                    converged = True
                    break

                if residual > prev_residual * 0.995:
                    plateau_count += 1
                    if plateau_count > 10:
                        break
                else:
                    plateau_count = 0
                prev_residual = residual

            if converged:
                # Verify full tensor
                T_check = (U[:, np.newaxis, np.newaxis, :] *
                           V[np.newaxis, :, np.newaxis, :] *
                           W[np.newaxis, np.newaxis, :, :])
                final_residual = np.abs(np.sum(T_check, axis=3) - M).max()

                if final_residual < 1e-8:
                    best_rank = rank
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    found = True
                    break

        if found:
            break

    if best_result is not None:
        return best_result
    else:
        # Fallback: schoolbook decomposition with rank = n*m*p
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
