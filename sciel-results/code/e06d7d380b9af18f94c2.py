import numpy as np
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s evaluator timeout

    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Precompute the schoolbook decomposition (rank = n*m*p = 96)
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1

    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_schoolbook

    # Helper: verify that U,V,W exactly reproduce T (within tolerance)
    def verify(U, V, W, tol=1e-9):
        S = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(S, T, atol=tol, rtol=0)

    # Helper: try to find a decomposition of a given rank using random restarts + gradient descent
    # with periodic ALS polishing and a final high-precision polish pass.
    def search_rank(rank, num_restarts, steps_per_restart, seed_offset=0):
        """
        Search for a rank-`rank` decomposition.
        Uses random init + gradient descent, with periodic ALS (alternating least squares)
        to polish. Returns (U, V, W) if found, else None.
        """
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                return None

            seed = seed_offset + restart
            np.random.seed(seed)

            # Initialize with small random values
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5

            # Gradient descent with adaptive learning rate
            lr = 0.02
            for step in range(steps_per_restart):
                if time.time() > deadline - 5:
                    return None

                # Compute current tensor product
                S = np.einsum('ir,jr,kr->ijk', U, V, W)
                residual = T - S
                loss = np.sum(residual ** 2)

                # Early stop if loss is very small
                if loss < 1e-12:
                    if verify(U, V, W, tol=1e-9):
                        return U, V, W
                    break

                # Compute gradients
                grad_U = np.einsum('ijk,jr,kr->ir', residual, V, W)
                grad_V = np.einsum('ijk,ir,kr->jr', residual, U, W)
                grad_W = np.einsum('ijk,ir,jr->kr', residual, U, V)

                # Adaptive learning rate: scale by inverse of gradient norm
                gnorm = np.sqrt(np.sum(grad_U ** 2) + np.sum(grad_V ** 2) + np.sum(grad_W ** 2))
                if gnorm > 1e-10:
                    step_size = lr * 0.1 / (gnorm + 1e-8)
                else:
                    step_size = lr

                U -= step_size * grad_U
                V -= step_size * grad_V
                W -= step_size * grad_W

                # Periodic ALS polish every 50 steps
                if step % 50 == 0 and step > 0:
                    # ALS: fix V,W and solve for U (linear least squares)
                    # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
                    # For each i: T[i,:,:] = sum_r U[i,r] * (V[r,:] * W[r,:])
                    # Reshape: T_flat (n*m, m*p*p*n) = U (n*m, rank) @ M (rank, m*p*p*n)
                    # where M[r, j*p*n+k] = V[j,r] * W[k,r]
                    M_VW = np.zeros((rank, m * p * p * n))
                    for j in range(m * p):
                        for k in range(p * n):
                            M_VW[:, j * p * n + k] = V[j, :] * W[k, :]
                    T_reshaped = T.reshape(n * m, m * p * p * n)
                    # Solve U = T_reshaped @ M_VW^T (M_VW @ M_VW^T)^{-1}
                    try:
                        A = M_VW @ M_VW.T  # (rank, rank)
                        b = T_reshaped @ M_VW.T  # (n*m, rank)
                        U_new = np.linalg.solve(A + 1e-10 * np.eye(rank), b).T  # (n*m, rank)
                        U = U_new
                    except np.linalg.LinAlgError:
                        pass

                    # Similarly for V
                    M_UW = np.zeros((rank, n * m * p * n))
                    for i in range(n * m):
                        for k in range(p * n):
                            M_UW[:, i * p * n + k] = U[i, :] * W[k, :]
                    T_reshaped_v = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
                    try:
                        A = M_UW @ M_UW.T
                        b = T_reshaped_v @ M_UW.T
                        V_new = np.linalg.solve(A + 1e-10 * np.eye(rank), b).T
                        V = V_new
                    except np.linalg.LinAlgError:
                        pass

                    # Similarly for W
                    M_UV = np.zeros((rank, n * m * m * p))
                    for i in range(n * m):
                        for j in range(m * p):
                            M_UV[:, i * m * p + j] = U[i, :] * V[j, :]
                    T_reshaped_w = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)
                    try:
                        A = M_UV @ M_UV.T
                        b = T_reshaped_w @ M_UV.T
                        W_new = np.linalg.solve(A + 1e-10 * np.eye(rank), b).T
                        W = W_new
                    except np.linalg.LinAlgError:
                        pass

            # Final verification
            if verify(U, V, W, tol=1e-9):
                return U, V, W

        return None

    # Target ranks to try, from low to high
    target_ranks = [64, 72, 80, 88, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64]
    # Remove duplicates while preserving order
    seen = set()
    unique_ranks = []
    for r in target_ranks:
        if r not in seen and r < best_rank:
            seen.add(r)
            unique_ranks.append(r)

    # Allocate time budget across ranks
    # Try harder for lower ranks, but with limited restarts
    for idx, rank in enumerate(unique_ranks):
        if time.time() > deadline - 10:
            break

        # Decreasing restarts for higher ranks (less likely to find)
        num_restarts = max(3, 15 - idx * 2)
        steps_per_restart = max(200, 1000 - idx * 30)

        print(f"Trying rank {rank} with {num_restarts} restarts, {steps_per_restart} steps each")
        result = search_rank(rank, num_restarts, steps_per_restart)
        if result is not None:
            U, V, W = result
            # Try to round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            if verify(U_rounded, V_rounded, W_rounded, tol=1e-10):
                best_result = {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
            else:
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
            best_rank = rank
            print(f"Found rank {rank} decomposition!")
            break

    # If no better decomposition found, return schoolbook
    if best_rank == rank_schoolbook:
        best_result = {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }

    return best_result
# EVOLVE-BLOCK-END
