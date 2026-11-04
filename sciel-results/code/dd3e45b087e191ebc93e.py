"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    start_time = time.time()
    deadline = start_time + 115  # 115s limit, leaving 5s margin

    # Trivial solution: rank = n*m*p = 90
    rank_trivial = n * m * p
    U_trivial = [[0] * rank_trivial for _ in range(nm)]
    V_trivial = [[0] * rank_trivial for _ in range(mp)]
    W_trivial = [[0] * rank_trivial for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j][col] = 1
                V_trivial[j * p + k][col] = 1
                W_trivial[k * n + i][col] = 1
                col += 1

    best_U, best_V, best_W = U_trivial, V_trivial, W_trivial
    best_rank = rank_trivial

    # Precompute index arrays for vectorized operations
    # For U: index i*m+j -> (i, j)
    i_idx_U = np.array([i for i in range(n) for j in range(m)])
    j_idx_U = np.array([j for i in range(n) for j in range(m)])
    
    # For V: index j*p+k -> (j, k)
    j_idx_V = np.array([j for j in range(m) for k in range(p)])
    k_idx_V = np.array([k for j in range(m) for k in range(p)])
    
    # For W: index k*n+i -> (k, i)
    k_idx_W = np.array([k for k in range(p) for i in range(n)])
    i_idx_W = np.array([i for k in range(p) for i in range(n)])
    
    # All (i,j,k) indices
    all_ijk = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    def compute_tensor(U, V, W, rank):
        """Compute the tensor T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]"""
        T = np.zeros((n, m, p))
        for r in range(rank):
            u_r = U[:, r].reshape(n, m)
            v_r = V[:, r].reshape(m, p)
            w_r = W[:, r].reshape(p, n)
            # Vectorize over k for each (i,j)
            for i in range(n):
                for j in range(m):
                    T[i, j, :] += u_r[i, j] * (v_r[j, :] * w_r[:, i].T)
        return T

    def compute_gradients(U, V, W, error, rank):
        """Compute gradients of loss with respect to U, V, W"""
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for r in range(rank):
            u_r = U[:, r].reshape(n, m)
            v_r = V[:, r].reshape(m, p)
            w_r = W[:, r].reshape(p, n)
            
            # dU[i*m+j, r] = sum_{k} error[i,j,k] * v_r[j,k] * w_r[k,i]
            for i in range(n):
                for j in range(m):
                    dU[i * m + j, r] = np.sum(error[i, j, :] * v_r[j, :] * w_r[:, i].T)
            
            # dV[j*p+k, r] = sum_{i} error[i,j,k] * u_r[i,j] * w_r[k,i]
            for j in range(m):
                for k in range(p):
                    dV[j * p + k, r] = np.sum(error[:, j, k] * u_r[:, j] * w_r[k, :])
            
            # dW[k*n+i, r] = sum_{j} error[i,j,k] * u_r[i,j] * v_r[j,k]
            for k in range(p):
                for i in range(n):
                    dW[k * n + i, r] = np.sum(error[i, :, k] * u_r[i, :] * v_r[:, k])
        
        return dU, dV, dW

    # Try a few candidate ranks
    candidate_ranks = [60, 65, 70, 75, 80, 85]

    for target_rank in candidate_ranks:
        if time.time() > deadline - 10:
            break
        if target_rank >= best_rank:
            continue

        # Try multiple random initializations
        for attempt in range(5):
            if time.time() > deadline - 5:
                break

            np.random.seed(42 + attempt * 1000 + target_rank)
            U = np.random.randn(nm, target_rank) * 0.5
            V = np.random.randn(mp, target_rank) * 0.5
            W = np.random.randn(pn, target_rank) * 0.5

            # Adaptive gradient descent
            lr = 0.01
            best_err = np.inf
            converged = False
            plateau_count = 0

            for step in range(300):
                if time.time() > deadline - 5:
                    break

                # Compute current tensor
                T_current = compute_tensor(U, V, W, target_rank)
                error = T_current - 1.0
                err_norm = np.linalg.norm(error)

                if err_norm < 1e-10:
                    converged = True
                    break

                # Adaptive learning rate
                if err_norm < best_err * 0.99:
                    lr = min(lr * 1.1, 0.1)
                    plateau_count = 0
                else:
                    plateau_count += 1
                    if plateau_count > 5:
                        lr *= 0.5
                        plateau_count = 0
                
                best_err = min(best_err, err_norm)

                # Compute gradients
                dU, dV, dW = compute_gradients(U, V, W, error, target_rank)

                # Gradient norm for scaling
                grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
                if grad_norm > 1e-10:
                    scale = 1.0 / grad_norm
                    U -= lr * scale * dU
                    V -= lr * scale * dV
                    W -= lr * scale * dW

            if converged:
                # Verify exactness and round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2

                # Check if rounded values still give exact decomposition
                T_check = compute_tensor(U_rounded, V_rounded, W_rounded, target_rank)

                if np.allclose(T_check, 1.0, atol=1e-9):
                    best_rank = target_rank
                    best_U = U_rounded.reshape(nm, target_rank).tolist()
                    best_V = V_rounded.reshape(mp, target_rank).tolist()
                    best_W = W_rounded.reshape(pn, target_rank).tolist()
                    break  # Found a valid decomposition for this rank, move to lower

    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
