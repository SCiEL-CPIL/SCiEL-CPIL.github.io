import numpy as np
import time
import random

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_schoolbook

    # Precompute tensor indices
    tensor_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor_indices.append((a, b, c))

    def verify(U, V, W):
        """Verify exact decomposition."""
        rank = U.shape[1]
        for a, b, c in tensor_indices:
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            if abs(val - 1) > 1e-9:
                return False
        # Check all zero positions
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if i_a == i_c and j_a == j_b and k_b == k_c:
                        continue
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val) > 1e-9:
                        return False
        return True

    def try_rank_opt(rank, num_restarts=2):
        """Try to find a rank-R decomposition using optimization."""
        for restart in range(num_restarts):
            if time.time() > deadline:
                return None
            
            # Initialize
            U0 = np.random.uniform(-1, 1, (n * m, rank))
            V0 = np.random.uniform(-1, 1, (m * p, rank))
            W0 = np.random.uniform(-1, 1, (p * n, rank))
            params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            def obj(params):
                U = params[:n*m*rank].reshape(n*m, rank)
                V = params[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
                W = params[n*m*rank+m*p*rank:].reshape(p*n, rank)
                err = 0
                for a, b, c in tensor_indices:
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    err += (val - 1) ** 2
                # Sample zero positions
                for _ in range(100):
                    a = random.randint(0, n * m - 1)
                    b = random.randint(0, m * p - 1)
                    c = random.randint(0, p * n - 1)
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if i_a == i_c and j_a == j_b and k_b == k_c:
                        continue
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    err += val ** 2
                return err
            
            from scipy.optimize import minimize
            result = minimize(obj, params0, method='L-BFGS-B', options={'maxiter': 200})
            
            if result.fun < 1e-8:
                U = result.x[:n*m*rank].reshape(n*m, rank)
                V = result.x[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
                W = result.x[n*m*rank+m*p*rank:].reshape(p*n, rank)
                
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if verify(U_r, V_r, W_r):
                    return {"U": U_r, "V": V_r, "W": W_r}
        return None

    # Try ranks from 30 down to 25
    for rank in range(30, 24, -1):
        if time.time() > deadline:
            break
        result = try_rank_opt(rank, num_restarts=2)
        if result is not None:
            best_result = {"U": result["U"].tolist(), "V": result["V"].tolist(), "W": result["W"].tolist()}
            best_rank = rank
            break

    return best_result
# EVOLVE-BLOCK-END
