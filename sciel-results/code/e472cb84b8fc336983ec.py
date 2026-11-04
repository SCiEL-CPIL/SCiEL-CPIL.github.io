import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    _start = time.time()
    _deadline = _start + 140.0
    
    n, m, p = DIMENSIONS
    
    def schoolbook():
        rank = n * m * p
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
    
    def verify(U, V, W, tol=1e-9):
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        rank = U_arr.shape[1]
        
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
                    if abs(s - 1.0) > max_err:
                        max_err = abs(s - 1.0)
        
        # Check off-diagonals
        for i in range(n):
            for j in range(m):
                for j_prime in range(m):
                    for k in range(p):
                        for i_prime in range(n):
                            for k_prime in range(p):
                                if i == i_prime and k == k_prime:
                                    continue
                                s = 0.0
                                for r in range(rank):
                                    s += U_3d[i, j, r] * V_3d[j_prime, k, r] * W_3d[k_prime, i_prime, r]
                                if abs(s) > max_err:
                                    max_err = abs(s)
        
        return max_err < tol

    def random_search(max_rank, num_trials=100):
        np.random.seed(42)
        
        for trial in range(num_trials):
            if time.time() > _deadline - 10:
                break
            
            U = np.random.randn(n * m, max_rank) * 0.1
            V = np.random.randn(m * p, max_rank) * 0.1
            W = np.random.randn(p * n, max_rank) * 0.1
            
            for step in range(50):
                if time.time() > _deadline - 10:
                    break
                
                # Compute gradient
                # This is complex, so we use a simple finite difference or just random perturbation
                # Given the time limit, random perturbation is safer.
                
                U_3d = U.reshape(n, m, max_rank)
                V_3d = V.reshape(m, p, max_rank)
                W_3d = W.reshape(p, n, max_rank)
                
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(max_rank):
                                s += U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
                            err += (s - 1.0) ** 2
                            
                # Off-diagonal error
                for i in range(n):
                    for j in range(m):
                        for j_prime in range(m):
                            for k in range(p):
                                for i_prime in range(n):
                                    for k_prime in range(p):
                                        if i == i_prime and k == k_prime:
                                            continue
                                        s = 0.0
                                        for r in range(max_rank):
                                            s += U_3d[i, j, r] * V_3d[j_prime, k, r] * W_3d[k_prime, i_prime, r]
                                        err += s ** 2
                                        
                if err < 1e-9:
                    return U.tolist(), V.tolist(), W.tolist()
                
                U += np.random.randn(n * m, max_rank) * 0.01
                V += np.random.randn(m * p, max_rank) * 0.01
                W += np.random.randn(p * n, max_rank) * 0.01
                
        return None

    # Try rank 10
    result = random_search(10, num_trials=50)
    if result is not None:
        U, V, W = result
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
            
    return schoolbook()
# EVOLVE-BLOCK-END
