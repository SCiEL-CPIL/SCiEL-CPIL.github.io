"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    start_time = time.time()
    deadline = start_time + 140
    
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
    
    def verify(U, V, W, num_trials=100):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        
        rng = np.random.RandomState(42)
        for trial in range(num_trials):
            A = rng.rand(n, m)
            B = rng.rand(m, p)
            C_expected = A @ B
            
            C_computed = np.zeros((n, p))
            for r in range(rank):
                u_val = np.dot(U_np[:, r], A.flatten())
                v_val = np.dot(V_np[:, r], B.flatten())
                w_vec = W_np[:, r]
                for idx in range(p * n):
                    k = idx // n
                    i = idx % n
                    C_computed[i, k] += u_val * v_val * w_vec[idx]
            
            if not np.allclose(C_computed, C_expected, atol=1e-8):
                return False
        return True
    
    def round_to_half(x):
        return round(2 * x) / 2
    
    def tensor_residual(U, V, W):
        n, m, p = DIMENSIONS
        residual = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    residual[i, j, k] = s - 1.0
        return residual
    
    def als_optimize(target_rank, seed, max_iters=100):
        """Alternating least squares optimization."""
        if time.time() - start_time > deadline:
            return None
        
        rng = np.random.RandomState(seed)
        
        U = rng.randn(n * m, target_rank) * 0.5
        V = rng.randn(m * p, target_rank) * 0.5
        W = rng.randn(p * n, target_rank) * 0.5
        
        min_residual = np.inf
        best_params = None
        
        for it in range(max_iters):
            if time.time() - start_time > deadline:
                break
            
            # Update U: for each (i,j), solve for U[i*m+j, :]
            for i in range(n):
                for j in range(m):
                    # We want to minimize sum_k (1 - sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r])^2
                    # This is: minimize ||b - A*u||^2 where u = U[i*m+j, :]
                    # A is (p x target_rank), b is (p x 1)
                    A_mat = np.zeros((p, target_rank))
                    b_vec = np.zeros(p)
                    for k in range(p):
                        for r in range(target_rank):
                            A_mat[k, r] = V[j * p + k, r] * W[k * n + i, r]
                        b_vec[k] = 1.0
                    
                    # Solve least squares
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    except:
                        pass
            
            # Update V: for each (j,k), solve for V[j*p+k, :]
            for j in range(m):
                for k in range(p):
                    A_mat = np.zeros((n, target_rank))
                    b_vec = np.zeros(n)
                    for i in range(n):
                        for r in range(target_rank):
                            A_mat[i, r] = U[i * m + j, r] * W[k * n + i, r]
                        b_vec[i] = 1.0
                    
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    except:
                        pass
            
            # Update W: for each (k,i), solve for W[k*n+i, :]
            for k in range(p):
                for i in range(n):
                    A_mat = np.zeros((m, target_rank))
                    b_vec = np.zeros(m)
                    for j in range(m):
                        for r in range(target_rank):
                            A_mat[j, r] = U[i * m + j, r] * V[j * p + k, r]
                        b_vec[j] = 1.0
                    
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    except:
                        pass
            
            # Compute residual
            res = tensor_residual(U, V, W)
            current_residual = np.sum(res ** 2)
            
            if current_residual < min_residual:
                min_residual = current_residual
                best_params = (U.copy(), V.copy(), W.copy())
            
            if current_residual < 1e-10:
                break
        
        if best_params is None:
            return None
        
        U, V, W = best_params
        
        # Round to half-integers
        U_r = np.array([[round_to_half(x) for x in row] for row in U])
        V_r = np.array([[round_to_half(x) for x in row] for row in V])
        W_r = np.array([[round_to_half(x) for x in row] for row in W])
        
        U_list = U_r.tolist()
        V_list = V_r.tolist()
        W_list = W_r.tolist()
        
        if verify(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Start with schoolbook
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find better decompositions
    for target_rank in range(best_rank - 1, max(best_rank - 15, 35), -1):
        if time.time() - start_time > deadline:
            break
        
        found = False
        for seed in range(30):
            if time.time() - start_time > deadline:
                break
            
            result = als_optimize(target_rank, seed)
            if result is not None:
                best_result = result
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
