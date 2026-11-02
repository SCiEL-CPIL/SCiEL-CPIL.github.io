import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = rank_school
    
    # Target Tensor T
    # T is in C x A* x B*
    # C index: k*n + i (0..p*n-1)
    # A index: i*m + j (0..n*m-1)
    # B index: j*p + k (0..m*p-1)
    
    T = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1
                
    # Try to find lower rank
    start_time = time.time()
    max_time = 100 # Leave 20s margin
    
    for R in range(48, rank_school):
        if time.time() - start_time > max_time:
            break
            
        # Try a few random initializations for this R
        for attempt in range(5):
            if time.time() - start_time > max_time:
                break
                
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # Simple gradient descent
            lr = 0.01
            for step in range(1000):
                if time.time() - start_time > max_time:
                    break
                
                # Compute error
                # E = T - sum_r u_r v_r w_r
                # This is slow in Python loops. Use einsum or matmul.
                # sum_r U[:, r] V[:, r] W[:, r]
                # We can reshape.
                
                # Approximate gradient
                # dE/du_r = v_r w_r^T
                # dE/dv_r = u_r w_r^T
                # dE/dw_r = u_r v_r^T
                
                # Compute current tensor approximation
                # approx = np.einsum('ir,jr,kr->ijk', U, V, W)
                # This is expensive for R=50.
                # 50 * 6 * 27 * 18 ~ 150k ops. 1000 steps = 150M ops. OK.
                
                approx = np.einsum('ir,jr,kr->ijk', U, V, W)
                error = T - approx
                err_norm = np.linalg.norm(error)
                
                if err_norm < 1e-6:
                    # Check if we can round to integers/half-integers
                    U_round = np.round(U * 2) / 2
                    V_round = np.round(V * 2) / 2
                    W_round = np.round(W * 2) / 2
                    approx_round = np.einsum('ir,jr,kr->ijk', U_round, V_round, W_round)
                    if np.linalg.norm(T - approx_round) < 1e-6:
                        best_U = U_round
                        best_V = V_round
                        best_W = W_round
                        best_rank = R
                        break
                else:
                    # Gradient step
                    # grad_U[r, i] = sum_jk error[i, j, k] V[j, r] W[k, r]
                    grad_U = np.einsum('ijk,jr,kr->ir', error, V, W)
                    grad_V = np.einsum('ijk,ir,kr->jr', error, U, W)
                    grad_W = np.einsum('ijk,ir,jr->kr', error, U, V)
                    
                    U -= lr * grad_U
                    V -= lr * grad_V
                    W -= lr * grad_W
            
            if best_rank == R:
                break
        
        if best_rank < rank_school:
            break

    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
