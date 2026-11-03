"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill

    # Schoolbook rank = n*m*p = 54
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None

    # Verify a decomposition
    def verify(U, V, W, rank):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Schoolbook decomposition
    def schoolbook_decomp():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank

    # Laderman's algorithm for 3x3 matrix multiplication (rank 23)
    # C = A * B where A is 3x3, B is 3x3
    # U has shape (9, 23), V has shape (9, 23), W has shape (9, 23)
    # Index conventions for 3x3x3:
    #   U row: i*3 + j
    #   V row: j*3 + k
    #   W row: k*3 + i
    
    # Laderman's 23-term algorithm
    # This is a well-known result. Let me construct it carefully.
    # The 23 terms for 3x3x3 multiplication.
    
    # Actually, let me use a verified rank-23 construction for 3x3x3.
    # I'll use the standard Laderman algorithm.
    
    # Laderman's algorithm (23 multiplications):
    # Let A = [a b c; d e f; g h i]
    # Let B = [x y z; u v w; r s t]
    
    # The 23 products are:
    # 1.  (a+b+c)(x+u+r)
    # 2.  (a+b-c)(x+u-r)
    # 3.  (a-b+c)(x-u+r)
    # 4.  (a-b-c)(x-u-r)
    # 5.  (d+e+f)(u+v+s)
    # 6.  (d+e-f)(u+v-s)
    # 7.  (d-e+f)(u-v+s)
    # 8.  (d-e-f)(u-v-s)
    # 9.  (g+h+i)(r+s+t)
    # 10. (g+h-i)(r+s-t)
    # 11. (g-h+i)(r-s+t)
    # 12. (g-h-i)(r-s-t)
    # 13. (b+f+s)(y+w+t)
    # 14. (b+f-s)(y+w-t)
    # 15. (b-f+s)(y-w+t)
    # 16. (b-f-s)(y-w-t)
    # 17. (c+e+r)(z+v+s)
    # 18. (c+e-r)(z+v-s)
    # 19. (c-e+r)(z-v+s)
    # 20. (c-e-r)(z-v-s)
    # 21. (a+e+s)(x+v+t)
    # 22. (a+e-s)(x+v-t)
    # 23. (a-e+s)(x-v+t)
    
    # This is getting complex. Let me use a different approach.
    # I'll try to find a low-rank decomposition using structured search.
    
    # For 3x3x6, we can think of it as two 3x3x3 multiplications.
    # If we have rank r for 3x3x3, we get rank 2r for 3x3x6.
    # But we might do better by sharing computations.
    
    # Let me try a direct approach: use the fact that p=6 and try to find
    # a rank-minimal decomposition.
    
    # Strategy: Try ranks from 23*2=46 down to 20
    # For each rank, try multiple random initializations with gradient descent
    
    candidate_ranks = [40, 42, 44, 46, 48, 50, 52]
    
    for target_rank in candidate_ranks:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
            
        for init_idx in range(10):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + target_rank * 100 + init_idx)
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            eta = 0.05
            for iteration in range(300):
                if time.time() > deadline:
                    break
                    
                # Compute error
                error = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(target_rank):
                                val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            error[i,j,k] = 1.0 - val
                
                if np.all(np.abs(error) < 1e-9):
                    best_rank = target_rank
                    best_U = U.tolist()
                    best_V = V.tolist()
                    best_W = W.tolist()
                    break
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            err = error[i,j,k]
                            if abs(err) < 1e-12:
                                continue
                            for r in range(target_rank):
                                dU[i*m+j, r] += err * V[j*p+k, r] * W[k*n+i, r]
                                dV[j*p+k, r] += err * U[i*m+j, r] * W[k*n+i, r]
                                dW[k*n+i, r] += err * U[i*m+j, r] * V[j*p+k, r]
                
                U -= eta * dU
                V -= eta * dV
                W -= eta * dW
                
                if iteration % 30 == 0:
                    eta *= 0.95
            
            if best_rank <= target_rank:
                break
    
    if best_U is None:
        best_U, best_V, best_W, best_rank = schoolbook_decomp()
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
