import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130
    
    # Build the target tensor T
    # T[a, b, c] = 1 if A[i,j] * B[j,k] contributes to C[i,k]
    # a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0

    # Schoolbook decomposition (rank 45)
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
    
    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    # Try to find lower rank decompositions
    # Strategy: Random init + Local Search + Quantization
    for target_rank in range(44, 40, -1):  # Try 44 down to 41
        if time.time() > deadline - 10:
            break
        
        success = False
        attempts = 20  # Reduced attempts to save time for quantization
        for attempt in range(attempts):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(attempt * 1000 + target_rank)
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            def compute_error(U, V, W):
                T_approx = np.zeros((n*m, m*p, p*n))
                for r in range(target_rank):
                    T_approx += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
                return np.sum((T - T_approx)**2)
            
            current_error = compute_error(U, V, W)
            
            # Local search
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Perturb U
                U_new = U + np.random.randn(n*m, target_rank) * 0.1
                new_error = compute_error(U_new, V, W)
                if new_error < current_error:
                    U = U_new
                    current_error = new_error
                    
                # Perturb V
                V_new = V + np.random.randn(m*p, target_rank) * 0.1
                new_error = compute_error(U, V_new, W)
                if new_error < current_error:
                    V = V_new
                    current_error = new_error
                    
                # Perturb W
                W_new = W + np.random.randn(p*n, target_rank) * 0.1
                new_error = compute_error(U, V, W_new)
                if new_error < current_error:
                    W = W_new
                    current_error = new_error
            
            if current_error < 1e-6:
                # Attempt quantization to half-integers
                U_q = np.round(U * 2) / 2
                V_q = np.round(V * 2) / 2
                W_q = np.round(W * 2) / 2
                
                error_q = compute_error(U_q, V_q, W_q)
                if error_q < 1e-6:
                    success = True
                    best_rank = target_rank
                    best_U = U_q
                    best_V = V_q
                    best_W = W_q
                    break
        
        if success:
            break
    
    if best_rank < rank_school:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
