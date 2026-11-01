"""Matrix multiplication 2,2,7 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 7]
START_TIME = time.time()
TIME_LIMIT = 130  # Leave margin before 150s

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 28
    
    U_dim = n * m  # 4
    V_dim = m * p  # 14
    W_dim = p * n  # 14
    
    # Build the matrix multiplication tensor
    T = np.zeros((U_dim, V_dim, W_dim), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition
    U_school = np.zeros((U_dim, rank_schoolbook))
    V_school = np.zeros((V_dim, rank_schoolbook))
    W_school = np.zeros((W_dim, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    def verify_fast(U, V, W, tol=1e-8):
        rank = U.shape[1]
        T_recon = np.zeros((U_dim, V_dim, W_dim), dtype=float)
        for b in range(V_dim):
            T_recon[:, b, :] = U @ np.diag(V[b, :]) @ W.T
        return np.allclose(T_recon, T, atol=tol)
    
    assert verify_fast(U_school, V_school, W_school), "Schoolbook verification failed"
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    
    # Try to find lower rank decompositions using gradient descent
    for target_rank in range(1, best_rank):
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break
        
        found = False
        for attempt in range(3):
            if time.time() - START_TIME > TIME_LIMIT - 10:
                break
            
            np.random.seed(target_rank * 1000 + attempt * 7 + 42)
            U_try = np.random.randn(U_dim, target_rank)
            V_try = np.random.randn(V_dim, target_rank)
            W_try = np.random.randn(W_dim, target_rank)
            
            # Adaptive learning rate gradient descent
            lr = 0.05
            prev_loss = np.inf
            for step in range(1000):
                if time.time() - START_TIME > TIME_LIMIT - 5:
                    break
                
                # Compute residual and loss
                T_recon = np.zeros((U_dim, V_dim, W_dim), dtype=float)
                for r in range(target_rank):
                    T_recon += np.einsum('a,b,c->abc', U_try[:, r], V_try[:, r], W_try[:, r])
                residual = T_recon - T
                loss = np.sum(residual**2)
                
                if loss < prev_loss * 0.99:
                    lr = min(lr * 1.5, 0.5)
                elif loss > prev_loss:
                    lr = lr * 0.5
                prev_loss = loss
                
                if loss < 1e-10:
                    break
                
                # Gradients
                grad_U = np.zeros_like(U_try)
                grad_V = np.zeros_like(V_try)
                grad_W = np.zeros_like(W_try)
                for r in range(target_rank):
                    grad_U[:, r] = np.einsum('abc,br,cr->a', residual, V_try[:, r], W_try[:, r])
                    grad_V[:, r] = np.einsum('abc,ar,cr->b', residual, U_try[:, r], W_try[:, r])
                    grad_W[:, r] = np.einsum('abc,ar,br->c', residual, U_try[:, r], V_try[:, r])
                
                # Update
                U_try -= lr * grad_U
                V_try -= lr * grad_V
                W_try -= lr * grad_W
            
            if verify_fast(U_try, V_try, W_try, tol=1e-6):
                best_rank = target_rank
                best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
                found = True
                break
        
        if found:
            break
    
    # Round to nearest half-integer
    def round_to_half(arr):
        return np.round(arr * 2) / 2
    
    U_out = round_to_half(best_U)
    V_out = round_to_half(best_V)
    W_out = round_to_half(best_W)
    
    # Verify after rounding; if fails, use unrounded
    if not verify_fast(U_out, V_out, W_out, tol=1e-6):
        U_out = best_U
        V_out = best_V
        W_out = best_W
    
    U_list = U_out.tolist()
    V_list = V_out.tolist()
    W_list = W_out.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

