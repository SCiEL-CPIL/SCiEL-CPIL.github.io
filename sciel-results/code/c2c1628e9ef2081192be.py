"""Rewritten program: attempts rank reduction for 4x4 * 4x9 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]
n, m, p = DIMENSIONS

# Deadline guard
DEADLINE = 135  # seconds, safety margin from 150s limit

def _time_left():
    return DEADLINE - time.time()

def construct():
    start_time = time.time()
    
    # Initialize with schoolbook decomposition
    rank_schoolbook = n * m * p
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # The minimum possible rank is at least max(n*m, m*p, p*n) = max(16, 36, 36) = 36
    # But we can likely do better than 144 (schoolbook)
    
    # For matrix multiplication, known lower bounds suggest rank around n*m*p^(2/3) or similar
    # Let's try random initialization and optimization
    
    target_ranks = [80, 70, 60, 55, 50, 45, 42, 40, 38, 36]
    
    for rank in target_ranks:
        if _time_left() < 30:
            break
            
        # Try multiple random restarts
        max_restarts = 50 if rank > 50 else 100
        best_loss_for_rank = float('inf')
        best_solution_for_rank = None
        
        for restart in range(max_restarts):
            if _time_left() < 15:
                break
                
            # Random initialization
            U_r = np.random.randn(n * m, rank)
            V_r = np.random.randn(m * p, rank)
            W_r = np.random.randn(p * n, rank)
            
            # Normalize to avoid scale issues
            U_r /= np.linalg.norm(U_r)
            V_r /= np.linalg.norm(V_r)
            W_r /= np.linalg.norm(W_r)
            
            # Use gradient-free optimization (Nelder-Mead style) or simple iterative approach
            # For tensor decomposition, we can use alternating minimization
            
            loss = float('inf')
            for iteration in range(200):
                if _time_left() < 10:
                    break
                    
                # Compute the tensor from current U, V, W
                # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                
                # Efficient computation using einsum
                U_reshaped = U_r.reshape(n, m, rank)
                V_reshaped = V_r.reshape(m, p, rank)
                W_reshaped = W_r.reshape(p, n, rank)
                
                # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
                # This is a bit tricky with einsum, let's compute directly
                T_computed = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_computed[i, j, k] = np.sum(U_r[i*m+j, :] * V_r[j*p+k, :] * W_r[k*n+i, :])
                
                # Target tensor: T_target[i,j,k] = 1 if this corresponds to a valid multiplication term
                # For matrix multiplication, the tensor is: T[i,j,k] = 1 if the (i,j) entry of A 
                # and (j,k) entry of B contribute to (i,k) entry of C
                # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
                # where i in [0,n), j in [0,m), k in [0,p)
                
                T_target = np.ones((n, m, p))
                
                # Compute loss
                loss = np.sum((T_computed - T_target) ** 2)
                
                if loss < best_loss_for_rank:
                    best_loss_for_rank = loss
                    best_solution_for_rank = (U_r.copy(), V_r.copy(), W_r.copy())
                
                # If loss is very small, we've found a good solution
                if loss < 1e-8:
                    break
                
                # Simple gradient step (alternating minimization)
                # Update U
                for i in range(n):
                    for j in range(m):
                        grad_U = np.zeros(rank)
                        for k in range(p):
                            grad_U += (T_computed[i, j, k] - T_target[i, j, k]) * V_r[j*p+k, :] * W_r[k*n+i, :]
                        U_r[i*m+j, :] -= 0.01 * grad_U
                        
                # Update V
                for j in range(m):
                    for k in range(p):
                        grad_V = np.zeros(rank)
                        for i in range(n):
                            grad_V += (T_computed[i, j, k] - T_target[i, j, k]) * U_r[i*m+j, :] * W_r[k*n+i, :]
                        V_r[j*p+k, :] -= 0.01 * grad_V
                        
                # Update W
                for k in range(p):
                    for i in range(n):
                        grad_W = np.zeros(rank)
                        for j in range(m):
                            grad_W += (T_computed[i, j, k] - T_target[i, j, k]) * U_r[i*m+j, :] * V_r[j*p+k, :]
                        W_r[k*n+i, :] -= 0.01 * grad_W
                
                # Periodic renormalization
                if iteration % 10 == 0:
                    U_r /= (np.linalg.norm(U_r) + 1e-10)
                    V_r /= (np.linalg.norm(V_r) + 1e-10)
                    W_r /= (np.linalg.norm(W_r) + 1e-10)
            
            # Check if we found a valid solution
            if best_loss_for_rank < 1e-6:
                U_sol, V_sol, W_sol = best_solution_for_rank
                best_U, best_V, best_W = U_sol, V_sol, W_sol
                best_rank = rank
                break
        
        # If we found a solution at this rank, try lower ranks
        if best_rank < rank:
            continue
    
    # Verify the best solution
    if best_rank < rank_schoolbook:
        U_reshaped = best_U.reshape(n, m, best_rank)
        V_reshaped = best_V.reshape(m, p, best_rank)
        W_reshaped = best_W.reshape(p, n, best_rank)
        
        T_computed = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_computed[i, j, k] = np.sum(best_U[i*m+j, :] * best_V[j*p+k, :] * best_W[k*n+i, :])
        
        T_target = np.ones((n, m, p))
        if np.sum((T_computed - T_target) ** 2) > 1e-4:
            # Solution is not valid, fall back to schoolbook
            best_U, best_V, best_W = U, V, W
            best_rank = rank_schoolbook
    
    # Convert to lists with integer/half-integer entries if possible
    # Round to nearest 0.5
    U_list = [[round(x * 2) / 2 for x in row] for row in best_U.tolist()]
    V_list = [[round(x * 2) / 2 for x in row] for row in best_V.tolist()]
    W_list = [[round(x * 2) / 2 for x in row] for row in best_W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
