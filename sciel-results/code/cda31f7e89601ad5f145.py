"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 145  # safety margin
    n, m, p = DIMENSIONS
    
    # We want to find a tensor decomposition of matrix multiplication:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In the trace convention: W[k*n+i, c] = sum_j U[i*m+j, c] * V[j*p+k, c]
    
    # The multiplication tensor M has shape (n, m, p) where M[i,j,k] = 1
    # We want a low-rank decomposition where each rank-1 component is a product
    # of vectors. The tensor for (n,m,p) matrix mult is T[i,j,k] = 1 for all i,j,k.
    # But actually it's more subtle: C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor T[i,j,k] = 1 (it's the all-ones tensor in the "multiplication" sense)
    
    # Actually, the standard matrix multiplication tensor for (n x m) * (m x p) -> (n x p)
    # has entries T[i, j, k] = 1 where i in [0,n), j in [0,m), k in [0,p)
    # And the decomposition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    # This is a 3-way tensor of size n*m x m*p x p*n, but with the constraint that
    # for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # Key insight: This is equivalent to finding the tensor rank of the 
    # (n x m x p) all-ones multiplication tensor.
    
    # Known results:
    # For 2x2: rank 7 (Strassen)
    # For 2x4: rank 18 or so
    # Let's try a structured approach based on known fast matrix multiplication
    
    # Strategy: Use recursive block structure or known decompositions
    # For 2x4x7, let's try to find a good rank
    
    # Let's use a numerical optimization approach to find a low-rank decomposition
    # Start from a reasonable upper bound and try to reduce rank
    
    # Upper bound from schoolbook: n*m*p = 2*4*7 = 56
    # We'll try to find a lower rank decomposition
    
    # Approach: Use the fact that matrix multiplication tensor rank satisfies
    # R(n,m,p) <= R(n1,m1,p1) * R(n/m, m/m, p/p) for certain divisors
    # But 2,4,7 don't divide nicely.
    
    # Let's try a direct approach: parametrize with random initial values and
    # use gradient descent to find a rank-r decomposition
    
    best_result = None
    best_rank = None
    
    # Try increasing rank from a lower bound
    # Lower bound: max(n*p, m*p, n*m) for some cases, but actual lower bounds are complex
    # Let's try ranks from 10 up to schoolbook
    
    np.random.seed(42)
    
    for rank in range(10, 60):
        if time.time() > deadline - 30:
            break
        
        if best_result is not None and rank >= best_rank:
            break
        
        # Try multiple random initializations
        attempts = 3
        success = False
        
        for attempt in range(attempts):
            if time.time() > deadline - 15:
                break
            
            # Initialize U, V, W with random values
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Normalize to avoid numerical issues
            for c in range(rank):
                U[:, c] = U[:, c] / (np.linalg.norm(U[:, c]) + 1e-10)
                V[:, c] = V[:, c] / (np.linalg.norm(V[:, c]) + 1e-10)
                W[:, c] = W[:, c] / (np.linalg.norm(W[:, c]) + 1e-10)
            
            # Define the loss: for each (i,j,k), we want
            # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # Loss = sum over all i,j,k of (computed - 1)^2
            
            def compute_loss(U, V, W):
                loss = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(rank):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            loss += (s - 1.0) ** 2
                return loss / (n*m*p)
            
            def compute_grad(U, V, W):
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(rank):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            error = (s - 1.0) * (2.0 / (n*m*p))
                            for c in range(rank):
                                grad_U[i*m+j, c] += error * V[j*p+k, c] * W[k*n+i, c]
                                grad_V[j*p+k, c] += error * U[i*m+j, c] * W[k*n+i, c]
                                grad_W[k*n+i, c] += error * U[i*m+j, c] * V[j*p+k, c]
                
                return grad_U, grad_V, grad_W
            
            # Adam optimizer
            lr = 0.01
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            
            m_U = np.zeros_like(U)
            v_U = np.zeros_like(U)
            m_V = np.zeros_like(V)
            v_V = np.zeros_like(V)
            m_W = np.zeros_like(W)
            v_W = np.zeros_like(W)
            t = 0
            
            loss_history = []
            max_iters = 2000
            
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                
                t += 1
                grad_U, grad_V, grad_W = compute_grad(U, V, W)
                
                # Adam update
                m_U = beta1 * m_U + (1 - beta1) * grad_U
                v_U = beta2 * v_U + (1 - beta2) * grad_U ** 2
                mhat_U = m_U / (1 - beta1 ** t)
                vhat_U = v_U / (1 - beta2 ** t)
                
                m_V = beta1 * m_V + (1 - beta1) * grad_V
                v_V = beta2 * v_V + (1 - beta2) * grad_V ** 2
                mhat_V = m_V / (1 - beta1 ** t)
                vhat_V = v_V / (1 - beta2 ** t)
                
                m_W = beta1 * m_W + (1 - beta1) * grad_W
                v_W = beta2 * v_W + (1 - beta2) * grad_W ** 2
                mhat_W = m_W / (1 - beta1 ** t)
                vhat_W = v_W / (1 - beta2 ** t)
                
                lr_t = lr * np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                
                U -= lr_t * mhat_U / (np.sqrt(vhat_U) + eps)
                V -= lr_t * mhat_V / (np.sqrt(vhat_V) + eps)
                W -= lr_t * mhat_W / (np.sqrt(vhat_W) + eps)
                
                if it % 100 == 0:
                    loss = compute_loss(U, V, W)
                    loss_history.append(loss)
                
                # Early stopping if loss is small
                if len(loss_history) > 0 and loss_history[-1] < 1e-10:
                    break
                if len(loss_history) > 5 and loss_history[-1] > loss_history[-5]:
                    # Loss increasing, might need to restart
                    break
            
            final_loss = compute_loss(U, V, W)
            if final_loss < 1e-8:
                success = True
                break
        
        if success:
            best_rank = rank
            # Convert to lists of float, round small values
            U_list = [[round(float(U[i,j]), 10) if abs(U[i,j]) > 1e-10 else 0.0 for j in range(rank)] for i in range(n*m)]
            V_list = [[round(float(V[i,j]), 10) if abs(V[i,j]) > 1e-10 else 0.0 for j in range(rank)] for i in range(m*p)]
            W_list = [[round(float(W[i,j]), 10) if abs(W[i,j]) > 1e-10 else 0.0 for j in range(rank)] for i in range(p*n)]
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: schoolbook decomposition
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

# EVOLVE-BLOCK-END
