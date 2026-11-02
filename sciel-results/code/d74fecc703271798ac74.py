"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin before 150s kill

    # Schoolbook decomposition of rank n*m*p
    def schoolbook():
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
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    # Theoretical lower bound
    lb = max(n * m, m * p, p * n)  # max(12, 20, 15) = 20

    # For 3x4x5 matrix multiplication, we try to find a lower rank decomposition
    # Using a structured approach based on known tensor decomposition techniques
    
    # The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1
    # We want to find U, V, W such that sum_t U[a,t]*V[b,t]*W[c,t] = T[a,b,c]
    
    # Strategy: Use a combination of:
    # 1. Try specific structured decompositions
    # 2. Use alternating minimization with better initialization
    
    # For small cases like 3x4x5, we can try to find decompositions with rank < 60
    # Known results suggest the rank might be around 40-50 for such dimensions
    
    # Try to find a decomposition using a more efficient approach
    # We'll use a variant of the "sum of products" approach
    
    # First, let's try a structured decomposition
    # For matrix multiplication, there are known lower bounds and constructions
    
    # Try ranks from lb to best_rank-1
    for r in range(lb, best_rank):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        found = False
        for trial in range(50):
            if time.time() > deadline - 20:
                break
            
            # Initialize with random values
            np.random.seed(42 + trial)
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Alternating least squares
            for it in range(500):
                if time.time() > deadline - 20:
                    break
                
                # Update U: for each row a, solve linear system
                # T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
                # For fixed a: T[a,:] = U[a,:] * (V * W element-wise)
                
                # Build the system: for each (b,c), we have
                # sum_t U[a,t] * V[b,t] * W[c,t] = T[a,b,c]
                
                # Let's vectorize: for fixed a, we need to solve
                # M_a * U[a,:] = T[a,:]
                # where M_a is a (m*p * p*n) x r matrix with
                # M_a[(b,c),t] = V[b,t] * W[c,t]
                
                # This is expensive, so let's use a gradient-based approach instead
                
                # Compute the current tensor
                # T_hat[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
                
                # Loss = 0.5 * sum_{a,b,c} (T_hat[a,b,c] - T[a,b,c])^2
                
                # Gradient with respect to U[a,t]:
                # dL/dU[a,t] = sum_{b,c} (T_hat[a,b,c] - T[a,b,c]) * V[b,t] * W[c,t]
                
                # This is still expensive. Let's try a different approach.
                
                # Use a simpler gradient descent
                # Compute T_hat efficiently
                # T_hat = sum_t U[:,t] outer V[:,t] outer W[:,t]
                
                # For efficiency, let's compute the loss and gradients
                # in a vectorized way
                
                # Actually, for small dimensions, let's just do direct computation
                T_hat = np.zeros((n * m, m * p, p * n))
                for t in range(r):
                    T_hat += np.einsum('a,b,c->abc', U[:, t], V[:, t], W[:, t])
                
                # Compute target tensor
                T_target = np.zeros((n * m, m * p, p * n))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            T_target[a, b, c] = 1
                
                # Compute loss
                loss = 0.5 * np.sum((T_hat - T_target) ** 2)
                
                if loss < 1e-10:
                    # Check if this is a valid decomposition
                    # Convert to integer/half-integer if possible
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Verify
                    T_check = np.zeros((n * m, m * p, p * n))
                    for t in range(r):
                        T_check += np.einsum('a,b,c->abc', U_rounded[:, t], V_rounded[:, t], W_rounded[:, t])
                    
                    if np.allclose(T_check, T_target, atol=1e-10):
                        # Found a valid decomposition
                        best = {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
                        best_rank = r
                        found = True
                        break
                
                # Compute gradients
                # dL/dU[a,t] = sum_{b,c} (T_hat[a,b,c] - T_target[a,b,c]) * V[b,t] * W[c,t]
                # dL/dV[b,t] = sum_{a,c} (T_hat[a,b,c] - T_target[a,b,c]) * U[a,t] * W[c,t]
                # dL/dW[c,t] = sum_{a,b} (T_hat[a,b,c] - T_target[a,b,c]) * U[a,t] * V[b,t]
                
                diff = T_hat - T_target
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for t in range(r):
                    # dU[:,t] = einsum('abc,b,c->a', diff, V[:,t], W[:,t])
                    dU[:, t] = np.einsum('abc,b,c->a', diff, V[:, t], W[:, t])
                    # dV[:,t] = einsum('abc,a,c->b', diff, U[:,t], W[:,t])
                    dV[:, t] = np.einsum('abc,a,c->b', diff, U[:, t], W[:, t])
                    # dW[:,t] = einsum('abc,a,b->c', diff, U[:,t], V[:,t])
                    dW[:, t] = np.einsum('abc,a,b->c', diff, U[:, t], V[:, t])
                
                # Gradient descent with adaptive learning rate
                lr = 0.01
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                if loss < 1e-12:
                    break
            
            if found:
                break
        
        if found:
            break
    
    return best
# EVOLVE-BLOCK-END
