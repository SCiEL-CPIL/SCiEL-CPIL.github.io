import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s to be safe under 150s limit
    
    # Schoolbook solution as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i*m*p + j*p + k
                U_sb[i*m+j, c] = 1
                V_sb[j*p+k, c] = 1
                W_sb[k*n+i, c] = 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank decompositions
    # Work in doubled space to handle half-integers
    # We want U, V, W with entries in {0, 0.5, 1, 1.5, ...}
    # So 2*U, 2*V, 2*W should be integers, and the product sum should equal 1
    # sum_c (2U[i,j,c]/2)*(2V[j,k,c]/2)*(2W[k,i,c]/2) = 1
    # => sum_c (2U[i,j,c])*(2V[j,k,c])*(2W[k,i,c]) = 8
    
    # Let's try random initializations and gradient descent
    np.random.seed(42)
    
    for target_rank in range(32, rank_sb):  # Start from lower bound
        if time.time() > deadline - 10:
            break
        
        # Try a few random initializations
        for trial in range(5):
            if time.time() > deadline - 5:
                break
            
            r = target_rank
            # Initialize with small random values
            scale = 0.5
            U = np.random.randn(n*m, r) * scale
            V = np.random.randn(m*p, r) * scale
            W = np.random.randn(p*n, r) * scale
            
            # Gradient descent to minimize sum of squared errors
            lr = 0.01
            for step in range(500):
                if time.time() > deadline - 5:
                    break
                
                # Compute residual
                errors = []
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            ui = U[i*m+j, :]
                            vk = V[j*p+k, :]
                            wi = W[k*n+i, :]
                            prod = np.sum(ui * vk * wi)
                            err = prod - 1.0
                            # Gradient of (err^2) w.r.t. U[i*m+j,c] = 2*err*vk[c]*wi[c]
                            grad_U[i*m+j, :] += 2 * err * vk * wi
                            grad_V[j*p+k, :] += 2 * err * ui * wi
                            grad_W[k*n+i, :] += 2 * err * ui * vk
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                if step % 50 == 0:
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                prod = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                                max_err = max(max_err, abs(prod - 1.0))
                    if max_err < 1e-6:
                        break
                
                # Adaptive learning rate
                if step % 100 == 0 and step > 0:
                    lr *= 0.5
            
            # Check if solution is valid
            valid = True
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        prod = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        max_err = max(max_err, abs(prod - 1.0))
            if max_err < 1e-4:
                # Try to round to half-integers
                # Check if 2*U, 2*V, 2*W are close to integers
                if (np.allclose(2*U, np.round(2*U), atol=1e-3) and
                    np.allclose(2*V, np.round(2*V), atol=1e-3) and
                    np.allclose(2*W, np.round(2*W), atol=1e-3)):
                    U_rounded = np.round(2*U) / 2
                    V_rounded = np.round(2*V) / 2
                    W_rounded = np.round(2*W) / 2
                    
                    # Verify rounded solution
                    max_err_r = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                prod = np.sum(U_rounded[i*m+j, :] * V_rounded[j*p+k, :] * W_rounded[k*n+i, :])
                                max_err_r = max(max_err_r, abs(prod - 1.0))
                    if max_err_r < 1e-6:
                        best_rank = r
                        best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                        break
        
        if best_rank == target_rank:
            break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
