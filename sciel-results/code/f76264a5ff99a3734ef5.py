"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36
    
    # Schoolbook as fallback
    rank_schoolbook = n * m * p  # 144
    U_fb = [[0] * rank_schoolbook for _ in range(nm)]
    V_fb = [[0] * rank_schoolbook for _ in range(mp)]
    W_fb = [[0] * rank_schoolbook for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    # Try to find lower rank decompositions
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We want: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Try ranks from lower to higher
    for target_rank in range(40, min(rank_schoolbook + 1, 100)):
        if time.time() > deadline:
            break
            
        # Try several random initializations
        for trial in range(50):
            if time.time() > deadline:
                break
                
            # Initialize with random values
            U = np.random.randn(nm, target_rank) * 0.1
            V = np.random.randn(mp, target_rank) * 0.1
            W = np.random.randn(pn, target_rank) * 0.1
            
            # Gradient descent on the loss
            # Loss = sum_{i,j,k} (T[i,j,k] - sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r])^2
            lr = 0.001
            for step in range(500):
                if time.time() > deadline:
                    break
                    
                # Compute current product
                # C[i,k] = sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                C = np.zeros((n, p))
                for i in range(n):
                    for k in range(p):
                        s = 0.0
                        for j in range(m):
                            for r in range(target_rank):
                                s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        C[i,k] = s
                
                # Compute gradient
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for k in range(p):
                        err = 1.0 - C[i,k]
                        for j in range(m):
                            for r in range(target_rank):
                                grad_U[i*m+j, r] += err * V[j*p+k, r] * W[k*n+i, r]
                                grad_V[j*p+k, r] += err * U[i*m+j, r] * W[k*n+i, r]
                                grad_W[k*n+i, r] += err * U[i*m+j, r] * V[j*p+k, r]
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                if step % 50 == 0:
                    max_err = np.max(np.abs(C - 1.0))
                    if max_err < 1e-6:
                        # Found a valid decomposition
                        if target_rank < best_rank:
                            best_rank = target_rank
                            best_U = [[float(x) for x in row] for row in U]
                            best_V = [[float(x) for x in row] for row in V]
                            best_W = [[float(x) for x in row] for row in W]
                        break
            
            if target_rank < best_rank:
                break
    
    # Return the best found
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
