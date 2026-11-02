"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Index mappings
    # U: (n*m, rank), U[i*m + j, c]
    # V: (m*p, rank), V[j*p + k, c]
    # W: (p*n, rank), W[k*n + i, c]
    
    # Tensor: T[(i,j),(j,k),(i,k)] = 1, else 0
    # Decomposition: sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] = T
    
    def verify(U, V, W, tol=1e-10):
        r = U.shape[1]
        # Check ones
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], np.multiply(V[j*p+k, :], W[k*n+i, :]))
                    if abs(val - 1.0) > tol:
                        return False
        # Check zeros - mismatched j
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    if j == j2:
                        continue
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], np.multiply(V[j2*p+k, :], W[k*n+i, :]))
                        if abs(val) > tol:
                            return False
        # Check zeros - mismatched i
        for i in range(n):
            for i2 in range(n):
                if i == i2:
                    continue
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], np.multiply(V[j*p+k, :], W[k*n+i2, :]))
                        if abs(val) > tol:
                            return False
        # Check zeros - mismatched k
        for k in range(p):
            for k2 in range(p):
                if k == k2:
                    continue
                for i in range(n):
                    for j in range(m):
                        val = np.dot(U[i*m+j, :], np.multiply(V[j*p+k2, :], W[k2*n+i, :]))
                        if abs(val) > tol:
                            return False
        return True
    
    # Schoolbook
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = schoolbook()
    best = (U_sb, V_sb, W_sb)
    best_rank = n * m * p
    
    # Try to find lower rank
    # Use a more efficient approach: set up as a system and use random + local search
    
    def optimize_rank(r, n_restarts=10, max_iters=5000):
        np.random.seed(12345)
        
        for restart in range(n_restarts):
            if time.time() > deadline - 20:
                break
            
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            lr = 0.005
            prev_loss = float('inf')
            stuck = 0
            
            for it in range(max_iters):
                if time.time() > deadline - 20:
                    break
                
                # Compute loss and gradient efficiently
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                # Ones: T[(i,j),(j,k),(i,k)] = 1
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u = U[i*m+j, :]
                            v = V[j*p+k, :]
                            w = W[k*n+i, :]
                            prod = v * w
                            val = np.dot(u, prod)
                            err = 1.0 - val
                            grad_U[i*m+j, :] += -2 * err * prod
                            grad_V[j*p+k, :] += -2 * err * (u * w)
                            grad_W[k*n+i, :] += -2 * err * (u * v)
                
                # Zeros: j mismatch
                for i in range(n):
                    for j in range(m):
                        for j2 in range(m):
                            if j == j2:
                                continue
                            for k in range(p):
                                u = U[i*m+j, :]
                                v = V[j2*p+k, :]
                                w = W[k*n+i, :]
                                prod = v * w
                                val = np.dot(u, prod)
                                grad_U[i*m+j, :] += 2 * val * prod
                                grad_V[j2*p+k, :] += 2 * val * (u * w)
                                grad_W[k*n+i, :] += 2 * val * (u * v)
                
                # Zeros: i mismatch
                for i in range(n):
                    for i2 in range(n):
                        if i == i2:
                            continue
                        for j in range(m):
                            for k in range(p):
                                u = U[i*m+j, :]
                                v = V[j*p+k, :]
                                w = W[k*n+i2, :]
                                prod = v * w
                                val = np.dot(u, prod)
                                grad_U[i*m+j, :] += 2 * val * prod
                                grad_V[j*p+k, :] += 2 * val * (u * w)
                                grad_W[k*n+i2, :] += 2 * val * (u * v)
                
                # Zeros: k mismatch
                for k in range(p):
                    for k2 in range(p):
                        if k == k2:
                            continue
                        for i in range(n):
                            for j in range(m):
                                u = U[i*m+j, :]
                                v = V[j*p+k2, :]
                                w = W[k2*n+i, :]
                                prod = v * w
                                val = np.dot(u, prod)
                                grad_U[i*m+j, :] += 2 * val * prod
                                grad_V[j*p+k2, :] += 2 * val * (u * w)
                                grad_W[k2*n+i, :] += 2 * val * (u * v)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                if it % 200 == 0:
                    if verify(U, V, W):
                        # Try to snap to nice values
                        U2 = np.round(U * 2) / 2
                        V2 = np.round(V * 2) / 2
                        W2 = np.round(W * 2) / 2
                        if verify(U2, V2, W2):
                            return U2, V2, W2
                        return U, V, W
                
                # Adaptive learning rate
                loss = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], np.multiply(V[j*p+k, :], W[k*n+i, :]))
                            loss += (val - 1)**2
                if loss > prev_loss:
                    lr *= 0.5
                    stuck += 1
                    if stuck > 50:
                        break
                else:
                    stuck = 0
                prev_loss = loss
            
            if verify(U, V, W):
                U2 = np.round(U * 2) / 2
                V2 = np.round(V * 2) / 2
                W2 = np.round(W * 2) / 2
                if verify(U2, V2, W2):
                    return U2, V2, W2
                return U, V, W
        
        return None
    
    # Try ranks from 1 upward
    for r in range(1, best_rank):
        if time.time() > deadline - 20:
            break
        res = optimize_rank(r)
        if res is not None:
            U, V, W = res
            if verify(U, V, W):
                best = (U, V, W)
                best_rank = r
                break
    
    U, V, W = best
    U_list = [[float(U[i][j]) for j in range(U.shape[1])] for i in range(U.shape[0])]
    V_list = [[float(V[i][j]) for j in range(V.shape[1])] for i in range(V.shape[0])]
    W_list = [[float(W[i][j]) for j in range(W.shape[1])] for i in range(W.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
