import time
import random
import numpy as np

def construct():
    deadline = time.time() + 140
    
    n, m, p = 4, 4, 5
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, r] = 1.0
                V_sb[j*p + k, r] = 1.0
                W_sb[k*n + i, r] = 1.0
                r += 1
    
    def verify_full(U, V, W, tol=1e-8):
        """Check if the decomposition exactly represents matrix multiplication."""
        # Check the "1" entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        
        # Check all "0" entries where at least one index differs
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Vary i
                    for i2 in range(n):
                        if i2 != i:
                            s = np.dot(U[i2*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            if abs(s) > tol:
                                return False
                    # Vary j
                    for j2 in range(m):
                        if j2 != j:
                            s = np.dot(U[i*m+j2, :], V[j2*p+k, :] * W[k*n+i, :])
                            if abs(s) > tol:
                                return False
                    # Vary k
                    for k2 in range(p):
                        if k2 != k:
                            s = np.dot(U[i*m+j, :], V[j*p+k2, :] * W[k2*n+i, :])
                            if abs(s) > tol:
                                return False
        
        return True
    
    def compute_loss(U, V, W):
        """Compute the total loss for the decomposition."""
        loss = 0.0
        # "1" entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    loss += (s - 1.0) ** 2
        
        # "0" entries (sample to save time)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        if i2 != i:
                            s = np.dot(U[i2*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            loss += s ** 2
                    for j2 in range(m):
                        if j2 != j:
                            s = np.dot(U[i*m+j2, :], V[j2*p+k, :] * W[k*n+i, :])
                            loss += s ** 2
                    for k2 in range(p):
                        if k2 != k:
                            s = np.dot(U[i*m+j, :], V[j*p+k2, :] * W[k2*n+i, :])
                            loss += s ** 2
        
        return loss
    
    def gradient_descent(U, V, W, max_iter=500, lr=0.01, deadline=None, tol=1e-10):
        """Perform gradient descent to refine the decomposition."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        rank = U.shape[1]
        
        prev_loss = compute_loss(U, V, W)
        
        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            # "1" entries gradients
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        err = s - 1.0
                        prod = V[j*p+k, :] * W[k*n+i, :]
                        dU[i*m+j, :] += 2 * err * prod
                        dV[j*p+k, :] += 2 * err * U[i*m+j, :] * W[k*n+i, :]
                        dW[k*n+i, :] += 2 * err * U[i*m+j, :] * V[j*p+k, :]
            
            # "0" entries gradients
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for i2 in range(n):
                            if i2 != i:
                                s = np.dot(U[i2*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                                prod = V[j*p+k, :] * W[k*n+i, :]
                                dU[i2*m+j, :] += 2 * s * prod
                                dV[j*p+k, :] += 2 * s * U[i2*m+j, :] * W[k*n+i, :]
                                dW[k*n+i, :] += 2 * s * U[i2*m+j, :] * V[j*p+k, :]
                        for j2 in range(m):
                            if j2 != j:
                                s = np.dot(U[i*m+j2, :], V[j2*p+k, :] * W[k*n+i, :])
                                prod = V[j2*p+k, :] * W[k*n+i, :]
                                dU[i*m+j2, :] += 2 * s * prod
                                dV[j2*p+k, :] += 2 * s * U[i*m+j2, :] * W[k*n+i, :]
                                dW[k*n+i, :] += 2 * s * U[i*m+j2, :] * V[j2*p+k, :]
                        for k2 in range(p):
                            if k2 != k:
                                s = np.dot(U[i*m+j, :], V[j*p+k2, :] * W[k2*n+i, :])
                                prod = V[j*p+k2, :] * W[k2*n+i, :]
                                dU[i*m+j, :] += 2 * s * prod
                                dV[j*p+k2, :] += 2 * s * U[i*m+j, :] * W[k2*n+i, :]
                                dW[k2*n+i, :] += 2 * s * U[i*m+j, :] * V[j*p+k2, :]
            
            # Update with learning rate
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            new_loss = compute_loss(U, V, W)
            
            if new_loss < tol:
                break
            if new_loss > prev_loss:
                # Backtrack
                lr *= 0.5
                U += lr * dU
                V += lr * dV
                W += lr * dW
            else:
                prev_loss = new_loss
                # Slowly increase learning rate if converging
                if it % 100 == 0 and it > 0:
                    lr *= 1.1
        
        return U, V, W
    
    # Try to find lower-rank decompositions
    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    np.random.seed(42)
    
    # Strategy: Try ranks from just below schoolbook down to a reasonable minimum
    # For 4x4 x 4x5, let's try down to rank 50 as a starting point
    min_rank_to_try = max(20, rank_sb - 30)
    
    for target_rank in range(rank_sb - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline - 20:
            break
        
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random values
            scale = (1.0 / target_rank) ** (1.0/3.0)
            U = np.random.randn(nm, target_rank) * scale
            V = np.random.randn(mp, target_rank) * scale
            W = np.random.randn(pn, target_rank) * scale
            
            # Run gradient descent
            U, V, W = gradient_descent(U, V, W, max_iter=300, lr=0.005, deadline=deadline - 10)
            
            # Verify
            if verify_full(U, V, W, tol=1e-6):
                # Do a high-precision polish
                U, V, W = gradient_descent(U, V, W, max_iter=200, lr=0.001, deadline=deadline - 5, tol=1e-12)
                
                if verify_full(U, V, W, tol=1e-8):
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                    break  # Found valid decomposition at this rank, try lower
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    # Final verification
    final_ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(best_rank):
                    s += U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r]
                if abs(s - 1.0) > 1e-8:
                    final_ok = False
                    break
            if not final_ok:
                break
        if not final_ok:
            break
    
    if not final_ok:
        # Fallback to schoolbook
        U_list = U_sb.tolist()
        V_list = V_sb.tolist()
        W_list = W_sb.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
