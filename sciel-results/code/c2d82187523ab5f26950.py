"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    deadline = time.time() + 95
    
    # Build target tensor T
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition (rank 48)
    r0 = n * m * p
    U0 = np.zeros((nm, r0), dtype=np.float64)
    V0 = np.zeros((mp, r0), dtype=np.float64)
    W0 = np.zeros((pn, r0), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * (m * p) + j * p + k
                U0[i*m+j, c] = 1.0
                V0[j*p+k, c] = 1.0
                W0[k*n+i, c] = 1.0
    
    best_rank = r0
    best_U, best_V, best_W = U0, V0, W0
    
    # Try to find a rank-47 decomposition
    # Strategy: use the fact that we can perturb the schoolbook and optimize
    
    def compute_tensor(U, V, W):
        r = U.shape[1]
        result = np.zeros((nm, mp, pn), dtype=np.float64)
        for t in range(r):
            result += np.outer(np.outer(U[:,t], V[:,t]), W[:,t])
        return result
    
    # Try rank 47
    if time.time() < deadline - 10:
        target_rank = 47
        found = False
        
        for restart in range(50):
            if time.time() > deadline - 5:
                break
            
            # Initialize close to schoolbook: drop one column and adjust
            # Pick a random column to drop
            drop_col = np.random.randint(0, r0)
            
            # Start with schoolbook minus one column
            U = np.delete(U0, drop_col, axis=1)
            V = np.delete(V0, drop_col, axis=1)
            W = np.delete(W0, drop_col, axis=1)
            
            # Add small random perturbation
            U += np.random.randn(nm, target_rank) * 0.1
            V += np.random.randn(mp, target_rank) * 0.1
            W += np.random.randn(pn, target_rank) * 0.1
            
            # Iterate: minimize ||T - tensor(U,V,W)||_F^2
            for it in range(200):
                if time.time() > deadline - 2:
                    break
                
                pred = compute_tensor(U, V, W)
                err = pred - T
                err_norm = np.sum(err**2)
                
                if err_norm < 1e-10:
                    break
                
                # Gradient update
                lr = 0.05
                for t in range(target_rank):
                    # dU/dt
                    grad_U = np.zeros(nm)
                    for a in range(nm):
                        grad_U[a] = 2 * np.dot(err[a, :, :], (V[:,t] * W[:,t]).flatten())
                    U[:,t] -= lr * grad_U
                    
                    # dV/dt
                    grad_V = np.zeros(mp)
                    for b in range(mp):
                        grad_V[b] = 2 * np.dot(err[:, b, :], (U[:,t] * W[:,t]).flatten())
                    V[:,t] -= lr * grad_V
                    
                    # dW/dt
                    grad_W = np.zeros(pn)
                    for c in range(pn):
                        grad_W[c] = 2 * np.dot(err[:, :, c], (U[:,t] * V[:,t]).flatten())
                    W[:,c] -= lr * grad_W
            
            # Check convergence
            pred = compute_tensor(U, V, W)
            if np.max(np.abs(pred - T)) < 1e-6:
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                pred_r = compute_tensor(U_r, V_r, W_r)
                if np.max(np.abs(pred_r - T)) < 1e-9:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_r, V_r, W_r
                    found = True
                    break
        
        if found:
            # Try rank 46
            if time.time() < deadline - 10:
                target_rank = 46
                for restart in range(30):
                    if time.time() > deadline - 5:
                        break
                    
                    drop_col1 = np.random.randint(0, r0)
                    drop_col2 = np.random.randint(0, r0)
                    if drop_col2 == drop_col1:
                        drop_col2 = (drop_col2 + 1) % r0
                    
                    cols_to_keep = [c for c in range(r0) if c != drop_col1 and c != drop_col2]
                    U = U0[:, cols_to_keep]
                    V = V0[:, cols_to_keep]
                    W = W0[:, cols_to_keep]
                    
                    U += np.random.randn(nm, target_rank) * 0.1
                    V += np.random.randn(mp, target_rank) * 0.1
                    W += np.random.randn(pn, target_rank) * 0.1
                    
                    for it in range(200):
                        if time.time() > deadline - 2:
                            break
                        
                        pred = compute_tensor(U, V, W)
                        err = pred - T
                        if np.sum(err**2) < 1e-10:
                            break
                        
                        lr = 0.05
                        for t in range(target_rank):
                            grad_U = np.zeros(nm)
                            for a in range(nm):
                                grad_U[a] = 2 * np.dot(err[a, :, :], (V[:,t] * W[:,t]).flatten())
                            U[:,t] -= lr * grad_U
                            
                            grad_V = np.zeros(mp)
                            for b in range(mp):
                                grad_V[b] = 2 * np.dot(err[:, b, :], (U[:,t] * W[:,t]).flatten())
                            V[:,t] -= lr * grad_V
                            
                            grad_W = np.zeros(pn)
                            for c in range(pn):
                                grad_W[c] = 2 * np.dot(err[:, :, c], (U[:,t] * V[:,t]).flatten())
                            W[:,c] -= lr * grad_W
                    
                    pred = compute_tensor(U, V, W)
                    if np.max(np.abs(pred - T)) < 1e-6:
                        U_r = np.round(U * 2) / 2
                        V_r = np.round(V * 2) / 2
                        W_r = np.round(W * 2) / 2
                        
                        pred_r = compute_tensor(U_r, V_r, W_r)
                        if np.max(np.abs(pred_r - T)) < 1e-9:
                            best_rank = target_rank
                            best_U, best_V, best_W = U_r, V_r, W_r
                            break
    
    def to_list(mat):
        return [[float(mat[i][j]) for j in range(mat.shape[1])] for i in range(mat.shape[0])]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
