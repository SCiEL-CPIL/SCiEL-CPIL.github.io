import numpy as np
import time

def construct():
    n, m, p = 3, 5, 5
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin

    # Baseline: schoolbook rank
    rank0 = n * m * p
    U0 = np.zeros((n*m, rank0))
    V0 = np.zeros((m*p, rank0))
    W0 = np.zeros((p*n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    
    best_rank = rank0
    best_U = U0.copy()
    best_V = V0.copy()
    best_W = W0.copy()
    
    # Try to reduce rank
    for r in range(rank0 - 1, 59, -1):
        if time.time() > deadline - 5.0:
            break
            
        found = False
        # Number of restarts per rank
        # With vectorized ALS and normal equations, we can do more restarts.
        
        for restart in range(30):
            if time.time() > deadline - 5.0:
                break
                
            np.random.seed(12345 + r * 100 + restart)
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            prev_err = float('inf')
            
            for it in range(300):
                if time.time() > deadline - 5.0:
                    break
                    
                # Update W
                for i in range(n):
                    for k in range(p):
                        u_i = U[i*m:(i+1)*m, :]
                        v_k = V[np.arange(m)*p + k, :]
                        
                        A = (u_i[:, np.newaxis, :] * v_k[np.newaxis, :, :]).reshape(m*m, r)
                        b = np.eye(m).reshape(m*m)
                        
                        # Solve normal equations: (A^T A) w = A^T b
                        ATA = A.T @ A + 1e-9 * np.eye(r)
                        ATb = A.T @ b
                        w = np.linalg.solve(ATA, ATb)
                        W[k*n+i, :] = w
                        
                # Update U
                V_rearr = V.reshape(m, p, r).transpose(1, 0, 2) # shape (p, m, r)
                
                for i in range(n):
                    w_i = W[np.arange(p)*n + i, :]
                    
                    for j1 in range(m):
                        A = (w_i[:, np.newaxis, :] * V_rearr).reshape(p*m, r)
                        b = np.zeros(p*m)
                        b[np.arange(p)*m + j1] = 1.0
                        
                        ATA = A.T @ A + 1e-9 * np.eye(r)
                        ATb = A.T @ b
                        u = np.linalg.solve(ATA, ATb)
                        U[i*m+j1, :] = u
                        
                # Update V
                U_rearr = U.reshape(n, m, r)
                
                for j2 in range(m):
                    for k in range(p):
                        w_k = W[k*n:(k+1)*n, :]
                        
                        A = (w_k[:, np.newaxis, :] * U_rearr).reshape(n*m, r)
                        b = np.zeros(n*m)
                        b[np.arange(n)*m + j2] = 1.0
                        
                        ATA = A.T @ A + 1e-9 * np.eye(r)
                        ATb = A.T @ b
                        v = np.linalg.solve(ATA, ATb)
                        V[j2*p+k, :] = v
                        
                # Check convergence
                W_res = W.reshape(p, n, r)
                U_res = U.reshape(n, m, r)
                V_res = V.reshape(m, p, r)
                
                S = np.einsum('kir,ijr,jkr->ikji', W_res, U_res, V_res)
                Target = np.eye(m)[np.newaxis, np.newaxis, :, :]
                
                err = np.sum((S - Target)**2)
                
                if err < 1e-8:
                    break
                if err > prev_err * 1.1:
                    break
                prev_err = err
            
            # Snap to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            if _verify(U_r, V_r, W_r, n, m, p):
                best_rank = r
                best_U = U_r.copy()
                best_V = V_r.copy()
                best_W = W_r.copy()
                found = True
                break
                
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

def _verify(U, V, W, n, m, p):
    r = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    W_res = W.reshape(p, n, r)
    U_res = U.reshape(n, m, r)
    V_res = V.reshape(m, p, r)
    
    S = np.einsum('kir,ijr,jkr->ikji', W_res, U_res, V_res)
    Target = np.eye(m)[np.newaxis, np.newaxis, :, :]
    
    return np.allclose(S, Target, atol=1e-6)
