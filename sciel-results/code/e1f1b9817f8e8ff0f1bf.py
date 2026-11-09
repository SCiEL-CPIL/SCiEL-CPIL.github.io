import numpy as np
import time

def construct():
    n, m, p = 3, 4, 5
    start_time = time.time()
    deadline = 140.0
    
    # Target tensor T
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i
    # else 0
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    # Schoolbook fallback (rank 60)
    best_rank = n * m * p
    best_U = np.zeros((n*m, best_rank))
    best_V = np.zeros((m*p, best_rank))
    best_W = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j, col] = 1
                best_V[j*p+k, col] = 1
                best_W[k*n+i, col] = 1
                col += 1
    
    # Precompute mode unfoldings for ALS
    # T_(0): shape (n*m, m*p * p*n) - mode-0 unfolding
    T_0 = T.transpose(0, 1, 2).reshape(n*m, m*p*p*n)
    # T_(1): shape (m*p, n*m * p*n) - mode-1 unfolding  
    T_1 = T.transpose(1, 0, 2).reshape(m*p, n*m*p*n)
    # T_(2): shape (p*n, n*m * m*p) - mode-2 unfolding
    T_2 = T.transpose(2, 0, 1).reshape(p*n, n*m*m*p)
    
    def khatri_rao(A, B):
        """Khatri-Rao product of A (da, r) and B (db, r) -> (da*db, r)"""
        return np.kron(A, B).reshape(A.shape[0]*B.shape[0], A.shape[1])
    
    def compute_error(U, V, W):
        T_approx = np.einsum('ic,jc,kc->ijk', U, V, W, optimize=True)
        return np.sum((T - T_approx)**2)
    
    # Try ranks from 20 up to 59
    for r in range(20, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        for attempt in range(30):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(attempt * 7919 + r * 131)
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # ALS iterations
            prev_error = np.inf
            for it in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Update U: T_0 = U @ (V ⊙ W)^T
                VW = khatri_rao(V, W)  # (m*p*p*n, r)
                try:
                    U = np.linalg.lstsq(VW.T, T_0.T, rcond=None)[0].T
                except:
                    break
                
                # Update V: T_1 = V @ (U ⊙ W)^T
                U_W = khatri_rao(U, W)  # (n*m*p*n, r)
                try:
                    V = np.linalg.lstsq(U_W.T, T_1.T, rcond=None)[0].T
                except:
                    break
                
                # Update W: T_2 = W @ (U ⊙ V)^T
                U_V = khatri_rao(U, V)  # (n*m*m*p, r)
                try:
                    W = np.linalg.lstsq(U_V.T, T_2.T, rcond=None)[0].T
                except:
                    break
                
                # Check convergence
                if it % 5 == 0:
                    error = compute_error(U, V, W)
                    if error < 1e-6:
                        break
                    if abs(prev_error - error) < 1e-8:
                        break
                    prev_error = error
            
            # Check final error
            error = compute_error(U, V, W)
            if error < 1e-6:
                if r < best_rank:
                    best_rank = r
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
    
    # Return the best found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
