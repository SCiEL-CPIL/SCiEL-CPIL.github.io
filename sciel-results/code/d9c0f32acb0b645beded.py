import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try different ranks
    best_result = None
    best_rank = n * m * p  # Schoolbook
    
    # Schoolbook
    U_school = np.zeros((n*m, best_rank))
    V_school = np.zeros((m*p, best_rank))
    W_school = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    best_result = (U_school, V_school, W_school)
    
    # Try to improve with lower rank
    for target_rank in [55, 50, 45, 40, 35, 30]:
        if time.time() - start_time > 100:
            break
        
        np.random.seed(target_rank)
        U = np.random.randn(n*m, target_rank) * 0.1
        V = np.random.randn(m*p, target_rank) * 0.1
        W = np.random.randn(p*n, target_rank) * 0.1
        
        converged = False
        for it in range(200):
            if time.time() - start_time > 110:
                break
            
            # Update U
            M = np.einsum('br,cr->bcr', V, W)  # (m*p, p*n, target_rank)
            M_flat = M.reshape(m*p*p*n, target_rank)
            T_flat = T.reshape(n*m, m*p*p*n).T
            U_new = np.linalg.lstsq(M_flat, T_flat, rcond=None)[0].T
            
            # Update V
            M = np.einsum('ar,cr->acr', U_new, W)
            M_flat = M.reshape(n*m*p*n, target_rank)
            T_flat = T.reshape(m*p, n*m*p*n).T
            V_new = np.linalg.lstsq(M_flat, T_flat, rcond=None)[0].T
            
            # Update W
            M = np.einsum('ar,br->abr', U_new, V_new)
            M_flat = M.reshape(n*m*m*p, target_rank)
            T_flat = T.reshape(p*n, n*m*m*p).T
            W_new = np.linalg.lstsq(M_flat, T_flat, rcond=None)[0].T
            
            U, V, W = U_new, V_new, W_new
        
        # Check if solution is good
        # Compute error
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.linalg.norm(recon - T) / np.linalg.norm(T)
        
        if error < 1e-6:
            best_rank = target_rank
            best_result = (U, V, W)
            break
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
