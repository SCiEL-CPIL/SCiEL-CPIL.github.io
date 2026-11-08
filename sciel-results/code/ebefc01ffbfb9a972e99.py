import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    pn = p * n  # 18
    nm = n * m  # 6
    mp = m * p  # 27
    
    # Build the tensor T[pn, nm, mp]
    T = np.zeros((pn, nm, mp))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k*n+i, i*m+j, j*p+k] = 1
    
    deadline = time.time() + 140
    
    # Try to find a low-rank decomposition
    best_rank = n*m*p
    best_U, best_V, best_W = None, None, None
    
    for target_rank in range(50, n*m*p):
        if time.time() > deadline - 5:
            break
        
        found = False
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            
            r = target_rank
            np.random.seed(trial * 1000 + r)
            
            W = np.random.randn(mp, r)
            U = np.random.randn(pn, r)
            V = np.random.randn(nm, r)
            
            for iteration in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Update U
                M_U = np.zeros((nm*mp, r))
                idx = 0
                for a in range(nm):
                    for b in range(mp):
                        M_U[idx, :] = V[a, :] * W[b, :]
                        idx += 1
                T_U = T.reshape(pn, nm*mp)
                U_new, _, _, _ = np.linalg.lstsq(M_U.T, T_U.T, rcond=None)
                U = U_new.T
                
                # Update V
                M_V = np.zeros((pn*mp, r))
                idx = 0
                for i in range(pn):
                    for b in range(mp):
                        M_V[idx, :] = U[i, :] * W[b, :]
                        idx += 1
                T_V = T.reshape(pn*mp, nm).T
                V_new, _, _, _ = np.linalg.lstsq(M_V.T, T_V.T, rcond=None)
                V = V_new.T
                
                # Update W
                M_W = np.zeros((pn*nm, r))
                idx = 0
                for i in range(pn):
                    for a in range(nm):
                        M_W[idx, :] = U[i, :] * V[a, :]
                        idx += 1
                T_W = T.reshape(pn*nm, mp).T
                W_new, _, _, _ = np.linalg.lstsq(M_W.T, T_W.T, rcond=None)
                W = W_new.T
                
                # Check error
                err = np.max(np.abs(np.einsum('ic,ac,bc->iab', U, V, W) - T))
                if err < 1e-10:
                    found = True
                    break
            
            if found:
                best_rank = r
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
        
        if found:
            break
    
    if best_U is None:
        # Fallback to schoolbook
        r = n*m*p
        U = np.zeros((pn, r))
        V = np.zeros((nm, r))
        W = np.zeros((mp, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[k*n+i, col] = 1
                    V[i*m+j, col] = 1
                    W[j*p+k, col] = 1
                    col += 1
        best_U, best_V, best_W = U, V, W
        best_rank = r
    
    # Round to half-integers
    U_list = [[int(round(x*2))/2 for x in best_U[i]] for i in range(pn)]
    V_list = [[int(round(x*2))/2 for x in best_V[i]] for i in range(nm)]
    W_list = [[int(round(x*2))/2 for x in best_W[i]] for i in range(mp)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
