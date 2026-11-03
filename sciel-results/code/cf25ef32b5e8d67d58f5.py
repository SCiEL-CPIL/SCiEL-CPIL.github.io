import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook rank
    R_schoolbook = n * m * p
    
    # Build target tensor T_3d of shape (nm, mp, pn)
    T_3d = np.zeros((nm, mp, pn))
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_3d[u, v, w] = 1.0
                indices.append((u, v, w))

    # Schoolbook solution
    U_sb = np.zeros((nm, R_schoolbook))
    V_sb = np.zeros((mp, R_schoolbook))
    W_sb = np.zeros((pn, R_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U_sb[u, col] = 1
                V_sb[v, col] = 1
                W_sb[w, col] = 1
                col += 1
                
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()
    best_rank = R_schoolbook
    
    start_time = time.time()
    time_limit = 115
    
    # Try to find lower rank
    for R in range(R_schoolbook - 1, max(50, R_schoolbook - 20), -1):
        if time.time() - start_time > time_limit:
            break
            
        found = False
        for attempt in range(20):
            if time.time() - start_time > time_limit:
                break
                
            np.random.seed(42 + attempt)
            U = np.random.randn(nm, R)
            V = np.random.randn(mp, R)
            W = np.random.randn(pn, R)
            
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            for it in range(50):
                if time.time() - start_time > time_limit:
                    break
                    
                # Update W
                A_all = np.zeros((nm * mp, R))
                for r in range(R):
                    A_all[:, r] = np.outer(U[:, r], V[:, r]).flatten()
                
                T_flat_T = T_3d.transpose(2, 0, 1).reshape(pn, nm * mp).T
                try:
                    W_new, _, _, _ = np.linalg.lstsq(A_all, T_flat_T, rcond=None)
                    W = W_new.T
                except:
                    pass
                    
                # Update U
                B_all = np.zeros((pn * mp, R))
                for r in range(R):
                    B_all[:, r] = np.outer(V[:, r], W[:, r]).flatten()
                    
                U_new = np.zeros((nm, R))
                for u in range(nm):
                    T_u_flat = T_3d[u].flatten()
                    U_u, _, _, _ = np.linalg.lstsq(B_all, T_u_flat, rcond=None)
                    U_new[u] = U_u
                U = U_new
                
                # Update V
                C_all = np.zeros((nm * pn, R))
                for r in range(R):
                    C_all[:, r] = np.outer(U[:, r], W[:, r]).flatten()
                    
                V_new = np.zeros((mp, R))
                for v in range(mp):
                    T_v_flat = T_3d[:, v, :].flatten()
                    V_v, _, _, _ = np.linalg.lstsq(C_all, T_v_flat, rcond=None)
                    V_new[v] = V_v
                V = V_new
                
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            T_recon = np.zeros((nm, mp, pn))
            for r in range(R):
                if np.all(U_r[:, r] == 0) or np.all(V_r[:, r] == 0) or np.all(W_r[:, r] == 0):
                    continue
                T_recon += np.einsum('u,v,w->uvw', U_r[:, r], V_r[:, r], W_r[:, r])
                
            diff = np.abs(T_recon - T_3d)
            if np.max(diff) < 1e-9:
                best_rank = R
                best_U = U_r.tolist()
                best_V = V_r.tolist()
                best_W = W_r.tolist()
                found = True
                break
                
        if found:
            break
            
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
