import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 7
    nm, mp, pn = n*m, m*p, p*n  # 9, 21, 21
    
    # Build target tensor T[a,b,c]
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def verify(U, V, W):
        r = U.shape[1]
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    val = sum(U[a,c2]*V[b,c2]*W[c,c2] for c2 in range(r))
                    if abs(val - T[a,b,c]) > 1e-9:
                        return False
        return True
    
    # Schoolbook as fallback
    rank0 = n*m*p
    U0 = np.zeros((nm, rank0))
    V0 = np.zeros((mp, rank0))
    W0 = np.zeros((pn, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    
    def als_decompose(r, max_iters=300, tol=1e-10, seed=0):
        np.random.seed(seed)
        U = np.random.randn(nm, r)
        V = np.random.randn(mp, r)
        W = np.random.randn(pn, r)
        
        T_flat = T.reshape(nm, mp*pn)
        T_flat2 = T.reshape(mp, nm*pn)
        T_flat3 = T.reshape(pn, nm*mp)
        
        for it in range(max_iters):
            # Update U
            VW = np.zeros((r, mp*pn))
            for rr in range(r):
                VW[rr, :] = np.outer(V[:,rr], W[:,rr]).flatten()
            U_new = np.linalg.lstsq(VW.T, T_flat, rcond=None)[0]
            U = U_new
            
            # Update V
            UW = np.zeros((r, nm*pn))
            for rr in range(r):
                UW[rr, :] = np.outer(U[:,rr], W[:,rr]).flatten()
            V_new = np.linalg.lstsq(UW.T, T_flat2, rcond=None)[0]
            V = V_new
            
            # Update W
            UV = np.zeros((r, nm*mp))
            for rr in range(r):
                UV[rr, :] = np.outer(U[:,rr], V[:,rr]).flatten()
            W_new = np.linalg.lstsq(UV.T, T_flat3, rcond=None)[0]
            W = W_new
            
            # Error check
            T_pred = np.zeros((nm, mp, pn))
            for rr in range(r):
                T_pred += np.outer(U[:,rr], np.outer(V[:,rr], W[:,rr]))
            err = np.linalg.norm(T_pred - T)
            if err < tol:
                break
        
        T_pred = np.zeros((nm, mp, pn))
        for rr in range(r):
            T_pred += np.outer(U[:,rr], np.outer(V[:,rr], W[:,rr]))
        err = np.linalg.norm(T_pred - T)
        return U, V, W, err
    
    deadline = time.time() + 100
    for r in range(20, 45):
        if time.time() > deadline:
            break
        for seed in range(10):
            if time.time() > deadline:
                break
            U_r, V_r, W_r, err = als_decompose(r, max_iters=200, seed=seed)
            if err < 1e-8:
                if verify(U_r, V_r, W_r):
                    U_list = [[float(U_r[i,j]) for j in range(r)] for i in range(nm)]
                    V_list = [[float(V_r[i,j]) for j in range(r)] for i in range(mp)]
                    W_list = [[float(W_r[i,j]) for j in range(r)] for i in range(pn)]
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    U_list = [[float(U0[i,j]) for j in range(rank0)] for i in range(nm)]
    V_list = [[float(V0[i,j]) for j in range(rank0)] for i in range(mp)]
    W_list = [[float(W0[i,j]) for j in range(rank0)] for i in range(pn)]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
