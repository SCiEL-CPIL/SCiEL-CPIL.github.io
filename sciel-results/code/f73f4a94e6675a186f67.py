import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    nm, mp, pn = n * m, m * p, p * n  # 6, 30, 20
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Build rank-55 Strassen solution
    def build_rank55():
        rank = 55
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        c = 0
        for i in range(n):
            for k in range(p):
                U[i*m+0, c] = 1
                V[0*p+k, c] = 1
                W[k*n+i, c] = 1
                c += 1
        
        for b in range(5):
            k0 = 2*b
            k1 = 2*b+1
            
            u11, u12 = 0*m+1, 0*m+2
            u21, u22 = 1*m+1, 1*m+2
            
            v11, v12 = 1*p+k0, 1*p+k1
            v21, v22 = 2*p+k0, 2*p+k1
            
            w00, w01 = k0*n+0, k1*n+0
            w10, w11 = k0*n+1, k1*n+1
            
            U[u11,c]+=1; U[u22,c]+=1; V[v11,c]+=1; V[v22,c]+=1; W[w00,c]+=1; W[w11,c]+=1; c+=1
            U[u21,c]+=1; U[u22,c]+=1; V[v11,c]+=1; W[w10,c]+=1; W[w11,c]+=-1; c+=1
            U[u11,c]+=1; V[v12,c]+=1; V[v22,c]+=-1; W[w01,c]+=1; W[w11,c]+=1; c+=1
            U[u22,c]+=1; V[v21,c]+=1; V[v11,c]+=-1; W[w00,c]+=1; W[w10,c]+=1; c+=1
            U[u11,c]+=1; U[u12,c]+=1; V[v22,c]+=1; W[w00,c]+=-1; W[w01,c]+=1; c+=1
            U[u21,c]+=1; U[u11,c]+=-1; V[v11,c]+=1; V[v12,c]+=1; W[w11,c]+=1; c+=1
            U[u12,c]+=1; U[u22,c]+=-1; V[v21,c]+=1; V[v22,c]+=1; W[w00,c]+=1; c+=1
        
        return U, V, W, rank
    
    def verify(U, V, W, tol=1e-6):
        r = U.shape[1]
        T_recon = np.zeros((nm, mp, pn))
        for r_idx in range(r):
            T_recon += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(T_recon[i*m+j, j*p+k, k*n+i] - 1.0) > tol:
                        return False
        zero_mask = (T == 0)
        max_err = np.max(np.abs(T_recon[zero_mask]))
        return max_err < tol
    
    # Fast vectorized ALS
    def fast_als(U_init, V_init, W_init, target_rank, max_iters=200, tol=1e-10):
        U = U_init[:, :target_rank].copy()
        V = V_init[:, :target_rank].copy()
        W = W_init[:, :target_rank].copy()
        r = target_rank
        
        # Precompute T slices
        T_a = T.reshape(nm, mp * pn)
        T_b = T.transpose(1, 0, 2).reshape(mp, nm * pn)
        T_c = T.transpose(2, 0, 1).reshape(pn, nm * mp)
        
        prev_err = float('inf')
        stall = 0
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Update U
            V_r = V.T.reshape(r, 1, mp)
            W_r = W.T.reshape(r, pn, 1)
            B_U_T = (V_r * W_r).reshape(r, mp * pn)
            B_U = B_U_T.T
            try:
                U = np.linalg.lstsq(B_U.T, T_a.T, rcond=None)[0].T
            except:
                pass
            
            # Update V
            U_r = U.T.reshape(r, 1, nm)
            W_r = W.T.reshape(r, pn, 1)
            B_V_T = (U_r * W_r).reshape(r, nm * pn)
            B_V = B_V_T.T
            try:
                V = np.linalg.lstsq(B_V.T, T_b.T, rcond=None)[0].T
            except:
                pass
            
            # Update W
            U_r = U.T.reshape(r, 1, nm)
            V_r = V.T.reshape(r, mp, 1)
            B_W_T = (U_r * V_r).reshape(r, nm * mp)
            B_W = B_W_T.T
            try:
                W = np.linalg.lstsq(B_W.T, T_c.T, rcond=None)[0].T
            except:
                pass
            
            # Check error every 5 iterations
            if it % 5 == 0:
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            err += (val - 1.0)**2
                if err < tol:
                    break
                if abs(prev_err - err) < 1e-12:
                    stall += 1
                    if stall > 5:
                        break
                else:
                    stall = 0
                prev_err = err
        
        return U, V, W
    
    def polish(U, V, W):
        for scale in [2, 1]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            if verify(U_r, V_r, W_r, tol=1e-9):
                return U_r, V_r, W_r
        return None
    
    # Get rank-55 baseline
    U55, V55, W55, rank55 = build_rank55()
    if not verify(U55, V55, W55, tol=1e-9):
        # Fallback: schoolbook
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W, best_rank = U55, V55, W55, rank55
    
    # Try lower ranks
    for target_rank in range(rank55 - 1, 45, -1):
        if time.time() > deadline - 20:
            break
        
        for restart in range(12):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(42 + restart * 7 + target_rank * 13)
            
            if restart == 0:
                U_init = U55[:, :target_rank] + 0.05 * np.random.randn(nm, target_rank)
                V_init = V55[:, :target_rank] + 0.05 * np.random.randn(mp, target_rank)
                W_init = W55[:, :target_rank] + 0.05 * np.random.randn(pn, target_rank)
            elif restart == 1:
                U_init = np.random.randn(nm, target_rank) * 0.5
                V_init = np.random.randn(mp, target_rank) * 0.5
                W_init = np.random.randn(pn, target_rank) * 0.5
            elif restart == 2:
                U_init = best_U[:, :target_rank] + 0.02 * np.random.randn(nm, target_rank)
                V_init = best_V[:, :target_rank] + 0.02 * np.random.randn(mp, target_rank)
                W_init = best_W[:, :target_rank] + 0.02 * np.random.randn(pn, target_rank)
            elif restart == 3:
                idx = np.random.choice(rank55, target_rank, replace=False)
                U_init = U55[:, idx] + 0.1 * np.random.randn(nm, target_rank)
                V_init = V55[:, idx] + 0.1 * np.random.randn(mp, target_rank)
                W_init = W55[:, idx] + 0.1 * np.random.randn(pn, target_rank)
            elif restart == 4:
                perm = np.random.permutation(rank55)
                idx = perm[:target_rank]
                U_init = U55[:, idx] + 0.08 * np.random.randn(nm, target_rank)
                V_init = V55[:, idx] + 0.08 * np.random.randn(mp, target_rank)
                W_init = W55[:, idx] + 0.08 * np.random.randn(pn, target_rank)
            elif restart == 5:
                U_init = np.random.randn(nm, target_rank)
                V_init = np.random.randn(mp, target_rank)
                W_init = np.random.randn(pn, target_rank)
            elif restart == 6:
                U_init = U55[:, :target_rank] + 0.01 * np.random.randn(nm, target_rank)
                V_init = V55[:, :target_rank] + 0.01 * np.random.randn(mp, target_rank)
                W_init = W55[:, :target_rank] + 0.01 * np.random.randn(pn, target_rank)
            elif restart == 7:
                U_init = np.random.randn(nm, target_rank) * 0.3
                V_init = np.random.randn(mp, target_rank) * 0.3
                W_init = np.random.randn(pn, target_rank) * 0.3
            elif restart == 8:
                idx = np.random.choice(rank55, target_rank, replace=False)
                U_init = U55[:, idx].copy()
                V_init = V55[:, idx].copy()
                W_init = W55[:, idx].copy()
            elif restart == 9:
                U_init = np.random.randn(nm, target_rank) * 0.7
                V_init = np.random.randn(mp, target_rank) * 0.7
                W_init = np.random.randn(pn, target_rank) * 0.7
            elif restart == 10:
                U_init = U55[:, :target_rank] + 0.15 * np.random.randn(nm, target_rank)
                V_init = V55[:, :target_rank] + 0.15 * np.random.randn(mp, target_rank)
                W_init = W55[:, :target_rank] + 0.15 * np.random.randn(pn, target_rank)
            else:
                U_init = np.random.randn(nm, target_rank) * 0.4
                V_init = np.random.randn(mp, target_rank) * 0.4
                W_init = np.random.randn(pn, target_rank) * 0.4
            
            U_opt, V_opt, W_opt = fast_als(U_init, V_init, W_init, target_rank, max_iters=200)
            
            polished = polish(U_opt, V_opt, W_opt)
            if polished is not None:
                best_U, best_V, best_W, best_rank = polished
                break
            
            if verify(U_opt, V_opt, W_opt, tol=1e-6):
                polished = polish(U_opt, V_opt, W_opt)
                if polished is not None:
                    best_U, best_V, best_W, best_rank = polished
                    break
                else:
                    best_U, best_V, best_W, best_rank = U_opt, V_opt, W_opt, target_rank
                    break
    
    if not verify(best_U, best_V, best_W, tol=1e-6):
        best_U, best_V, best_W, best_rank = U55, V55, W55, rank55
    
    return {
        "U": [[float(x) for x in row] for row in best_U],
        "V": [[float(x) for x in row] for row in best_V],
        "W": [[float(x) for x in row] for row in best_W]
    }
