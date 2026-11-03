import numpy as np

def construct():
    n, m, p = 2, 3, 4
    nm, mp, pn = n * m, m * p, p * n
    
    # Construct Target Tensor T (flattened or 3D)
    T = np.zeros((nm, mp, pn), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Target ranks to try
    target_ranks = [18, 19, 20, 21, 22, 23]
    
    best_rank = nm * mp * pn
    best_U, best_V, best_W = None, None, None

    # Fallback schoolbook
    U_fb = np.zeros((nm, best_rank), dtype=int)
    V_fb = np.zeros((mp, best_rank), dtype=int)
    W_fb = np.zeros((pn, best_rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    
    # ALS Search
    for R in target_ranks:
        for attempt in range(50): # 50 attempts per rank
            # Random init with small integers
            U = np.random.randint(-2, 3, size=(nm, R))
            V = np.random.randint(-2, 3, size=(mp, R))
            W = np.random.randint(-2, 3, size=(pn, R))
            
            # Ensure not all zero
            if np.all(U == 0): U[0,0] = 1
            if np.all(V == 0): V[0,0] = 1
            if np.all(W == 0): W[0,0] = 1

            success = False
            # Iterate ALS
            for iter in range(100):
                # Solve U
                # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                # For fixed V, W:
                # T[a, b, c] = sum_r U[a, r] (V[b, r] W[c, r])
                # Let M_VW[b, c, r] = V[b, r] W[c, r]
                # T[a, :, :] = sum_r U[a, r] M_VW[:, :, r]
                # This is a tensor equation.
                # We can flatten T and M_VW.
                # T_flat[a, b*c] = sum_r U[a, r] M_VW_flat[r, b*c]
                # So T_flat = U * M_VW_flat
                # U = T_flat * M_VW_flat^+ (Pseudoinverse)
                
                M_VW = np.zeros((R, mp * pn), dtype=float)
                for b in range(mp):
                    for c in range(pn):
                        for r in range(R):
                            M_VW[r, b * pn + c] = V[b, r] * W[c, r]
                
                T_flat = T.reshape(nm, mp * pn).astype(float)
                
                try:
                    U_new = np.linalg.lstsq(M_VW.T, T_flat.T, rcond=None)[0].T
                except:
                    break
                
                U_int = np.round(U_new).astype(int)
                
                # Solve V
                M_UW = np.zeros((R, nm * pn), dtype=float)
                for a in range(nm):
                    for c in range(pn):
                        for r in range(R):
                            M_UW[r, a * pn + c] = U_int[a, r] * W[c, r]
                
                T_flat_v = T.reshape(mp, nm * pn).T # Wait, T is (nm, mp, pn). 
                # T[b, a, c] = sum_r V[b, r] U[a, r] W[c, r]
                # T[b, a*pn+c] = sum_r V[b, r] M_UW[r, a*pn+c]
                # T_flat_v should be (mp, nm*pn)
                T_flat_v = np.zeros((mp, nm * pn), dtype=float)
                for a in range(nm):
                    for c in range(pn):
                        T_flat_v[:, a * pn + c] = T[a, :, c] # T[a, b, c] -> T_flat_v[b, a*pn+c]
                
                try:
                    V_new = np.linalg.lstsq(M_UW.T, T_flat_v.T, rcond=None)[0].T
                except:
                    break
                V_int = np.round(V_new).astype(int)
                
                # Solve W
                M_UV = np.zeros((R, nm * mp), dtype=float)
                for a in range(nm):
                    for b in range(mp):
                        for r in range(R):
                            M_UV[r, a * mp + b] = U_int[a, r] * V_int[b, r]
                
                T_flat_w = np.zeros((pn, nm * mp), dtype=float)
                for a in range(nm):
                    for b in range(mp):
                        T_flat_w[:, a * mp + b] = T[a, b, :] # T[a, b, c] -> T_flat_w[c, a*mp+b]
                
                try:
                    W_new = np.linalg.lstsq(M_UV.T, T_flat_w.T, rcond=None)[0].T
                except:
                    break
                W_int = np.round(W_new).astype(int)
                
                # Check convergence
                # Reconstruct
                T_recon = np.zeros((nm, mp, pn), dtype=int)
                for r in range(R):
                    T_recon += np.einsum('a,b,c->abc', U_int[:, r], V_int[:, r], W_int[:, r])
                
                if np.all(T_recon == T):
                    success = True
                    break
                
                U, V, W = U_int, V_int, W_int
            
            if success:
                best_rank = R
                best_U, best_V, best_W = U_int.tolist(), V_int.tolist(), W_int.tolist()
                break
        
        if success:
            break

    if best_U is None:
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    else:
        return {"U": best_U, "V": best_V, "W": best_W}
