import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130
    nm, mp, pn = n*m, m*p, p*n
    
    # Build tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook
    r0 = n*m*p
    U0 = np.zeros((nm, r0))
    V0 = np.zeros((mp, r0))
    W0 = np.zeros((pn, r0))
    for idx, (i,j,k) in enumerate([(i,j,k) for i in range(n) for j in range(m) for k in range(p)]):
        U0[i*m+j, idx] = 1
        V0[j*p+k, idx] = 1
        W0[k*n+i, idx] = 1
    
    best = (U0, V0, W0, r0)
    
    # Try ALS for various ranks
    for r in range(25, r0):
        if time.time() > deadline - 20:
            break
        
        for seed in range(30):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(seed)
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)
            
            for it in range(500):
                if time.time() > deadline - 10:
                    break
                
                # Update U
                MW = (V[:, :, None] * W[None, :, :]).reshape(mp*pn, r)
                TU = T.reshape(nm, -1)
                U = TU @ np.linalg.pinv(MW)
                
                # Update V
                MUW = (U[:, :, None] * W[None, :, :]).reshape(nm*pn, r)
                TV = T.transpose(1,0,2).reshape(mp, -1)
                V = TV @ np.linalg.pinv(MUW)
                
                # Update W
                MUV = (U[:, :, None] * V[None, :, :]).reshape(nm*mp, r)
                TW = T.transpose(2,0,1).reshape(pn, -1)
                W = TW @ np.linalg.pinv(MUV)
                
                if it % 100 == 0:
                    Trec = np.einsum('ad,bd,cd->abc', U, V, W)
                    if np.allclose(Trec, T, atol=1e-9):
                        best = (U, V, W, r)
                        break
            
            if best[3] == r:
                break
        
        if best[3] < r0:
            break
    
    U, V, W, r = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
