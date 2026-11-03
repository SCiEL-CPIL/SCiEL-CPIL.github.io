import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 105
    
    nm, mp, pn = n*m, m*p, p*n  # 6, 21, 14
    
    # Target tensor
    target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try ranks from lower bound
    lower_bound = max(nm, mp, pn)  # 21
    
    for R in range(lower_bound, 43):
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            np.random.seed(1000 + R * 10 + attempt)
            U = np.random.randn(nm, R) * 0.5
            V = np.random.randn(mp, R) * 0.5
            W = np.random.randn(pn, R) * 0.5
            
            best_err = np.inf
            for it in range(2000):
                if time.time() > deadline:
                    break
                
                # Solve for U
                VW = np.einsum('br,cr->bcr', V, W).reshape(mp*pn, R)
                tf = target.reshape(nm, mp*pn)
                U = tf @ np.linalg.pinv(VW)
                
                # Solve for V
                UW = np.einsum('ar,cr->acr', U, W).reshape(nm*pn, R)
                tfv = target.transpose(1,0,2).reshape(mp, nm*pn)
                V = tfv @ np.linalg.pinv(UW)
                
                # Solve for W
                UV = np.einsum('ar,br->abr', U, V).reshape(nm*mp, R)
                tfw = target.transpose(2,0,1).reshape(pn, nm*mp)
                W = tfw @ np.linalg.pinv(UV)
                
                if it % 100 == 99:
                    recon = np.einsum('ar,br,cr->abc', U, V, W)
                    err = np.linalg.norm(recon - target)
                    if err < best_err:
                        best_err = err
                        if err < 1e-8:
                            found = True
                            break
            
            if found:
                recon = np.einsum('ar,br,cr->abc', U, V, W)
                if np.allclose(recon, target, atol=1e-7):
                    # Try to round to half-integers
                    U2 = np.round(U*2)/2
                    V2 = np.round(V*2)/2
                    W2 = np.round(W*2)/2
                    recon2 = np.einsum('ar,br,cr->abc', U2, V2, W2)
                    if np.allclose(recon2, target, atol=1e-6):
                        return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback
    R = n*m*p
    U = np.zeros((nm, R))
    V = np.zeros((mp, R))
    W = np.zeros((pn, R))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i*m*p + j*p + k
                U[i*m+j, r] = 1
                V[j*p+k, r] = 1
                W[k*n+i, r] = 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
