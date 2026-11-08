import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 24
    pn = p * n  # 24
    
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    a_idx = np.array([i * m + j for (i, j, k) in triples])
    b_idx = np.array([j * p + k for (i, j, k) in triples])
    c_idx = np.array([k * n + i for (i, j, k) in triples])
    
    def eval_fast(U, V, W):
        UV = U[a_idx, :] * V[b_idx, :]
        vals = np.sum(UV * W[c_idx, :], axis=1)
        return np.max(np.abs(vals - 1.0))
    
    # Try ranks from low to high
    for rank in range(2, 73):
        if time.time() > deadline - 15:
            break
        
        best_err = np.inf
        best_U = best_V = best_W = None
        
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            for it in range(300):
                if time.time() > deadline - 10:
                    break
                
                err = eval_fast(U, V, W)
                if err < 1e-10:
                    break
                
                # Update U via least squares
                for a in range(nm):
                    mask = (a_idx == a)
                    r_indices = np.where(mask)[0]
                    if len(r_indices) == 0:
                        continue
                    A = V[b_idx[r_indices], :] * W[c_idx[r_indices], :]
                    try:
                        U[a, :] = np.linalg.lstsq(A, np.ones(len(r_indices)), rcond=None)[0]
                    except:
                        pass
                
                # Update V via least squares
                for b in range(mp):
                    mask = (b_idx == b)
                    r_indices = np.where(mask)[0]
                    if len(r_indices) == 0:
                        continue
                    A = U[a_idx[r_indices], :] * W[c_idx[r_indices], :]
                    try:
                        V[b, :] = np.linalg.lstsq(A, np.ones(len(r_indices)), rcond=None)[0]
                    except:
                        pass
                
                # Update W via least squares
                for c in range(pn):
                    mask = (c_idx == c)
                    r_indices = np.where(mask)[0]
                    if len(r_indices) == 0:
                        continue
                    A = U[a_idx[r_indices], :] * V[b_idx[r_indices], :]
                    try:
                        W[c, :] = np.linalg.lstsq(A, np.ones(len(r_indices)), rcond=None)[0]
                    except:
                        pass
                
                err = eval_fast(U, V, W)
                if err < 1e-10:
                    break
            
            err = eval_fast(U, V, W)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < 1e-8:
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
