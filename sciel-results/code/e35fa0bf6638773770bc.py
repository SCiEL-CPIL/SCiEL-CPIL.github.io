import numpy as np
import time

def construct():
    n, m, p = 2, 4, 7
    nm, mp, pn = n*m, m*p, p*n
    
    # Non-zero entries of the tensor
    nnz = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nnz.append((i*m+j, j*p+k, k*n+i))
    
    start_time = time.time()
    deadline = start_time + 100
    
    def schoolbook():
        rank = n*m*p
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
        return U.tolist(), V.tolist(), W.tolist()
    
    # Try ranks from low to high
    for rank in range(30, n*m*p+1):
        if time.time() > deadline - 5:
            break
        
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(rank * 100 + trial)
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Alternating least squares
            for it in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Update U
                for a in range(nm):
                    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                    # For fixed a: T[a,:,:] = sum_r U[a,r] (V[:,r] * W[:,r])
                    # Let P[b,c,r] = V[b,r] * W[c,r]
                    # T[a,:,:] = U[a,:] @ P (where P is reshaped)
                    P = (V[:, :, np.newaxis] * W[np.newaxis, :, :]).transpose(0, 2, 1).reshape(mp*pn, rank)
                    # Actually, P should be: P[b*pn+c, r] = V[b,r] * W[c,r]
                    P = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(mp*pn, rank)
                    # T[a, :, :] flattened is target
                    # We need U[a,:] such that U[a,:] @ P^T = T[a,:,:].flatten()
                    # i.e., P @ U[a,:]^T = T[a,:,:].flatten()
                    target = np.zeros(mp*pn)
                    for (a2, b, c) in nnz:
                        if a2 == a:
                            target[b*pn+c] = 1.0
                    U[a, :] = np.linalg.lstsq(P, target, rcond=None)[0]
                
                # Update V
                for b in range(mp):
                    P = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(nm*pn, rank)
                    target = np.zeros(nm*pn)
                    for (a, b2, c) in nnz:
                        if b2 == b:
                            target[a*pn+c] = 1.0
                    V[b, :] = np.linalg.lstsq(P, target, rcond=None)[0]
                
                # Update W
                for c in range(pn):
                    P = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(nm*mp, rank)
                    target = np.zeros(nm*mp)
                    for (a, b, c2) in nnz:
                        if c2 == c:
                            target[a*mp+b] = 1.0
                    W[c, :] = np.linalg.lstsq(P, target, rcond=None)[0]
                
                # Check error
                err = 0
                for (a, b, c) in nnz:
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    err += (val - 1)**2
                # Also check zero entries (sample)
                if err < 1e-15:
                    # Check some zero entries
                    zero_err = 0
                    for _ in range(100):
                        a = np.random.randint(nm)
                        b = np.random.randint(mp)
                        c = np.random.randint(pn)
                        if (a, b, c) not in nnz:
                            val = np.dot(U[a, :], V[b, :] * W[c, :])
                            zero_err += val**2
                    if zero_err < 1e-15:
                        # Try rounding
                        for mult in [1, 2]:
                            U_r = np.round(U * mult) / mult
                            V_r = np.round(V * mult) / mult
                            W_r = np.round(W * mult) / mult
                            err = 0
                            for (a, b, c) in nnz:
                                val = np.dot(U_r[a, :], V_r[b, :] * W_r[c, :])
                                err += (val - 1)**2
                            zero_err = 0
                            for _ in range(100):
                                a = np.random.randint(nm)
                                b = np.random.randint(mp)
                                c = np.random.randint(pn)
                                if (a, b, c) not in nnz:
                                    val = np.dot(U_r[a, :], V_r[b, :] * W_r[c, :])
                                    zero_err += val**2
                            if err < 1e-10 and zero_err < 1e-10:
                                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                        break
            
            if time.time() > deadline - 5:
                break
    
    # Fallback
    U, V, W = schoolbook()
    return {"U": U, "V": V, "W": W}
