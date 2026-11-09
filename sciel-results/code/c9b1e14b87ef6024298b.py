import numpy as np
import time
import random

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    deadline = time.time() + 140
    
    # Schoolbook baseline
    rank_sb = n * m * p  # 60
    U = np.zeros((nm, rank_sb))
    V = np.zeros((mp, rank_sb))
    W = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Precompute target entries
    targets = [(i*m+j, j*p+k, k*n+i) for i in range(n) for j in range(m) for k in range(p)]
    
    # Try to find lower rank with limited attempts
    for target_rank in range(rank_sb - 1, 19, -1):
        if time.time() > deadline - 20:
            break
        
        success = False
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            
            U_r = np.random.randn(nm, target_rank) * 0.5
            V_r = np.random.randn(mp, target_rank) * 0.5
            W_r = np.random.randn(pn, target_rank) * 0.5
            
            lr = 0.1
            for it in range(300):
                if time.time() > deadline - 2:
                    break
                
                dU = np.zeros_like(U_r)
                dV = np.zeros_like(V_r)
                dW = np.zeros_like(W_r)
                loss = 0.0
                
                for (a, b, c) in targets:
                    pred = np.dot(U_r[a,:] * V_r[b,:], W_r[c,:])
                    err = pred - 1.0
                    loss += err * err
                    dU[a,:] += V_r[b,:] * W_r[c,:] * err
                    dV[b,:] += U_r[a,:] * W_r[c,:] * err
                    dW[c,:] += U_r[a,:] * V_r[b,:] * err
                
                for _ in range(100):
                    a = random.randint(0, nm-1)
                    b = random.randint(0, mp-1)
                    c = random.randint(0, pn-1)
                    ia, ja = divmod(a, m)
                    jb, kb = divmod(b, p)
                    kc, ic = divmod(c, n)
                    if ia != ic or ja != jb:
                        pred = np.dot(U_r[a,:] * V_r[b,:], W_r[c,:])
                        loss += pred * pred
                        dU[a,:] += V_r[b,:] * W_r[c,:] * pred
                        dV[b,:] += U_r[a,:] * W_r[c,:] * pred
                        dW[c,:] += U_r[a,:] * V_r[b,:] * pred
                
                if loss < 1e-8:
                    success = True
                    break
                
                U_r -= lr * dU
                V_r -= lr * dV
                W_r -= lr * dW
            
            if success:
                best_rank = target_rank
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                break
        
        if success:
            break
    
    # Verify the best result
    def verify(U_mat, V_mat, W_mat):
        for (a, b, c) in targets:
            val = np.dot(U_mat[a,:] * V_mat[b,:], W_mat[c,:])
            if abs(val - 1.0) > 1e-6:
                return False
        for _ in range(500):
            a = random.randint(0, nm-1)
            b = random.randint(0, mp-1)
            c = random.randint(0, pn-1)
            ia, ja = divmod(a, m)
            jb, kb = divmod(b, p)
            kc, ic = divmod(c, n)
            if ia != ic or ja != jb:
                val = np.dot(U_mat[a,:] * V_mat[b,:], W_mat[c,:])
                if abs(val) > 1e-6:
                    return False
        return True
    
    if not verify(best_U, best_V, best_W):
        # Fall back to schoolbook
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = rank_sb
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
