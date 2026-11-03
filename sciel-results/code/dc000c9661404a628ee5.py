import numpy as np
import time
import random

def construct():
    n, m, p = 3, 5, 6
    deadline = time.time() + 140
    nm, mp, pn = n * m, m * p, p * n  # 15, 30, 18
    
    # Build target tensor T[a, b, c] = 1 if a = i*m + j, b = j*p + k, c = k*n + i
    # i in [0,n), j in [0,m), k in [0,p)
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try ranks from lower bound up
    # Lower bound: max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # But matrix multiplication rank can be lower. For 3x5*5x6, 
    # the trivial rank is 3*5*6=90. Known bounds suggest much lower.
    # Let's try from 30 up.
    
    best_result = None
    
    for rank in range(30, 91):
        if time.time() > deadline - 20:
            break
        
        # Try several random initializations
        for attempt in range(20):
            if time.time() > deadline - 15:
                break
            
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # ALS iterations
            converged = False
            for it in range(200):
                if time.time() > deadline - 10:
                    break
                
                # Solve for U: T[a,b,c] = sum_k U[a,k] V[b,k] W[c,k]
                # Z_U[k, (b,c)] = V[b,k] * W[c,k]
                Z_U = np.einsum('bk,ck->kbc', V, W).reshape(rank, mp * pn)
                T_2d = T.reshape(nm, mp * pn)
                try:
                    U_new = np.linalg.lstsq(Z_U.T, T_2d.T, rcond=None)[0].T
                except:
                    break
                U = U_new
                
                # Solve for V
                Z_V = np.einsum('ak,ck->kac', U, W).reshape(rank, nm * pn)
                T_2d_v = T.transpose(1, 0, 2).reshape(mp, nm * pn)
                try:
                    V_new = np.linalg.lstsq(Z_V.T, T_2d_v.T, rcond=None)[0].T
                except:
                    break
                V = V_new
                
                # Solve for W
                Z_W = np.einsum('ak,bk->kab', U, V).reshape(rank, nm * mp)
                T_2d_w = T.transpose(2, 0, 1).reshape(pn, nm * mp)
                try:
                    W_new = np.linalg.lstsq(Z_W.T, T_2d_w.T, rcond=None)[0].T
                except:
                    break
                W = W_new
                
                # Check error
                T_comp = np.einsum('ak,bk,ck->abc', U, V, W, optimize=True)
                err = np.linalg.norm(T_comp - T)
                if err < 1e-10:
                    # Verify exactness by checking integer/half-integer rounding
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    T_check = np.einsum('ak,bk,ck->abc', U_rounded, V_rounded, W_rounded, optimize=True)
                    if np.allclose(T_check, T, atol=1e-12):
                        return {'U': U_rounded, 'V': V_rounded, 'W': W_rounded}
                    elif best_result is None or rank < best_result[0]:
                        best_result = (rank, U_rounded, V_rounded, W_rounded)
                    converged = True
                    break
            
            if converged:
                break
    
    if best_result is not None:
        return {'U': best_result[1], 'V': best_result[2], 'W': best_result[3]}
    
    # Fallback: schoolbook decomposition (rank = n*m*p = 90)
    U_fb = np.zeros((nm, n*m*p))
    V_fb = np.zeros((mp, n*m*p))
    W_fb = np.zeros((pn, n*m*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_fb[a, idx] = 1.0
                V_fb[b, idx] = 1.0
                W_fb[c, idx] = 1.0
    return {'U': U_fb, 'V': V_fb, 'W': W_fb}
