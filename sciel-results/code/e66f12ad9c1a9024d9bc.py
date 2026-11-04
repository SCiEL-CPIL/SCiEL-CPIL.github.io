import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145
    
    # Schoolbook
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_factors = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    try:
        for R in range(rank_sb - 1, 49, -1):
            if time.time() > deadline - 20:
                break
                
            result = try_find_decomposition(n, m, p, R, time_limit=15)
            if result is not None:
                best_factors = result
                best_rank = R
    except Exception:
        pass
        
    U, V, W = best_factors
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def try_find_decomposition(n, m, p, R, time_limit=15):
    start_time = time.time()
    
    for restart in range(5):
        if time.time() > start_time + time_limit:
            break
            
        U = np.random.randn(n, m, R) * 0.1
        V = np.random.randn(m, p, R) * 0.1
        W = np.random.randn(p, n, R) * 0.1
        
        lr = 0.01
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        
        m_U = np.zeros_like(U)
        v_U = np.zeros_like(U)
        m_V = np.zeros_like(V)
        v_V = np.zeros_like(V)
        m_W = np.zeros_like(W)
        v_W = np.zeros_like(W)
        
        t = 0
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        while time.time() < start_time + time_limit:
            t += 1
            
            T = np.einsum('ijr,jkr,kir->ijk', U, V, W)
            err = np.sum((T - 1)**2)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if err < 1e-8:
                break
                
            A = 2 * (T - 1)
            G_U = np.einsum('ijp,jpr,pnr->ijr', A, V, W)
            G_V = np.einsum('ijp,ijr,pnr->jpr', A, U, W)
            G_W = np.einsum('ijp,jpr,ijr->pnr', A, V, U)
            
            def adam_update(m, v, grad, t, lr, beta1, beta2, eps):
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad ** 2)
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                return m, v, lr * m_hat / (np.sqrt(v_hat) + eps)
                
            m_U, v_U, upd_U = adam_update(m_U, v_U, G_U, t, lr, beta1, beta2, eps)
            m_V, v_V, upd_V = adam_update(m_V, v_V, G_V, t, lr, beta1, beta2, eps)
            m_W, v_W, upd_W = adam_update(m_W, v_W, G_W, t, lr, beta1, beta2, eps)
            
            U -= upd_U
            V -= upd_V
            W -= upd_W
            
            if t % 100 == 0:
                lr *= 0.9
                
        if best_err < 1e-5:
            U_r = np.round(2 * best_U) / 2
            V_r = np.round(2 * best_V) / 2
            W_r = np.round(2 * best_W) / 2
            
            T_r = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            err_r = np.sum((T_r - 1)**2)
            
            if err_r < 1e-6:
                return (U_r.flatten().reshape(n*m, R), V_r.flatten().reshape(m*p, R), W_r.flatten().reshape(p*n, R))
                
            U_i = np.round(best_U)
            V_i = np.round(best_V)
            W_i = np.round(best_W)
            T_i = np.einsum('ijr,jkr,kir->ijk', U_i, V_i, W_i)
            err_i = np.sum((T_i - 1)**2)
            if err_i < 1e-6:
                return (U_i.flatten().reshape(n*m, R), V_i.flatten().reshape(m*p, R), W_i.flatten().reshape(p*n, R))
                
    return None
