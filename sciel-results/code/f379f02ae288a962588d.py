import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    
    T = np.zeros((U_dim, V_dim, W_dim), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try ranks from 49 upwards
    for rank in range(49, 126):
        if time.time() > deadline - 15.0:
            break
            
        result = _try_find_decomposition(T, U_dim, V_dim, W_dim, rank, deadline)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook
    return _schoolbook_decomposition(n, m, p)

def _schoolbook_decomposition(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_find_decomposition(T, U_dim, V_dim, W_dim, rank, deadline):
    # Returns (U, V, W) if found, else None
    num_restarts = 10
    lr = 0.05
    b1, b2, eps = 0.9, 0.999, 1e-8
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            return None
        
        U = np.random.randn(U_dim, rank) * 0.1
        V = np.random.randn(V_dim, rank) * 0.1
        W = np.random.randn(W_dim, rank) * 0.1
        
        m_U = np.zeros_like(U)
        v_U = np.zeros_like(U)
        m_V = np.zeros_like(V)
        v_V = np.zeros_like(V)
        m_W = np.zeros_like(W)
        v_W = np.zeros_like(W)
        
        t = 1
        prev_err = np.inf
        
        # Run until time limit or convergence
        while time.time() < deadline - 5.0:
            T_hat = np.einsum('ur,vr,wr->uvw', U, V, W)
            E = T_hat - T
            err_norm = np.linalg.norm(E)
            
            if err_norm < 1e-9:
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                T_hat_r = np.einsum('ur,vr,wr->uvw', U_r, V_r, W_r)
                if np.allclose(T_hat_r, T, atol=1e-9):
                    return (U_r, V_r, W_r)
            
            # Early break if no progress
            if err_norm > prev_err * 1.01 and t > 100:
                break
            prev_err = err_norm
            
            grad_U = np.einsum('uvw,vr,wr->ur', E, V, W)
            grad_V = np.einsum('uvw,ur,wr->vr', E, U, W)
            grad_W = np.einsum('uvw,ur,vr->wr', E, U, V)
            
            m_U = b1 * m_U + (1 - b1) * grad_U
            v_U = b2 * v_U + (1 - b2) * (grad_U ** 2)
            m_hat_U = m_U / (1 - b1 ** t)
            v_hat_U = v_U / (1 - b2 ** t)
            U -= lr * m_hat_U / (np.sqrt(v_hat_U) + eps)
            
            m_V = b1 * m_V + (1 - b1) * grad_V
            v_V = b2 * v_V + (1 - b2) * (grad_V ** 2)
            m_hat_V = m_V / (1 - b1 ** t)
            v_hat_V = v_V / (1 - b2 ** t)
            V -= lr * m_hat_V / (np.sqrt(v_hat_V) + eps)
            
            m_W = b1 * m_W + (1 - b1) * grad_W
            v_W = b2 * v_W + (1 - b2) * (grad_W ** 2)
            m_hat_W = m_W / (1 - b1 ** t)
            v_hat_W = v_W / (1 - b2 ** t)
            W -= lr * m_hat_W / (np.sqrt(v_hat_W) + eps)
            
            t += 1
            
    return None
