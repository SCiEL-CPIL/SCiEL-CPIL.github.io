"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    def compute_tensor(U, V, W):
        """Compute T[i,j,k,l] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+l,r]"""
        T = np.zeros((n, m, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    for l in range(p):
                        T[i,j,k,l] = np.dot(U[i*m+j], V[j*p+k] * W[k*n+l])
        return T
    
    def target_tensor():
        T_target = np.zeros((n, m, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    for l in range(p):
                        T_target[i,j,k,l] = 1.0 if j == k else 0.0
        return T_target
    
    def try_rank(R, seed=42, max_iter=3000):
        """Try to find a decomposition of rank R using Adam gradient descent."""
        rng = np.random.RandomState(seed)
        
        U = rng.randn(n * m, R) * 0.1
        V = rng.randn(m * p, R) * 0.1
        W = rng.randn(p * n, R) * 0.1
        
        T_target = target_tensor()
        
        m_U = np.zeros_like(U)
        v_U = np.zeros_like(U)
        m_V = np.zeros_like(V)
        v_V = np.zeros_like(V)
        m_W = np.zeros_like(W)
        v_W = np.zeros_like(W)
        
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        lr = 0.05
        
        for t in range(1, max_iter + 1):
            T_current = compute_tensor(U, V, W)
            error = T_current - T_target
            loss = np.sum(error**2)
            
            if loss < 1e-12:
                break
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(m):
                        for l in range(p):
                            e = error[i,j,k,l]
                            for r in range(R):
                                dU[i*m+j, r] += 2 * e * V[j*p+k, r] * W[k*n+l, r]
                                dV[j*p+k, r] += 2 * e * U[i*m+j, r] * W[k*n+l, r]
                                dW[k*n+l, r] += 2 * e * U[i*m+j, r] * V[j*p+k, r]
            
            m_U = beta1 * m_U + (1 - beta1) * dU
            v_U = beta2 * v_U + (1 - beta2) * dU**2
            m_U_hat = m_U / (1 - beta1**t)
            v_U_hat = v_U / (1 - beta2**t)
            U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
            
            m_V = beta1 * m_V + (1 - beta1) * dV
            v_V = beta2 * v_V + (1 - beta2) * dV**2
            m_V_hat = m_V / (1 - beta1**t)
            v_V_hat = v_V / (1 - beta2**t)
            V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
            
            m_W = beta1 * m_W + (1 - beta1) * dW
            v_W = beta2 * v_W + (1 - beta2) * dW**2
            m_W_hat = m_W / (1 - beta1**t)
            v_W_hat = v_W / (1 - beta2**t)
            W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
        
        T_final = compute_tensor(U, V, W)
        max_err = np.max(np.abs(T_final - T_target))
        
        if max_err < 1e-6:
            return U, V, W, max_err
        return None, None, None, max_err
    
    def to_exact(U, V, W):
        """Convert to exact integer or half-integer values."""
        best_result = None
        best_max_err = float('inf')
        
        for scale in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            sU = U * scale
            sV = V * scale
            sW = W * scale
            
            # Try integers
            Ui = np.round(sU)
            Vi = np.round(sV)
            Wi = np.round(sW)
            err = max(np.max(np.abs(Ui - sU)), np.max(np.abs(Vi - sV)), np.max(np.abs(Wi - sW)))
            if err < 1e-6 and err < best_max_err:
                best_max_err = err
                Uf = Ui.astype(float)
                Vf = Vi.astype(float)
                Wf = Wi.astype(float)
                T_check = compute_tensor(Uf, Vf, Wf)
                T_target = target_tensor()
                if np.max(np.abs(T_check - T_target)) < 1e-8:
                    best_result = {"U": [[int(x) for x in row] for row in Uf],
                                   "V": [[int(x) for x in row] for row in Vf],
                                   "W": [[int(x) for x in row] for row in Wf]}
            
            # Try half-integers
            Uh = np.round(sU * 2) / 2
            Vh = np.round(sV * 2) / 2
            Wh = np.round(sW * 2) / 2
            err = max(np.max(np.abs(Uh - sU)), np.max(np.abs(Vh - sV)), np.max(np.abs(Wh - sW)))
            if err < 1e-6 and err < best_max_err:
                best_max_err = err
                Uf = Uh
                Vf = Vh
                Wf = Wh
                T_check = compute_tensor(Uf, Vf, Wf)
                T_target = target_tensor()
                if np.max(np.abs(T_check - T_target)) < 1e-8:
                    best_result = {"U": [[float(x) for x in row] for row in Uf],
                                   "V": [[float(x) for x in row] for row in Vf],
                                   "W": [[float(x) for x in row] for row in Wf]}
        
        return best_result
    
    # Try ranks from 8 upward with multiple seeds
    for R in range(8, 13):
        for seed in [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]:
            U, V, W, err = try_rank(R, seed, max_iter=3000)
            if U is not None:
                result = to_exact(U, V, W)
                if result is not None:
                    return result
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
