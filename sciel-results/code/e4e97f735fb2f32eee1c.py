import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 3]
DEADLINE = 118.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    def iu(i, j): return i * m + j
    def iv(j, k): return j * p + k
    def iw(k, i): return k * n + i
    
    def verify(U, V, W, atol=1e-10):
        R = U.shape[1]
        T_recon = np.zeros((n, m, p))
        for r in range(R):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        T_target = np.ones((n, m, p))
        return np.allclose(T_recon, T_target, atol=atol)
    
    # Rank 26 decomposition: Strassen (7) + 19 border terms
    def rank26():
        R = 26
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        # Strassen m1 = (a00+a11)(b00+b11) -> contributes to c00, c11
        r = 0
        U[iu(0,0), r] = 1; U[iu(1,1), r] = 1
        V[iv(0,0), r] = 1; V[iv(1,1), r] = 1
        W[iw(0,0), r] = 1; W[iw(1,1), r] = 1
        
        # m2 = (a10+a11)b00 -> contributes to c10, subtracts from c11
        r = 1
        U[iu(1,0), r] = 1; U[iu(1,1), r] = 1
        V[iv(0,0), r] = 1
        W[iw(0,1), r] = 1; W[iw(1,1), r] = -1
        
        # m3 = a00(b01-b11) -> contributes to c01, c11
        r = 2
        U[iu(0,0), r] = 1
        V[iv(0,1), r] = 1; V[iv(1,1), r] = -1
        W[iw(1,0), r] = 1; W[iw(1,1), r] = 1
        
        # m4 = a11(b10-b00) -> contributes to c00, c10
        r = 3
        U[iu(1,1), r] = 1
        V[iv(1,0), r] = 1; V[iv(0,0), r] = -1
        W[iw(0,0), r] = 1; W[iw(0,1), r] = 1
        
        # m5 = (a00+a01)b11 -> subtracts from c00, contributes to c01
        r = 4
        U[iu(0,0), r] = 1; U[iu(0,1), r] = 1
        V[iv(1,1), r] = 1
        W[iw(0,0), r] = -1; W[iw(1,0), r] = 1
        
        # m6 = (a10-a00)(b00+b01) -> contributes to c11
        r = 5
        U[iu(1,0), r] = 1; U[iu(0,0), r] = -1
        V[iv(0,0), r] = 1; V[iv(0,1), r] = 1
        W[iw(1,1), r] = 1
        
        # m7 = (a01-a11)(b10+b11) -> contributes to c00
        r = 6
        U[iu(0,1), r] = 1; U[iu(1,1), r] = -1
        V[iv(1,0), r] = 1; V[iv(1,1), r] = 1
        W[iw(0,0), r] = 1
        
        # Border terms
        col = 7
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i < 2 and j < 2 and k < 2:
                        continue
                    U[iu(i,j), col] = 1
                    V[iv(j,k), col] = 1
                    W[iw(k,i), col] = 1
                    col += 1
        
        return U, V, W
    
    # Rank 27 schoolbook
    def rank27():
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[iu(i,j), col] = 1
                    V[iv(j,k), col] = 1
                    W[iw(k,i), col] = 1
                    col += 1
        return U, V, W
    
    # Try rank 26 first
    U26, V26, W26 = rank26()
    if verify(U26, V26, W26):
        return {"U": U26.tolist(), "V": V26.tolist(), "W": W26.tolist()}
    
    def try_find_rank(R, max_restarts=10):
        """
        Attempts to find a rank-R decomposition using random restarts and optimization.
        Returns (U, V, W) if successful, else None.
        """
        if time.time() - start_time > DEADLINE - 10:
            return None
            
        def objective(x):
            U = x[:n*m*R].reshape(n*m, R)
            V = x[n*m*R:2*n*m*R].reshape(m*p, R)
            W = x[2*n*m*R:].reshape(p*n, R)
            
            T_recon = np.zeros((n, m, p))
            for r in range(R):
                u_r = U[:, r].reshape(n, m)
                v_r = V[:, r].reshape(m, p)
                w_r = W[:, r].reshape(p, n)
                # T_r[i,j,k] = u_r[i,j] * v_r[j,k] * w_r[k,i]
                # w_r.T[i,k] = w_r[k,i]
                term = u_r[:, :, np.newaxis] * v_r[np.newaxis, :, :] * w_r.T[:, np.newaxis, :]
                T_recon += term
                
            T_target = np.ones((n, m, p))
            diff = T_recon - T_target
            return np.sum(diff**2)

        best_loss = float('inf')
        best_x = None
        
        for restart in range(max_restarts):
            if time.time() - start_time > DEADLINE - 5:
                break
                
            U_r = np.random.randn(n*m, R) * 0.1
            V_r = np.random.randn(m*p, R) * 0.1
            W_r = np.random.randn(p*n, R) * 0.1
            x_r = np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()])
            
            try:
                res = minimize(objective, x_r, method='BFGS', options={'maxiter': 100, 'disp': False})
                if res.fun < best_loss:
                    best_loss = res.fun
                    best_x = res.x
            except Exception:
                continue
        
        if best_x is not None and best_loss < 1e-8:
            U = best_x[:n*m*R].reshape(n*m, R)
            V = best_x[n*m*R:2*n*m*R].reshape(m*p, R)
            W = best_x[2*n*m*R:].reshape(p*n, R)
            
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            if verify(U_rounded, V_rounded, W_rounded, atol=1e-6):
                return U_rounded, V_rounded, W_rounded
            else:
                return U, V, W
                
        return None

    # Try rank 25
    if time.time() - start_time < DEADLINE - 30:
        result_25 = try_find_rank(25, max_restarts=5)
        if result_25:
            U25, V25, W25 = result_25
            if verify(U25, V25, W25):
                return {"U": U25.tolist(), "V": V25.tolist(), "W": W25.tolist()}

    # Try rank 24
    if time.time() - start_time < DEADLINE - 20:
        result_24 = try_find_rank(24, max_restarts=5)
        if result_24:
            U24, V24, W24 = result_24
            if verify(U24, V24, W24):
                return {"U": U24.tolist(), "V": V24.tolist(), "W": W24.tolist()}

    # Try rank 23
    if time.time() - start_time < DEADLINE - 10:
        result_23 = try_find_rank(23, max_restarts=5)
        if result_23:
            U23, V23, W23 = result_23
            if verify(U23, V23, W23):
                return {"U": U23.tolist(), "V": V23.tolist(), "W": W23.tolist()}

    # Fallback to rank 26
    U26, V26, W26 = rank26()
    if verify(U26, V26, W26):
        return {"U": U26.tolist(), "V": V26.tolist(), "W": W26.tolist()}
    
    # Fallback to rank 27
    U27, V27, W27 = rank27()
    return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
