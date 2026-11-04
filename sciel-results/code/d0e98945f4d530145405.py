import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    
    def schoolbook():
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    def round_to_half(x):
        return np.round(2*x) / 2.0
    
    def verify(U, V, W):
        residual = np.einsum('at,bt,ct->abc', U, V, W)
        return np.max(np.abs(residual - T)) < 1e-9
    
    def als_decompose(r, seed=42, max_iters=100, tol=1e-10):
        if time.time() > deadline - 10:
            return None
        
        np.random.seed(seed)
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            U_new = np.zeros_like(U)
            for a in range(n*m):
                B_a = np.einsum('bt,ct->bct', V, W).reshape(-1, r)
                rhs = T[a, :, :].ravel()
                U_new[a, :] = np.linalg.lstsq(B_a, rhs, rcond=None)[0]
            U = U_new
            
            V_new = np.zeros_like(V)
            for b in range(m*p):
                B_b = np.einsum('at,ct->act', U, W).reshape(-1, r)
                rhs = T[:, b, :].ravel()
                V_new[b, :] = np.linalg.lstsq(B_b, rhs, rcond=None)[0]
            V = V_new
            
            W_new = np.zeros_like(W)
            for c in range(p*n):
                B_c = np.einsum('at,bt->abt', U, V).reshape(-1, r)
                rhs = T[:, :, c].ravel()
                W_new[c, :] = np.linalg.lstsq(B_c, rhs, rcond=None)[0]
            W = W_new
            
            residual = np.einsum('at,bt,ct->abc', U, V, W)
            max_err = np.max(np.abs(residual - T))
            
            if max_err < tol:
                # Try rounding
                U_r = round_to_half(U)
                V_r = round_to_half(V)
                W_r = round_to_half(W)
                if verify(U_r, V_r, W_r):
                    return U_r, V_r, W_r, r
                # Also try integer rounding
                U_i = np.round(U)
                V_i = np.round(V)
                W_i = np.round(W)
                if verify(U_i, V_i, W_i):
                    return U_i, V_i, W_i, r
                # Return unrounded (shouldn't happen per problem statement)
                return U, V, W, r
        
        return None
    
    for r in range(60, 49, -1):
        if time.time() > deadline - 15:
            break
        result = als_decompose(r, seed=r*1000+1)
        if result is not None:
            U, V, W, rank = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
