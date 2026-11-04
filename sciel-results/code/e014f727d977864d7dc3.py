import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    def check_solution(U, V, W):
        result = np.einsum('ac,bc,dc->abd', U, V, W)
        return np.max(np.abs(result - T)) < 1e-10
    
    def als_decompose(r, n_iters=300, seed=42):
        rng = np.random.RandomState(seed)
        U = rng.randn(nm, r) * 0.5
        V = rng.randn(mp, r) * 0.5
        W = rng.randn(pn, r) * 0.5
        
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(n_iters):
            if time.time() > deadline:
                break
            # Solve for W
            W_new = np.zeros((pn, r))
            for c in range(pn):
                A_c = np.outer(U.flatten(), V.flatten()).reshape(nm * mp, r) if False else None
                # Build A_c: shape (nm*mp, r), A_c[(a,b), c'] = U[a,c'] * V[b,c']
                A_c = np.zeros((nm * mp, r))
                for cp in range(r):
                    A_c[:, cp] = (U[:, cp] * V[:, cp]).flatten()
                rhs = T[:, :, c].flatten()
                sol, _, _, _ = np.linalg.lstsq(A_c, rhs, rcond=None)
                W_new[c, :] = sol
            W = W_new
            
            # Solve for U
            U_new = np.zeros((nm, r))
            for a in range(nm):
                B_a = np.zeros((mp * pn, r))
                for cp in range(r):
                    B_a[:, cp] = (V[:, cp] * W[:, cp]).flatten()
                rhs = T[a, :, :].flatten()
                sol, _, _, _ = np.linalg.lstsq(B_a, rhs, rcond=None)
                U_new[a, :] = sol
            U = U_new
            
            # Solve for V
            V_new = np.zeros((mp, r))
            for b in range(mp):
                C_b = np.zeros((nm * pn, r))
                for cp in range(r):
                    C_b[:, cp] = (U[:, cp] * W[:, cp]).flatten()
                rhs = T[:, b, :].flatten()
                sol, _, _, _ = np.linalg.lstsq(C_b, rhs, rcond=None)
                V_new[b, :] = sol
            V = V_new
            
            err = np.max(np.abs(np.einsum('ac,bc,dc->abd', U, V, W) - T))
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        return best_U, best_V, best_W, best_err
    
    best_solution = None
    best_rank = None
    
    for r in range(18, 31):
        if time.time() > deadline - 15:
            break
        for seed in range(100):
            if time.time() > deadline - 10:
                break
            U, V, W, err = als_decompose(r, n_iters=150, seed=seed)
            if err < 1e-8 and check_solution(U, V, W):
                best_solution = (U, V, W)
                best_rank = r
                break
        if best_solution:
            break
    
    if best_solution:
        U, V, W = best_solution
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
