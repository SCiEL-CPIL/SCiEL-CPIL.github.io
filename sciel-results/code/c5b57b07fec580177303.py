import numpy as np
import time

def construct():
    n, m, p = 2, 3, 3
    deadline = time.time() + 140
    
    # Precompute the target tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a,b,c] = 1
    
    nz_idx = np.argwhere(T != 0)
    a_idx = nz_idx[:, 0]
    b_idx = nz_idx[:, 1]
    c_idx = nz_idx[:, 2]
    
    def fast_als_decompose(rank, max_iters=50, tol=1e-6):
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        prev_cost = np.inf
        for it in range(max_iters):
            # Update U
            U_new = np.zeros_like(U)
            for a in range(n*m):
                mask = (a_idx == a)
                if not np.any(mask):
                    continue
                b_sub = b_idx[mask]
                c_sub = c_idx[mask]
                D = V[b_sub, :] * W[c_sub, :]
                try:
                    U_new[a, :] = np.linalg.lstsq(D, np.ones(len(b_sub)), rcond=None)[0]
                except:
                    U_new[a, :] = U[a, :]
            U = U_new
            
            # Update V
            V_new = np.zeros_like(V)
            for b in range(m*p):
                mask = (b_idx == b)
                if not np.any(mask):
                    continue
                a_sub = a_idx[mask]
                c_sub = c_idx[mask]
                D = U[a_sub, :] * W[c_sub, :]
                try:
                    V_new[b, :] = np.linalg.lstsq(D, np.ones(len(a_sub)), rcond=None)[0]
                except:
                    V_new[b, :] = V[b, :]
            V = V_new
            
            # Update W
            W_new = np.zeros_like(W)
            for c in range(p*n):
                mask = (c_idx == c)
                if not np.any(mask):
                    continue
                a_sub = a_idx[mask]
                b_sub = b_idx[mask]
                D = U[a_sub, :] * V[b_sub, :]
                try:
                    W_new[c, :] = np.linalg.lstsq(D, np.ones(len(a_sub)), rcond=None)[0]
                except:
                    W_new[c, :] = W[c, :]
            W = W_new
            
            # Compute cost
            R_nz = np.zeros(len(nz_idx))
            for s in range(rank):
                R_nz += U[a_idx, s] * V[b_idx, s] * W[c_idx, s]
            cost = np.sum((R_nz - 1.0)**2)
            
            if abs(prev_cost - cost) < tol:
                break
            prev_cost = cost
        
        return U, V, W, cost

    def try_rank(rank, num_restarts=10, time_budget=10):
        start_time = time.time()
        best = None
        best_cost = np.inf
        
        for r in range(num_restarts):
            if time.time() - start_time > time_budget:
                break
            np.random.seed(1000 + rank * 10 + r)
            U, V, W, cost = fast_als_decompose(rank, max_iters=100, tol=1e-8)
            if cost < best_cost:
                best_cost = cost
                best = (U, V, W)
            if cost < 1e-6:
                break
        
        if best is None or best_cost > 1e-4:
            return None, best_cost
        
        U, V, W = best
        
        # Snap to half-integers
        U_half = np.round(U * 2) / 2
        V_half = np.round(V * 2) / 2
        W_half = np.round(W * 2) / 2
        
        R_check = np.einsum('as,bs,cs->abc', U_half, V_half, W_half)
        if np.allclose(R_check, T, atol=1e-9):
            return (U_half, V_half, W_half), best_cost
        
        # Try integer rounding
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        R_check2 = np.einsum('as,bs,cs->abc', U_int, V_int, W_int)
        if np.allclose(R_check2, T, atol=1e-9):
            return (U_int, V_int, W_int), best_cost
        
        return None, best_cost

    # Try rank 15
    sol, cost = try_rank(15, num_restarts=20, time_budget=15)
    if sol is not None:
        U, V, W = sol
        U_out = [[float(x) for x in row] for row in U]
        V_out = [[float(x) for x in row] for row in V]
        W_out = [[float(x) for x in row] for row in W]
        return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
