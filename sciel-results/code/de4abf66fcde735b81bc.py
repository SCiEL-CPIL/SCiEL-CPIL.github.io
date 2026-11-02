import time
import random
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 3]

def _verify(U, V, W, n, m, p, tol=1e-9):
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _round_half(mat):
    return [[round(x * 2) / 2.0 for x in row] for row in mat]

def _residual(U, V, W, n, m, p):
    r = len(U[0])
    res = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                res.append(s - 1.0)
    return np.array(res)

def _als_step_U(V, W, n, m, p, r):
    U = [[0.0]*r for _ in range(n*m)]
    for i in range(n):
        for j in range(m):
            A = np.zeros((p, r))
            for k in range(p):
                for c in range(r):
                    A[k, c] = V[j*p+k][c] * W[k*n+i][c]
            b = np.ones(p)
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            for c in range(r):
                U[i*m+j][c] = float(sol[c])
    return U

def _als_step_V(U, W, n, m, p, r):
    V = [[0.0]*r for _ in range(m*p)]
    for j in range(m):
        for k in range(p):
            A = np.zeros((n, r))
            for i in range(n):
                for c in range(r):
                    A[i, c] = U[i*m+j][c] * W[k*n+i][c]
            b = np.ones(n)
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            for c in range(r):
                V[j*p+k][c] = float(sol[c])
    return V

def _als_step_W(U, V, n, m, p, r):
    W = [[0.0]*r for _ in range(p*n)]
    for k in range(p):
        for i in range(n):
            A = np.zeros((m, r))
            for j in range(m):
                for c in range(r):
                    A[j, c] = U[i*m+j][c] * V[j*p+k][c]
            b = np.ones(m)
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            for c in range(r):
                W[k*n+i][c] = float(sol[c])
    return W

def _als_optimize(U0, V0, W0, n, m, p, r, iters, deadline):
    U, V, W = U0, V0, W0
    for _ in range(iters):
        if time.time() > deadline:
            break
        U = _als_step_U(V, W, n, m, p, r)
        V = _als_step_V(U, W, n, m, p, r)
        W = _als_step_W(U, V, n, m, p, r)
    return U, V, W

def _rank11_solution():
    n, m, p = 2, 2, 3
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def set_u(i, j, r, val): U[i*m+j][r] = val
    def set_v(j, k, r, val): V[j*p+k][r] = val
    def set_w(k, i, r, val): W[k*n+i][r] = val

    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1); set_w(1, 1, 0, 1)

    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1); set_w(1, 1, 1, -1)

    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1); set_w(1, 1, 2, 1)

    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1); set_w(0, 1, 3, 1)

    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1); set_w(1, 0, 4, 1)

    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)

    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    set_w(2, 0, 7, 1)

    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1)

    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    set_w(2, 1, 9, 1)

    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    set_w(2, 1, 10, 1)

    return {"U": U, "V": V, "W": W}

def _try_rank10_pruning(deadline):
    n, m, p = 2, 2, 3
    r = 10
    sol11 = _rank11_solution()
    U11 = np.array(sol11["U"], dtype=float)
    V11 = np.array(sol11["V"], dtype=float)
    W11 = np.array(sol11["W"], dtype=float)
    
    # Try removing each term from the rank 11 solution and re-optimizing
    for term_to_remove in range(11):
        if time.time() > deadline:
            break
            
        # Create rank 10 matrices by removing the term
        U_init = np.delete(U11, term_to_remove, axis=1)
        V_init = np.delete(V11, term_to_remove, axis=1)
        W_init = np.delete(W11, term_to_remove, axis=1)
        
        U, V, W = _als_optimize(U_init.tolist(), V_init.tolist(), W_init.tolist(), n, m, p, r, iters=50, deadline=deadline)
        
        Ur, Vr, Wr = _round_half(U), _round_half(V), _round_half(W)
        
        if _verify(Ur, Vr, Wr, n, m, p, tol=1e-9):
            return Ur, Vr, Wr
            
        # If ALS didn't work, try LM polish
        if time.time() < deadline:
            try:
                def _flatten(U, V, W):
                    return np.concatenate([np.array(U).flatten(), np.array(V).flatten(), np.array(W).flatten()])
                
                def _unflatten(x):
                    U = x[:n*m*r].reshape(n*m, r)
                    V = x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                    W = x[n*m*r+m*p*r:].reshape(p*n, r)
                    return U, V, W
                
                def _objective(x):
                    U, V, W = _unflatten(x)
                    res = []
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(r))
                                res.append(s - 1.0)
                    return res
                
                x0 = _flatten(U, V, W)
                result = least_squares(_objective, x0, method='lm', max_nfev=5000, xtol=1e-14, ftol=1e-14)
                U, V, W = _unflatten(result.x)
                Ur, Vr, Wr = _round_half(U.tolist()), _round_half(V.tolist()), _round_half(W.tolist())
                if _verify(Ur, Vr, Wr, n, m, p, tol=1e-9):
                    return Ur, Vr, Wr
            except Exception:
                pass
                
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n, m, p = DIMENSIONS
    
    best = _rank11_solution()
    
    if time.time() < deadline:
        result10 = _try_rank10_pruning(deadline)
        if result10 is not None:
            best = {"U": result10[0], "V": result10[1], "W": result10[2]}
    
    return best
