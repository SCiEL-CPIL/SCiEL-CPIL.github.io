import time
import random
import math
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 7]

def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    for i in range(n):
        for j in range(m):
            u_row = U_arr[i * m + j]
            for k in range(p):
                v_row = V_arr[j * p + k]
                w_row = W_arr[k * n + i]
                s = float(np.dot(u_row, v_row * w_row))
                if abs(s - 1.0) > tol:
                    return False
    return True

def _to_half_int_matrix(M, tol=1e-6):
    M = np.array(M, dtype=float)
    scaled = M * 2.0
    rounded = np.round(scaled)
    if np.max(np.abs(scaled - rounded)) > tol:
        return None
    return (rounded / 2.0)

def _residuals(x, n, m, p, rank):
    nm = n * m
    mp = m * p
    pn = p * n
    
    U = x[:rank*nm].reshape(nm, rank)
    V = x[rank*nm:rank*nm+rank*mp].reshape(mp, rank)
    W = x[rank*nm+rank*mp:].reshape(pn, rank)
    
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            T[i, j, :] = U_3d[i, j, :] @ (V_3d[j] * W_3d[:, i]).T
            
    return (T - 1.0).flatten()

def _try_rank(n, m, p, rank, deadline, max_restarts=5):
    nm = n * m
    mp = m * p
    pn = p * n
    total_vars = rank * (nm + mp + pn)
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            return None
        
        seed = restart * 12345 + 6789
        rng = np.random.RandomState(seed)
        
        # Initialize
        U = rng.randn(nm, rank)
        V = rng.randn(mp, rank)
        W = rng.randn(pn, rank)
        
        # Normalize to avoid huge/small numbers
        U = U / np.linalg.norm(U) * np.sqrt(rank)
        V = V / np.linalg.norm(V) * np.sqrt(rank)
        W = W / np.linalg.norm(W) * np.sqrt(rank)
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            res = least_squares(_residuals, x0, args=(n, m, p, rank), method='lm', max_nfev=1000)
        except Exception:
            continue
            
        if res.success or res.cost < 1e-8:
            # Extract solution
            U_sol = res.x[:rank*nm].reshape(nm, rank)
            V_sol = res.x[rank*nm:rank*nm+rank*mp].reshape(mp, rank)
            W_sol = res.x[rank*nm+rank*mp:].reshape(pn, rank)
            
            # Try rounding
            Uh = _to_half_int_matrix(U_sol, tol=1e-4)
            Vh = _to_half_int_matrix(V_sol, tol=1e-4)
            Wh = _to_half_int_matrix(W_sol, tol=1e-4)
            
            if Uh is not None and Vh is not None and Wh is not None:
                if _verify(Uh.tolist(), Vh.tolist(), Wh.tolist()):
                    return {"U": Uh.tolist(), "V": Vh.tolist(), "W": Wh.tolist()}
            
            # If not half-integers, check if it's a valid real solution
            # The task requires integer/half-integer. So we must return that.
            # If we can't round, this solution is useless for the specific constraint.
            # But maybe we can scale it?
            # The solution is unique up to scaling?
            # If we find a real solution, we can try to "snap" it.
            
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p # 56
    deadline = time.time() + 140
    
    # Schoolbook fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try lower ranks
    for rank in range(28, schoolbook_rank):
        if time.time() > deadline - 10:
            break
            
        res = _try_rank(n, m, p, rank, deadline, max_restarts=10)
        if res is not None:
            best_result = res
            break
            
    return best_result
