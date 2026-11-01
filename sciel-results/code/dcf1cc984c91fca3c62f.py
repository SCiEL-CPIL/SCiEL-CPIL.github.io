import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Precompute the target tensor T
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    T_flat = T.flatten()
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Try ranks from 17 down to 9
    # Schoolbook is 18. We want to find r < 18.
    # Theoretical lower bound is max(nm, mp, pn) = 9.
    
    for rank in range(17, 8, -1):
        if time.time() > deadline - 15:
            break
            
        best_sol = None
        best_cost = np.inf
        
        # Multiple random restarts
        max_restarts = 20
        for trial in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(42 + rank * 1000 + trial)
            # Scale initialization to avoid numerical issues
            U0 = np.random.randn(nm, rank) * 0.5
            V0 = np.random.randn(mp, rank) * 0.5
            W0 = np.random.randn(pn, rank) * 0.5
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            def residual(x):
                U = x[:nm*rank].reshape(nm, rank)
                V = x[nm*rank:nm*rank + mp*rank].reshape(mp, rank)
                W = x[nm*rank + mp*rank:].reshape(pn, rank)
                
                R = np.einsum('as,bs,cs->abc', U, V, W)
                return (R - T).flatten()
            
            try:
                sol = least_squares(residual, x0, max_nfev=500, method='trf', ftol=1e-10, xtol=1e-10, gtol=1e-10)
                cost = np.sum(sol.fun**2)
                if cost < best_cost:
                    best_cost = cost
                    best_sol = sol.x
                    if cost < 1e-8:
                        break
            except Exception:
                pass
        
        if best_sol is not None and best_cost < 1e-6:
            U = best_sol[:nm*rank].reshape(nm, rank)
            V = best_sol[nm*rank:nm*rank + mp*rank].reshape(mp, rank)
            W = best_sol[nm*rank + mp*rank:].reshape(pn, rank)
            
            # Try to round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            R_check = np.einsum('as,bs,cs->abc', U_rounded, V_rounded, W_rounded)
            if np.allclose(R_check, T, atol=1e-9):
                U_out = [[float(x) for x in row] for row in U_rounded]
                V_out = [[float(x) for x in row] for row in V_rounded]
                W_out = [[float(x) for x in row] for row in W_rounded]
                return {"U": U_out, "V": V_out, "W": W_out}
            
            # If rounding doesn't work, try integer rounding
            U_int = np.round(U).astype(int)
            V_int = np.round(V).astype(int)
            W_int = np.round(W).astype(int)
            
            R_check2 = np.einsum('as,bs,cs->abc', U_int.astype(float), V_int.astype(float), W_int.astype(float))
            if np.allclose(R_check2, T, atol=1e-9):
                U_out = [[float(x) for x in row] for row in U_int]
                V_out = [[float(x) for x in row] for row in V_int]
                W_out = [[float(x) for x in row] for row in W_int]
                return {"U": U_out, "V": V_out, "W": W_out}
            
            # If neither works, return the unrounded solution as float
            # But the problem requires integer/half-integer entries
            # So we need to find a valid decomposition
            # Let's try a local search to snap to half-integers
            for _ in range(100):
                if time.time() > deadline - 5:
                    break
                # Randomly adjust one entry by ±0.5
                mat_idx = np.random.randint(3)
                if mat_idx == 0:
                    r_idx, c_idx = np.random.randint(nm), np.random.randint(rank)
                    U_rounded[r_idx, c_idx] += np.random.choice([-0.5, 0.5])
                elif mat_idx == 1:
                    r_idx, c_idx = np.random.randint(mp), np.random.randint(rank)
                    V_rounded[r_idx, c_idx] += np.random.choice([-0.5, 0.5])
                else:
                    r_idx, c_idx = np.random.randint(pn), np.random.randint(rank)
                    W_rounded[r_idx, c_idx] += np.random.choice([-0.5, 0.5])
                
                R_check3 = np.einsum('as,bs,cs->abc', U_rounded, V_rounded, W_rounded)
                if np.allclose(R_check3, T, atol=1e-9):
                    U_out = [[float(x) for x in row] for row in U_rounded]
                    V_out = [[float(x) for x in row] for row in V_rounded]
                    W_out = [[float(x) for x in row] for row in W_rounded]
                    return {"U": U_out, "V": V_out, "W": W_out}

    # Fallback: Schoolbook decomposition of rank n*m*p
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
# EVOLVE-BLOCK-END
