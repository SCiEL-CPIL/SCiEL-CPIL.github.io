import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Precompute target tensor T
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a, b, c = i * m + j, j * p + k, k * n + i
                T[a, b, c] = 1.0
    T_flat = T.flatten()
    
    start_time = time.time()
    deadline = start_time + 140
    
    def verify(U, V, W):
        """Check if decomposition is exact with half-integer entries"""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        R = np.einsum('as,bs,cs->abc', U_np, V_np, W_np)
        if not np.allclose(R, T, atol=1e-9):
            return False
        # Check half-integer constraint
        for mat in [U_np, V_np, W_np]:
            doubled = mat * 2
            if not np.allclose(doubled, np.round(doubled), atol=1e-6):
                return False
        return True
    
    def round_half(x):
        """Round to nearest half-integer"""
        return round(x * 2) / 2.0
    
    def try_rank(rank, max_trials=50, seed_base=0):
        """Try to find a rank-r decomposition using least squares"""
        if time.time() > deadline - 10:
            return None
        
        best_cost = np.inf
        best_x = None
        
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(seed_base + trial)
            
            # Border-rank inspired initialization:
            # Start with a structured base + random perturbation
            U0 = np.zeros((nm, rank))
            V0 = np.zeros((mp, rank))
            W0 = np.zeros((pn, rank))
            
            # Structured part: schoolbook-like but compressed
            col = 0
            for j in range(m):
                for k in range(p):
                    if col >= rank:
                        break
                    # First component: i=0
                    U0[0 * m + j, col] = 1.0
                    V0[j * p + k, col] = 1.0
                    W0[k * n + 0, col] = 1.0
                    col += 1
                    # Second component: i=1
                    if col < rank:
                        U0[1 * m + j, col] = 1.0
                        V0[j * p + k, col] = 1.0
                        W0[k * n + 1, col] = 1.0
                        col += 1
            
            # Add random perturbation for remaining columns
            for s in range(col, rank):
                U0[:, s] = np.random.randn(nm) * 0.3
                V0[:, s] = np.random.randn(mp) * 0.3
                W0[:, s] = np.random.randn(pn) * 0.3
            
            # Also try purely random initializations for some trials
            if trial % 5 == 0:
                U0 = np.random.randn(nm, rank) * 0.5
                V0 = np.random.randn(mp, rank) * 0.5
                W0 = np.random.randn(pn, rank) * 0.5
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            def residual(x):
                U = x[:nm * rank].reshape(nm, rank)
                V = x[nm * rank:nm * rank + mp * rank].reshape(mp, rank)
                W = x[nm * rank + mp * rank:].reshape(pn, rank)
                R = np.einsum('as,bs,cs->abc', U, V, W)
                return (R - T).flatten()
            
            try:
                sol = least_squares(residual, x0, max_nfev=300, method='trf',
                                    ftol=1e-10, xtol=1e-10, gtol=1e-10)
                cost = np.sum(sol.fun ** 2)
                if cost < best_cost:
                    best_cost = cost
                    best_x = sol.x.copy()
                    if cost < 1e-12:
                        break
            except Exception:
                pass
        
        if best_x is not None and best_cost < 1e-8:
            U = best_x[:nm * rank].reshape(nm, rank)
            V = best_x[nm * rank:nm * rank + mp * rank].reshape(mp, rank)
            W = best_x[nm * rank + mp * rank:].reshape(pn, rank)
            
            # Round to half-integers
            U_r = [[round_half(x) for x in row] for row in U]
            V_r = [[round_half(x) for x in row] for row in V]
            W_r = [[round_half(x) for x in row] for row in W]
            
            if verify(U_r, V_r, W_r):
                return {"U": U_r, "V": V_r, "W": W_r}
            
            # Try without rounding - maybe the solution is already integer/half-integer
            if verify(U.tolist(), V.tolist(), W.tolist()):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None
    
    # Try ranks from 9 (theoretical lower bound) up to 17
    # Theoretical lower bound: max(nm, mp, pn) = 9
    for rank in range(9, 18):
        if time.time() > deadline - 15:
            break
        
        # More trials for lower ranks (harder to find)
        if rank <= 12:
            trials = 80
        elif rank <= 15:
            trials = 50
        else:
            trials = 30
        
        result = try_rank(rank, max_trials=trials, seed_base=rank * 1000)
        if result is not None:
            return result
    
    # Fallback: Schoolbook decomposition of rank n*m*p = 18
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
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
