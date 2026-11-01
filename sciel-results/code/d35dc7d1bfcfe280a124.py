"""Matrix multiplication 2x3x4 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    # Schoolbook decomposition as fallback
    def schoolbook():
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
    
    if time.time() > deadline:
        return schoolbook()
    
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor
    target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1
    
    def compute_tensor(U, V, W, rank):
        T = np.zeros((nm, mp, pn))
        for r in range(rank):
            T += np.einsum('ij,jk->ijk', U[r].reshape(n, m), V[r].reshape(m, p), W[r].reshape(p, n))
        return T
    
    def try_rank(rank):
        if time.time() > deadline:
            return None
        
        from scipy.optimize import minimize
        
        def residual(params):
            U = params[:rank * nm].reshape(rank, nm)
            V = params[rank * nm:rank * nm + rank * mp].reshape(rank, mp)
            W = params[rank * nm + rank * mp:].reshape(rank, pn)
            T = compute_tensor(U, V, W, rank)
            return np.sum((T - target) ** 2)
        
        # Try a few random initializations
        for trial in range(3):
            if time.time() > deadline:
                return None
            np.random.seed(trial * 1000 + rank)
            params = np.random.randn(rank * (nm + mp + pn)) * 0.5
            
            result = minimize(residual, params, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-8})
            
            if result.fun < 1e-6:
                U = result.x[:rank * nm].reshape(rank, nm)
                V = result.x[rank * nm:rank * nm + rank * mp].reshape(rank, mp)
                W = result.x[rank * nm + rank * mp:].reshape(rank, pn)
                
                # Round to half-integers
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                
                # Verify
                T = compute_tensor(U_round, V_round, W_round, rank)
                if np.allclose(T, target, atol=1e-10):
                    U_frac = [[Fraction(int(round(x * 2))) / 2 for x in U_round[r]] for r in range(rank)]
                    V_frac = [[Fraction(int(round(x * 2))) / 2 for x in V_round[r]] for r in range(rank)]
                    W_frac = [[Fraction(int(round(x * 2))) / 2 for x in W_round[r]] for r in range(rank)]
                    return {"U": U_frac, "V": V_frac, "W": W_frac}
        
        return None
    
    # Try ranks from 12 to 23
    for rank in range(12, 24):
        if time.time() > deadline:
            break
        result = try_rank(rank)
        if result is not None:
            return result
    
    return schoolbook()
