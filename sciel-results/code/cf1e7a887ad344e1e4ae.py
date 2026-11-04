import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return {'U': U, 'V': V, 'W': W}
    
    best = schoolbook()
    best_rank = n * m * p
    deadline = time.time() + 140  # safety margin before 150s kill
    
    def tensor_residual(params, rank):
        """Compute residual for a given parameter vector and rank."""
        U = params[:nm*rank].reshape(nm, rank)
        V = params[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
        W = params[nm*rank+mp*rank:].reshape(pn, rank)
        
        # Compute tensor: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        residual = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    residual[i,j,k] = val - 1.0
        return residual.ravel()
    
    def try_rank(rank, max_iter=100):
        """Try to find a rank-r decomposition."""
        if rank >= best_rank:
            return None
        if time.time() > deadline:
            return None
            
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            np.random.seed(rank * 1000 + attempt * 7 + 42)
            params = np.random.randn(nm*rank + mp*rank + pn*rank) * 0.5
            
            try:
                result = least_squares(
                    tensor_residual, params, args=(rank,),
                    method='lm', max_nfev=max_iter, ftol=1e-10, xtol=1e-10, gtol=1e-10
                )
            except Exception:
                continue
            
            err = np.max(np.abs(result.fun)) if hasattr(result, 'fun') else float('inf')
            if err < 1e-8:
                U = result.x[:nm*rank].reshape(nm, rank)
                V = result.x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                W = result.x[nm*rank+mp*rank:].reshape(pn, rank)
                return {'U': U, 'V': V, 'W': W}
        
        return None
    
    # Search for lower ranks
    # For 3×3×5, let's try from a reasonable lower bound
    # Theoretical lower bound: rank >= max(n*m, m*p, p*n) is not valid
    # A better lower bound comes from the dimension of the space
    # Let's just try a range of ranks
    for rank in range(18, best_rank):
        if time.time() > deadline:
            break
        result = try_rank(rank)
        if result is not None:
            best = result
            best_rank = rank
            # Continue searching for even lower ranks
            if rank <= 20:
                break
    
    # Convert to lists
    return {'U': best['U'].tolist(), 'V': best['V'].tolist(), 'W': best['W'].tolist()}
