"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 24
    pn = p * n  # 16
    total_eq = n * m * p  # 48
    
    def try_decompose(rank):
        """Try to find a decomposition of given rank."""
        # Variables: U (nm x rank), V (mp x rank), W (pn x rank)
        # Total variables: (nm + mp + pn) * rank
        
        def residuals(params):
            # Unpack
            U_flat = params[:nm * rank]
            V_flat = params[nm * rank : nm * rank + mp * rank]
            W_flat = params[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
            
            U = U_flat.reshape((nm, rank))
            V = V_flat.reshape((mp, rank))
            W = W_flat.reshape((pn, rank))
            
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        res.append(val - 1.0)
            return np.array(res)
        
        # Initial guess: schoolbook-like
        np.random.seed(42)
        U0 = np.zeros((nm, rank))
        V0 = np.zeros((mp, rank))
        W0 = np.zeros((pn, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < rank:
                        U0[i*m+j, col] = 1.0
                        V0[j*p+k, col] = 1.0
                        W0[k*n+i, col] = 1.0
                        col += 1
        
        # If rank < total_eq, we need to distribute
        if rank < total_eq:
            # Scale down
            scale = rank / total_eq
            U0 *= scale
            V0 *= scale
            W0 *= scale
        
        params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            result = least_squares(residuals, params0, max_nfev=10000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if result.success or np.max(np.abs(result.fun)) < 1e-8:
                U = result.x[:nm*rank].reshape((nm, rank))
                V = result.x[nm*rank:nm*rank+mp*rank].reshape((mp, rank))
                W = result.x[nm*rank+mp*rank:].reshape((pn, rank))
                return U, V, W
        except:
            pass
        return None
    
    # Try ranks from schoolbook down
    rank_school = total_eq
    best = None
    
    for r in range(rank_school, max(rank_school - 5, 0), -1):
        result = try_decompose(r)
        if result is not None:
            best = result
            break
    
    if best is None:
        # Fallback to schoolbook
        U = np.zeros((nm, rank_school))
        V = np.zeros((mp, rank_school))
        W = np.zeros((pn, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        best = (U, V, W)
    
    U, V, W = best
    rank = U.shape[1]
    
    U_list = [[float(U[i, c]) for c in range(rank)] for i in range(nm)]
    V_list = [[float(V[i, c]) for c in range(rank)] for i in range(mp)]
    W_list = [[float(W[i, c]) for c in range(rank)] for i in range(pn)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
