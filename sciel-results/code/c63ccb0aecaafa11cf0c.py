"""Matrix multiplication 2,4,4 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = 2, 4, 4
    
    # Build the target tensor T[a,b,c]
    # T[a,b,c] = 1 iff a = i*m+j, b = j*p+k, c = k*n+i for some valid i,j,k
    # a in [0, n*m), b in [0, m*p), c in [0, p*n)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try to find low-rank decomposition
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    def residual(params, rank):
        """Compute residual for given parameters and rank."""
        U = params[0:rank * n * m].reshape(rank, n * m).T  # (n*m, rank)
        V = params[rank * n * m:2 * rank * n * m + rank * m * p - rank * n * m].reshape(rank, m * p).T
        W = params[2 * rank * n * m + rank * m * p - rank * n * m:]
        W = W.reshape(rank, p * n).T
        
        # Compute sum of outer products
        # T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Use einsum
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((T - T_recon) ** 2)
    
    # Try ranks from low to high
    best_result = None
    for rank in [20, 24, 28, 32]:
        # Try multiple random initializations
        found = False
        for trial in range(10):
            np.random.seed(trial * 42 + rank)
            # Initialize with small random values
            U_init = np.random.randn(n * m, rank) * 0.1
            V_init = np.random.randn(m * p, rank) * 0.1
            W_init = np.random.randn(p * n, rank) * 0.1
            
            params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Use L-BFGS-B
            result = minimize(
                residual,
                params,
                args=(rank,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            
            if result.fun < 1e-6:
                # Found a valid decomposition
                U_opt = result.x[0:rank * n * m].reshape(rank, n * m).T
                V_opt = result.x[rank * n * m:2 * rank * n * m + rank * m * p - rank * n * m].reshape(rank, m * p).T
                W_opt = result.x[2 * rank * n * m + rank * m * p - rank * n * m:].reshape(rank, p * n).T
                
                # Round to nearest half-integer if close
                def round_to_half(x):
                    return np.round(x * 2) / 2
                
                U_rounded = round_to_half(U_opt)
                V_rounded = round_to_half(V_opt)
                W_rounded = round_to_half(W_opt)
                
                # Verify rounded solution
                T_recon = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
                if np.max(np.abs(T - T_recon)) < 1e-9:
                    return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                
                # If rounding doesn't work, use the unrounded solution
                return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
        
        if not found and best_result is None:
            # Fallback to schoolbook
            pass
    
    # Fallback: schoolbook rank 32
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
