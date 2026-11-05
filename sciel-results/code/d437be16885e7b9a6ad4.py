import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    Y = np.zeros((n, m, m, p))
    for i in range(n):
        for j in range(m):
            for j_prime in range(m):
                for k in range(p):
                    if j == j_prime:
                        Y[i, j, j_prime, k] = 1.0
    Y_flat = Y.flatten()

    def compute_error(params, rank):
        nm = n * m
        mp = m * p
        pn = p * n
        U = params[:nm * rank].reshape(n, m, rank)
        V = params[nm * rank:nm * rank + mp * rank].reshape(m, p, rank)
        W = params[nm * rank + mp * rank:].reshape(p, n, rank)
        
        D = np.einsum('ijc,jkc,kic->ijjk', U, V, W)
        err = D.flatten() - Y_flat
        return np.sum(err**2)

    def get_matrices(params, rank):
        nm = n * m
        mp = m * p
        pn = p * n
        U = params[:nm * rank].reshape(n * m, rank)
        V = params[nm * rank:nm * rank + mp * rank].reshape(m * p, rank)
        W = params[nm * rank + mp * rank:].reshape(p * n, rank)
        return U, V, W

    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is exact."""
        D = np.einsum('ijc,jkc,kic->ijjk', U, V, W)
        err = np.max(np.abs(D - Y))
        return err < 1e-6, err

    best_rank = n * m * p
    best_U = np.zeros((n * m, best_rank))
    best_V = np.zeros((m * p, best_rank))
    best_W = np.zeros((p * n, best_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j, col] = 1
                best_V[j * p + k, col] = 1
                best_W[k * n + i, col] = 1
                col += 1

    # Search for better rank with structured initialization
    search_deadline = start_time + 130
    
    # Try ranks from low to high, with multiple attempts per rank
    # Focus on ranks that are achievable: try 21-40 range first
    ranks_to_try = list(range(21, 45)) + list(range(45, best_rank))
    
    for rank in ranks_to_try:
        if time.time() > search_deadline:
            break
        
        found = False
        max_attempts = 3 if rank <= 35 else 2
        
        for attempt in range(max_attempts):
            if time.time() > search_deadline:
                break
            
            # Structured initialization: mix of random and identity-like
            if attempt == 0:
                # Identity-like initialization
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                W = np.zeros((p * n, rank))
                # Fill first min(rank, n*m*p) columns with identity pattern
                for c in range(min(rank, n * m * p)):
                    i = c // (m * p)
                    rem = c % (m * p)
                    j = rem // p
                    k = rem % p
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                # Add small random perturbation to remaining columns
                for c in range(min(rank, n * m * p), rank):
                    U[:, c] = np.random.randn(n * m) * 0.1
                    V[:, c] = np.random.randn(m * p) * 0.1
                    W[:, c] = np.random.randn(p * n) * 0.1
            elif attempt == 1:
                # Random initialization
                U = np.random.randn(n * m, rank)
                V = np.random.randn(m * p, rank)
                W = np.random.randn(p * n, rank)
            else:
                # Scale-mixed initialization
                U = np.random.randn(n * m, rank) * 0.5
                V = np.random.randn(m * p, rank) * 0.5
                W = np.random.randn(p * n, rank) * 0.5
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(compute_error, x0, args=(rank,), method='L-BFGS-B', 
                             options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
                if res.fun < 1e-4:
                    U_opt, V_opt, W_opt = get_matrices(res.x, rank)
                    
                    # Try rounding to half-integers
                    U_rounded = np.round(U_opt * 2) / 2
                    V_rounded = np.round(V_opt * 2) / 2
                    W_rounded = np.round(W_opt * 2) / 2
                    
                    valid_rounded, err_rounded = verify_decomposition(U_rounded, V_rounded, W_rounded, rank)
                    
                    if valid_rounded:
                        best_rank = rank
                        best_U = U_rounded
                        best_V = V_rounded
                        best_W = W_rounded
                        found = True
                        break
                    elif res.fun < 1e-8:
                        valid_opt, err_opt = verify_decomposition(U_opt, V_opt, W_opt, rank)
                        if valid_opt:
                            best_rank = rank
                            best_U = U_opt
                            best_V = V_opt
                            best_W = W_opt
                            found = True
                            break
            except Exception:
                continue
        
        if found:
            # Continue searching for even lower rank
            continue
        else:
            # If we failed at this rank, try next
            pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
