"""Program for matrix multiplication tensor decomposition 2x4x7.
Uses alternating minimization and local search to find low-rank decompositions.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin: 110s < 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    def make_schoolbook():
        """Return the standard rank n*m*p decomposition."""
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
        return {"U": U, "V": V, "W": W}
    
    best_result = make_schoolbook()
    best_rank = n * m * p  # 56
    
    def verify(U, V, W, tol=1e-8):
        """Check if decomposition is valid."""
        r = U.shape[1]
        # Vectorized check
        # Res[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
        # We can check max absolute error
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U[i * m + j, :]
                # V[j*p:(j+1)*p, :] * W[i::n, :]
                # W indices for k=0..p-1 are k*n+i -> i, n+i, ...
                # So W[i::n] is correct.
                prod = V[j * p:(j + 1) * p, :] * W[i::n, :]
                dots = prod @ u_row
                err = np.max(np.abs(dots - 1.0))
                if err > max_err:
                    max_err = err
        return max_err < tol, max_err
    
    def try_alternating_minimization(rank, max_iters=200, tol=1e-7, seed=0, init=None):
        """Try alternating minimization to find a rank-r decomposition."""
        if time.time() > deadline:
            return None
        
        np.random.seed(seed)
        
        if init is not None:
            U = init["U"].copy()
            V = init["V"].copy()
            W = init["W"].copy()
            # If init rank is larger, we need to truncate or project.
            # For simplicity, we only call this with init of same rank or we handle it.
            # Here we assume init is for this rank or we random init.
            if U.shape[1] > rank:
                U = U[:, :rank]
                V = V[:, :rank]
                W = W[:, :rank]
            elif U.shape[1] < rank:
                # Pad with zeros
                U_new = np.zeros((U.shape[0], rank))
                U_new[:, :U.shape[1]] = U
                U = U_new
                V_new = np.zeros((V.shape[0], rank))
                V_new[:, :V.shape[1]] = V
                V = V_new
                W_new = np.zeros((W.shape[0], rank))
                W_new[:, :W.shape[1]] = W
                W = W_new
        else:
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
        
        best_res = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U
            for i in range(n):
                for j in range(m):
                    A = V[j * p:(j + 1) * p, :] * W[i::n, :]
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    B = U[j::m, :] * W[k * n:(k + 1) * n, :]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(B.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    C = U[i * m:(i + 1) * m, :] * V[k::p, :]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check residual
            if it % 5 == 4 or it == max_iters - 1:
                valid, max_res = verify(U, V, W)
                if max_res < best_res:
                    best_res = max_res
                    if max_res < tol:
                        return {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
        
        return None

    # Try to find lower rank decompositions
    # Start from schoolbook and try to reduce rank
    # We try ranks 55, 54, 53...
    # We use the schoolbook solution as a "warm start" by truncating it? 
    # Truncating schoolbook doesn't work directly. 
    # Instead, we try random AM for lower ranks.
    
    ranks_to_try = [55, 54, 53, 52]
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        found = None
        # Try a few seeds
        for seed in range(5):
            if time.time() > deadline - 2:
                break
            found = try_alternating_minimization(r, max_iters=100, seed=seed)
            if found is not None:
                best_rank = r
                best_result = found
                return best_result
    
    return best_result
# EVOLVE-BLOCK-END
