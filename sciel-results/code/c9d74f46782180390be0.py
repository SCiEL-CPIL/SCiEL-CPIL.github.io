import time
import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 40s margin for safety
    
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx, col] = 1.0
                    V[v_idx, col] = 1.0
                    W[w_idx, col] = 1.0
                    col += 1
        return U, V, W
    
    def verify(U, V, W, atol=1e-8):
        r = U.shape[1]
        # Check all n*m*p target equations equal 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                    if not np.isclose(np.sum(prod), 1.0, atol=atol):
                        return False
        return True
    
    def try_rank_reduction(target_rank, max_attempts=5):
        """Try to find a decomposition of the given target_rank using alternating least squares."""
        if target_rank >= n * m * p:
            return None
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            np.random.seed(42 + attempt * 13 + target_rank)
            
            # Initialize with random values
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Normalize columns to prevent scale drift
            for t in range(target_rank):
                norms = [np.linalg.norm(U[:, t]), np.linalg.norm(V[:, t]), np.linalg.norm(W[:, t])]
                if all(x > 1e-10 for x in norms):
                    U[:, t] /= np.linalg.norm(U[:, t])
                    V[:, t] /= np.linalg.norm(V[:, t])
                    W[:, t] /= np.linalg.norm(W[:, t])
            
            best_obj = np.inf
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Alternating least squares with multiple restarts
            for iteration in range(200):
                if time.time() > deadline:
                    break
                
                # Update U: for each (i,j), solve U[i*m+j, :] such that sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = 1
                # This is a linear system in U[i*m+j, :]
                for i in range(n):
                    for j in range(m):
                        # Build system: for each k, sum_t U[i*m+j,t] * (V[j*p+k,t] * W[k*n+i,t]) = 1
                        A = np.zeros((p, target_rank))
                        b = np.ones(p)
                        for k in range(p):
                            A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        # Solve least squares
                        U[i*m+j, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                
                # Update V: for each (j,k), solve V[j*p+k, :] such that sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = 1
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, target_rank))
                        b = np.ones(n)
                        for i in range(n):
                            A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        V[j*p+k, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                
                # Update W: for each (k,i), solve W[k*n+i, :] such that sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = 1
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, target_rank))
                        b = np.ones(m)
                        for j in range(m):
                            A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        W[k*n+i, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                
                # Compute objective
                errors = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            prod = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                            errors.append(np.sum(prod) - 1.0)
                
                obj = np.sum(np.array(errors) ** 2)
                
                if obj < best_obj:
                    best_obj = obj
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if best_obj < 1e-10:
                    break
            
            # Check if we found a valid solution
            if best_obj < 1e-8:
                # Try to round to half-integers
                U_round = np.round(best_U * 2) / 2
                V_round = np.round(best_V * 2) / 2
                W_round = np.round(best_W * 2) / 2
                
                if verify(U_round, V_round, W_round):
                    return U_round, V_round, W_round
                
                # If rounding doesn't work, try with original
                if verify(best_U, best_V, best_W):
                    return best_U, best_V, best_W
        
        return None
    
    # Get schoolbook decomposition
    U_sb, V_sb, W_sb = get_schoolbook()
    
    # Verify schoolbook
    if not verify(U_sb, V_sb, W_sb):
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # For [3,3,5], known minimum is 45, but try a few lower ranks
    min_rank_to_try = max(40, best_rank - 10)
    
    for target_rank in range(best_rank - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline - 15:
            break
        
        result = try_rank_reduction(target_rank, max_attempts=3)
        
        if result is not None:
            U_new, V_new, W_new = result
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = target_rank
            break  # Found a valid lower rank, stop searching
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
