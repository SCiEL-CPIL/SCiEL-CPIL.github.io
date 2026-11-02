import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def _check_decomposition(U, V, W, n, m, p):
    """Check if U, V, W form an exact decomposition of J_{n,m,p}."""
    rank = len(U[0])
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Compute sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # This should be 1 for all i, j, k
    
    # Vectorize for speed
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # Reshape to tensors
    # U: (n, m, rank)
    U_t = U_arr.reshape(n, m, rank)
    # V: (m, p, rank)
    V_t = V_arr.reshape(m, p, rank)
    # W: (p, n, rank)
    W_t = W_arr.reshape(p, n, rank)
    
    # Sum over rank
    # Result: (n, m, p)
    # T[i, j, k] = sum_r U_t[i, j, r] * V_t[j, k, r] * W_t[k, i, r]
    
    # We can do this by iterating over r or using einsum
    # einsum is efficient
    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    
    # Check if T is all ones
    if not np.all(T == 1):
        return False
    return True

def _search_rank_233(max_rank=15, time_limit=10):
    """Search for a rank decomposition of J_{2,3,3}."""
    n, m, p = 2, 3, 3
    start_time = time.time()
    
    for r in range(12, max_rank + 1):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random initializations
        for _ in range(50):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize with random -1, 1
            U = [[1 if np.random.rand() > 0.5 else -1 for _ in range(r)] for _ in range(n*m)]
            V = [[1 if np.random.rand() > 0.5 else -1 for _ in range(r)] for _ in range(m*p)]
            W = [[1 if np.random.rand() > 0.5 else -1 for _ in range(r)] for _ in range(p*n)]
            
            if _check_decomposition(U, V, W, n, m, p):
                return U, V, W, r
                
    return None

def _construct_239_from_233(U233, V233, W233, R):
    """Construct J_{2,3,9} from J_{2,3,3} using Kronecker product with J_{1,1,3}."""
    n, m, p = 2, 3, 9
    rank = 3 * R
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U239[i*m+j, t*3+t'] = U233[i*m+j, t]
    for i in range(2):
        for j in range(3):
            idx_233 = i * 3 + j
            for t in range(R):
                val = U233[idx_233][t]
                for t_prime in range(3):
                    idx_239 = idx_233
                    col = t * 3 + t_prime
                    U[idx_239][col] = val
                    
    # V239[j*9 + 3*k1 + k2, t*3+t'] = V233[j*3 + k1, t]
    for j in range(3):
        for k1 in range(3):
            idx_233 = j * 3 + k1
            for t in range(R):
                val = V233[idx_233][t]
                for k2 in range(3):
                    idx_239 = j * 9 + 3 * k1 + k2
                    for t_prime in range(3):
                        col = t * 3 + t_prime
                        V[idx_239][col] = val
                        
    # W239[(3*k1+k2)*2 + i, t*3+t'] = W233[k1*2 + i, t]
    for k1 in range(3):
        for i in range(2):
            idx_233 = k1 * 2 + i
            for t in range(R):
                val = W233[idx_233][t]
                for k2 in range(3):
                    idx_239 = (3 * k1 + k2) * 2 + i
                    for t_prime in range(3):
                        col = t * 3 + t_prime
                        W[idx_239][col] = val
                        
    return {"U": U, "V": V, "W": W}

def construct():
    # Schoolbook decomposition
    n, m, p = DIMENSIONS
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
    
    schoolbook = {"U": U, "V": V, "W": W}
    
    # Try to find a better decomposition
    try:
        result_233 = _search_rank_233(max_rank=14, time_limit=10)
        if result_233:
            U233, V233, W233, R = result_233
            result = _construct_239_from_233(U233, V233, W233, R)
            # Verify the result
            if _check_decomposition(result["U"], result["V"], result["W"], n, m, p):
                return result
    except Exception:
        pass
        
    return schoolbook
