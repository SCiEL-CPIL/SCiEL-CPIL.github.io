import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    n, m, p = DIMENSIONS
    # Index mappings
    # U: (n*m, rank) -> index i*m + j
    # V: (m*p, rank) -> index j*p + k
    # W: (p*n, rank) -> index k*n + i  (as per prompt/seed)
    
    # Target Tensor T (dense)
    # Shape: (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1.0
                
    def tensor_error(params, rank):
        # params is a flat vector of size rank * (n*m + m*p + p*n)
        # Split into U, V, W
        size_u = n * m
        size_v = m * p
        size_w = p * n
        
        U = params[:rank*size_u].reshape(rank, size_u).T
        V = params[rank*size_u:rank*(size_u+size_v)].reshape(rank, size_v).T
        W = params[rank*(size_u+size_v):].reshape(rank, size_w).T
        
        # Compute tensor sum: sum_r U[:,r] (x) V[:,r] (x) W[:,r]
        # Efficient computation using einsum
        # T_comp[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # This is equivalent to np.einsum('ar,br,cr->abc', U, V, W)
        T_comp = np.einsum('ar,br,cr->abc', U, V, W)
        
        return np.sum((T_comp - T)**2)

    def check_and_round(U, V, W, rank):
        # Check if U, V, W entries are close to half-integers
        # And if the decomposition is exact
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Verify exactness with rounded values
        T_comp = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.allclose(T_comp, T, atol=1e-9):
            # Ensure exact half-integer representation
            # Convert to list of lists as required by interface?
            # The interface expects {'U': ..., 'V': ..., 'W': ...}
            # Usually lists or arrays. The seed used lists.
            return U_r.tolist(), V_r.tolist(), W_r.tolist()
        return None

    # First, return schoolbook if time is short or as fallback
    def get_schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return U, V, W

    schoolbook_U, schoolbook_V, schoolbook_W = get_schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = schoolbook_U, schoolbook_V, schoolbook_W

    # Try to find lower rank
    # We'll try ranks 41, 40, ...
    # Use random restarts for each rank
    
    num_restarts = 5
    
    for rank in range(best_rank - 1, 25, -1):
        if time.time() > deadline - 10:
            break
            
        found = False
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize parameters
            # Small random values
            np.random.seed(restart)
            size_u = n * m
            size_v = m * p
            size_w = p * n
            total_size = rank * (size_u + size_v + size_w)
            
            # Initialize with small noise around 0
            params = np.random.normal(0, 0.1, total_size)
            
            # Try to optimize
            try:
                res = minimize(tensor_error, params, args=(rank,), method='L-BFGS-B', 
                               options={'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-6})
                
                if res.fun < 1e-6:
                    # Extract U, V, W
                    U = res.x[:rank*size_u].reshape(rank, size_u).T
                    V = res.x[rank*size_u:rank*(size_u+size_v)].reshape(rank, size_v).T
                    W = res.x[rank*(size_u+size_v):].reshape(rank, size_w).T
                    
                    result = check_and_round(U, V, W, rank)
                    if result:
                        U_r, V_r, W_r = result
                        best_rank = rank
                        best_U, best_V, best_W = U_r, V_r, W_r
                        found = True
                        break
            except Exception as e:
                pass
        
        if found:
            # If we found a lower rank, we can try to go even lower
            # But we break the inner loop, so we continue to next rank
            pass

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
