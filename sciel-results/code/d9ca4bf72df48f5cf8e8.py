import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Fallback rank
    fallback_rank = n * m * p
    
    start_total = time.time()
    deadline = start_total + 140  # 10s safety margin
    
    # Construct the target tensor T
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomposition(U, V, W, tol=1e-9):
        """Check if the decomposition is exact (within tolerance)."""
        E = np.zeros((n * m, m * p, p * n))
        for r in range(U.shape[1]):
            E += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.linalg.norm(E - T) < tol

    def polish_decomposition(U, V, W, max_iter=1000, tol=1e-12):
        """Polish a near-exact decomposition using local optimization."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        rank = U.shape[1]
        
        for it in range(max_iter):
            # Compute error
            E = T.copy()
            for r in range(rank):
                E -= np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
            err = np.linalg.norm(E)
            if err < tol:
                break
            
            # Compute gradients
            Temp1 = np.einsum('abc,br->acr', E, V)
            G_U = np.einsum('acr,cr->ar', Temp1, W)
            
            Temp2 = np.einsum('abc,ar->bcr', E, U)
            G_V = np.einsum('bcr,cr->br', Temp2, W)
            
            G_W = np.einsum('acr,ar->cr', Temp1, U)
            
            # Adaptive learning rate
            grad_norm = np.sqrt(np.sum(G_U**2) + np.sum(G_V**2) + np.sum(G_W**2))
            if grad_norm < 1e-15:
                break
            
            lr = 0.1 * err / (grad_norm + 1e-15)
            lr = min(lr, 0.5)
            
            # Update
            U -= lr * G_U
            V -= lr * G_V
            W -= lr * G_W
        
        return U, V, W

    def try_find_decomposition(rank, time_limit=15, seed=42):
        """Try to find a decomposition using gradient descent with polishing."""
        start_time = time.time()
        np.random.seed(seed)
        
        # Initialize U, V, W
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Scale to help convergence
        U /= np.sqrt(rank)
        V /= np.sqrt(rank)
        W /= np.sqrt(rank)
        
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Gradient descent phase
        lr = 0.01
        for step in range(2000):
            if time.time() - start_time > time_limit * 0.7:
                break
            
            E = T.copy()
            for r in range(rank):
                E -= np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
            err = np.linalg.norm(E)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-8:
                break
            
            # Compute gradients
            Temp1 = np.einsum('abc,br->acr', E, V)
            G_U = np.einsum('acr,cr->ar', Temp1, W)
            
            Temp2 = np.einsum('abc,ar->bcr', E, U)
            G_V = np.einsum('bcr,cr->br', Temp2, W)
            
            G_W = np.einsum('acr,ar->cr', Temp1, U)
            
            # Update
            U -= lr * G_U
            V -= lr * G_V
            W -= lr * G_W
        
        # Polish phase
        if time.time() - start_time < time_limit:
            U_polished, V_polished, W_polished = polish_decomposition(best_U, best_V, best_W)
            err_polished = np.linalg.norm(T - sum(np.einsum('a,b,c->abc', U_polished[:, r], V_polished[:, r], W_polished[:, r]) for r in range(rank)))
            if err_polished < best_err:
                best_U, best_V, best_W = U_polished, V_polished, W_polished
                best_err = err_polished
        
        # Round to half-integers and check
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        
        if verify_decomposition(U_rounded, V_rounded, W_rounded):
            return U_rounded, V_rounded, W_rounded
        
        # Try rounding to integers
        U_int = np.round(best_U)
        V_int = np.round(best_V)
        W_int = np.round(best_W)
        
        if verify_decomposition(U_int, V_int, W_int):
            return U_int, V_int, W_int
        
        return None

    # Try to find a lower rank decomposition
    result = None
    
    # Try rank 23 first
    if time.time() < deadline:
        result = try_find_decomposition(23, time_limit=15, seed=42)
    
    # Try rank 26 if rank 23 failed
    if not result and time.time() < deadline:
        result = try_find_decomposition(26, time_limit=15, seed=123)
    
    # Try rank 27 with different seed
    if not result and time.time() < deadline:
        result = try_find_decomposition(27, time_limit=15, seed=777)
    
    # Try rank 28 with different seed
    if not result and time.time() < deadline:
        result = try_find_decomposition(28, time_limit=15, seed=999)
    
    if result:
        U, V, W = result
    else:
        # Fallback to schoolbook
        rank = fallback_rank
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
    
    # Convert to lists as required by interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
