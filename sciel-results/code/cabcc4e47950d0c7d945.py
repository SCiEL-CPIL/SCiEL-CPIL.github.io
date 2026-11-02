import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Target rank to search for
    target_rank = 23
    # Fallback rank
    fallback_rank = n * m * p
    
    start_total = time.time()
    
    # Construct the target tensor T
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def try_find_decomposition(rank, time_limit=10, seed=42):
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
        
        lr = 0.01
        # Simple gradient descent
        for step in range(1000):
            if time.time() - start_time > time_limit:
                break
            
            E = T.copy()
            for r in range(rank):
                E -= np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
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
            
            # Check convergence
            err = np.linalg.norm(E)
            if err < 1e-9:
                break
        
        # Round to half-integers
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Check exactness
        E_rounded = T.copy()
        for r in range(rank):
            E_rounded -= np.einsum('a,b,c->abc', U_rounded[:, r], V_rounded[:, r], W_rounded[:, r])
        
        if np.linalg.norm(E_rounded) < 1e-9:
            return U_rounded, V_rounded, W_rounded
        else:
            return None

    # Try to find a lower rank decomposition
    result = None
    if time.time() - start_total < 100:
        result = try_find_decomposition(target_rank, time_limit=10, seed=42)
    
    if not result and time.time() - start_total < 100:
        # Try rank 26 with a different seed
        result = try_find_decomposition(26, time_limit=10, seed=123)
        
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
