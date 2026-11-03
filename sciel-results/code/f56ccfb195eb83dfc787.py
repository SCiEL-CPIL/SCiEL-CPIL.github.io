# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape for efficient contraction
    # U: (n*m, rank) -> (n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    # W: (p*n, rank) -> (p, n, rank)
    U_t = U_arr.reshape(n, m, rank)
    V_t = V_arr.reshape(m, p, rank)
    W_t = W_arr.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # We can do this by iterating over i, j, k or using einsum.
    # np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    try:
        T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    except Exception:
        return False
        
    # The target tensor is 1 for all i,j,k
    target = np.ones((n, m, p))
    if not np.allclose(T, target, atol=tol):
        return False
    return True

def _schoolbook(n, m, p):
    """Return the standard schoolbook decomposition of rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}

def _gradient_search(n, m, p, time_limit=120):
    """Search for low-rank decomposition using gradient descent."""
    start_time = time.time()
    best_result = None
    best_rank = n * m * p
    
    # Try ranks from schoolbook down to a reasonable lower bound
    # Lower bound for 3x3x3 is 19 (Strassen is 23, but 19 is unknown/conjectured? Actually 19 is not proven for 3x3x3, best known is 23. 
    # Wait, 3x3x3 rank is 23 (Strassen). 
    # For 3x3x6, we can think of it as two 3x3x3s. 
    # If we share 0, rank is 46. 
    # If we share some, maybe less.
    # Let's try ranks from 53 down to 40.
    
    for r in range(53, 39, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize with small random values
        # Using a fixed seed for reproducibility in the search, but varying by rank
        np.random.seed(42 + r)
        
        # Initialize U, V, W
        # Shape (n*m, r), (m*p, r), (p*n, r)
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        # Reshape for einsum
        # U: (n, m, r)
        # V: (m, p, r)
        # W: (p, n, r)
        
        lr = 0.01
        # Number of steps per rank attempt
        # We need to be fast.
        steps = 200
        
        for step in range(steps):
            if time.time() - start_time > time_limit - 5:
                break
                
            U_t = U.reshape(n, m, r)
            V_t = V.reshape(m, p, r)
            W_t = W.reshape(p, n, r)
            
            # Compute current tensor T
            T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
            target = np.ones((n, m, p))
            
            # Check convergence
            error = np.sum((T - target) ** 2)
            if error < 1e-10:
                # Found valid decomposition
                if r < best_rank:
                    best_rank = r
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                # Try to find even lower rank? 
                # The loop continues to lower r, so we just break inner loop
                break
            
            # Compute gradients
            # dL/dU[i,j,c] = 2 * (T[i,j,k] - 1) * V[j,k,c] * W[k,i,c]
            # We need to sum over k.
            # grad_U[i,j,c] = 2 * sum_k (T[i,j,k] - 1) * V[j,k,c] * W[k,i,c]
            
            # Vectorized gradient calculation
            # Reshape target to match T
            # diff = T - target (n, m, p)
            diff = T - target
            
            # grad_U[i,j,c] = 2 * einsum('ijk,jkc,kic->ijc', diff, V_t, W_t)
            grad_U = 2 * np.einsum('ijk,jkc,kic->ijc', diff, V_t, W_t)
            
            # grad_V[j,k,c] = 2 * sum_i (T[i,j,k] - 1) * U[i,j,c] * W[k,i,c]
            # grad_V[j,k,c] = 2 * einsum('ijk,ijc,kic->jkc', diff, U_t, W_t)
            grad_V = 2 * np.einsum('ijk,ijc,kic->jkc', diff, U_t, W_t)
            
            # grad_W[k,i,c] = 2 * sum_j (T[i,j,k] - 1) * U[i,j,c] * V[j,k,c]
            # grad_W[k,i,c] = 2 * einsum('ijk,ijc,jkc->kic', diff, U_t, V_t)
            grad_W = 2 * np.einsum('ijk,ijc,jkc->kic', diff, U_t, V_t)
            
            # Update
            U -= lr * grad_U.reshape(n * m, r)
            V -= lr * grad_V.reshape(m * p, r)
            W -= lr * grad_W.reshape(p * n, r)
            
            # Adaptive learning rate?
            # If error increases, reduce lr.
            # Simple: keep lr constant for now, maybe decay.
            lr *= 0.995
    
    return best_result

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Deadline guard: 150s wall-clock, use 140s as safety margin
    deadline = time.time() + 140
    
    # Try gradient search
    # We allocate most of the time to this
    search_time_limit = 130
    grad_result = _gradient_search(n, m, p, time_limit=search_time_limit)
    
    if grad_result is not None:
        # Verify
        if _verify_decomposition(grad_result["U"], grad_result["V"], grad_result["W"], n, m, p):
            result = grad_result
            best_rank = len(grad_result["U"][0])
    
    return result
# EVOLVE-BLOCK-END
