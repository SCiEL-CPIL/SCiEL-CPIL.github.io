import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def _verify(U, V, W, n, m, p):
    rank = U.shape[1]
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    result = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    return np.allclose(result, 1.0, atol=1e-9)

def _error(U, V, W, n, m, p):
    rank = U.shape[1]
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    result = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    return np.sum((result - 1.0) ** 2)

def _discrete_search(rank, n, m, p, deadline, max_iters=5000, max_restarts=20):
    """
    Discrete local search for integer/half-integer solutions.
    Variables are stored in a flat array, scaled by 2 to be integers.
    """
    size_U = n * m * rank
    size_V = m * p * rank
    size_W = p * n * rank
    total_size = size_U + size_V + size_W
    
    best_solution = None
    best_err = float('inf')
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize with random integers in {-2, -1, 0, 1, 2} scaled by 2
        # So values are -4, -2, 0, 2, 4 -> /2 -> -2, -1, 0, 1, 2
        x = np.random.randint(-2, 3, size=total_size) * 2
        
        current_err = _error_from_flat(x, n, m, p, rank)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Pick a random index
            idx = np.random.randint(0, total_size)
            
            # Try +/- 1 (which corresponds to +/- 0.5 in actual value)
            old_val = x[idx]
            
            # Try +1
            x[idx] = old_val + 1
            new_err = _error_from_flat(x, n, m, p, rank)
            
            if new_err < current_err:
                current_err = new_err
                if current_err < 1e-9:
                    # Found solution
                    U, V, W = _unflatten(x, n, m, p, rank)
                    if _verify(U, V, W, n, m, p):
                        return U, V, W
            else:
                # Try -1
                x[idx] = old_val - 1
                new_err = _error_from_flat(x, n, m, p, rank)
                if new_err < current_err:
                    current_err = new_err
                    if current_err < 1e-9:
                        U, V, W = _unflatten(x, n, m, p, rank)
                        if _verify(U, V, W, n, m, p):
                            return U, V, W
                else:
                    # Revert
                    x[idx] = old_val
            
            # If no improvement in last few steps, might be stuck, but we don't track history here.
            # Simple hill climbing can get stuck. 
            # We can add a random jump if stuck.
            
            if current_err < best_err:
                best_err = current_err
                best_solution = x.copy()
                
    return None

def _error_from_flat(x, n, m, p, rank):
    size_U = n * m * rank
    size_V = m * p * rank
    U = x[:size_U].reshape(n, m, rank) / 2.0
    V = x[size_U:size_U+size_V].reshape(m, p, rank) / 2.0
    W = x[size_U+size_V:].reshape(p, n, rank) / 2.0
    Ur = U
    Vr = V
    Wr = W
    result = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    return np.sum((result - 1.0) ** 2)

def _unflatten(x, n, m, p, rank):
    size_U = n * m * rank
    size_V = m * p * rank
    U = x[:size_U].reshape(n, m, rank) / 2.0
    V = x[size_U:size_U+size_V].reshape(m, p, rank) / 2.0
    W = x[size_U+size_V:].reshape(p, n, rank) / 2.0
    return U, V, W

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Try rank 12
    result = _discrete_search(12, n, m, p, deadline)
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Try rank 13
    result = _discrete_search(13, n, m, p, deadline)
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback to schoolbook rank 24
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
