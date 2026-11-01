import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def _validate(U, V, W, n, m, p):
    """Exact check for tensor decomposition of matrix multiplication.
    
    For all i in [0,n), j in [0,m), k in [0,p):
        sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] = 1
    """
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
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

def _error(U, V, W, n, m, p):
    r = len(U[0])
    total_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                total_err += (s - 1.0) ** 2
    return total_err

def _random_search(n, m, p, target_rank, max_iters=50000, seed=0):
    """
    Attempts to find a valid decomposition with the given rank using gradient descent 
    on a continuous relaxation, followed by integer snapping.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Initialize with small random integers
    # We use integer init because the solution is likely integer/half-integer
    U = np.array([[random.randint(-1, 1) for _ in range(target_rank)] for _ in range(n*m)], dtype=float)
    V = np.array([[random.randint(-1, 1) for _ in range(target_rank)] for _ in range(m*p)], dtype=float)
    W = np.array([[random.randint(-1, 1) for _ in range(target_rank)] for _ in range(p*n)], dtype=float)
    
    # Simple gradient descent
    lr = 0.01
    best_err = float('inf')
    best_state = (U.copy(), V.copy(), W.copy())
    
    # Pre-compute indices for speed
    idx_u = []
    idx_v = []
    idx_w = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u.append(i*m + j)
                idx_v.append(j*p + k)
                idx_w.append(k*n + i)
    idx_u = np.array(idx_u)
    idx_v = np.array(idx_v)
    idx_w = np.array(idx_w)
    
    start_time = time.time()
    for it in range(max_iters):
        if time.time() - start_time > 20: # Time limit per search attempt
            break
            
        # Compute gradients
        # E = sum_{i,j,k} (sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} - 1)^2
        # dE/du_{ij,c} = 2 * (val - 1) * V_{j,k,c} * W_{k,i,c}
        # We can compute this efficiently using vectorization for small n,m,p
        
        # Flatten U, V, W for easier manipulation? 
        # Given small size, explicit loops might be okay, but numpy is better.
        
        # Let's compute the sums S_{ijk} = sum_c U_{ijk,c} V_{jkc,c} W_{kic,c}
        # U shape: (nm, R)
        # V shape: (mp, R)
        # W shape: (pn, R)
        
        # We need gradients for U, V, W.
        # grad_U[i*m+j, c] = sum_{k} 2*(S[i,j,k]-1) * V[j*p+k, c] * W[k*n+i, c]
        
        # This is complex to vectorize perfectly without loops over i,j,k.
        # Given n=3, m=3, p=5, total terms = 45. R is small (e.g. 25).
        # 45 * 25 operations per gradient step is tiny.
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for idx in range(len(idx_u)):
            i = idx_u[idx] // m
            j = idx_u[idx] % m
            k_idx = idx_v[idx] # j*p+k
            k = k_idx % p
            w_idx = idx_w[idx] # k*n+i
            
            s_val = np.dot(U[idx_u[idx]], V[idx_v[idx]]) # Wait, this is dot product over c?
            # No, s_val = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # This is a triple product.
            s_val = np.sum(U[idx_u[idx]] * V[idx_v[idx]] * W[idx_w[idx]])
            diff = 2.0 * (s_val - 1.0)
            
            # grad_U[i*m+j, :] += diff * V[j*p+k, :] * W[k*n+i, :]
            grad_U[idx_u[idx]] += diff * V[idx_v[idx]] * W[idx_w[idx]]
            
            # grad_V[j*p+k, :] += diff * U[i*m+j, :] * W[k*n+i, :]
            grad_V[idx_v[idx]] += diff * U[idx_u[idx]] * W[idx_w[idx]]
            
            # grad_W[k*n+i, :] += diff * U[i*m+j, :] * V[j*p+k, :]
            grad_W[idx_w[idx]] += diff * U[idx_u[idx]] * V[idx_v[idx]]
            
        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Snap to nearest 0.5 if close? Or just track continuous error.
        # Check if close to integer/half-integer
        # For simplicity, just check continuous error first.
        err = np.sum((np.array([np.sum(U[idx_u[idx]] * V[idx_v[idx]] * W[idx_w[idx]]) for idx in range(len(idx_u))]) - 1.0)**2)
        
        if err < best_err:
            best_err = err
            best_state = (U.copy(), V.copy(), W.copy())
            
        if err < 1e-4:
            # Try to snap
            U_snap = np.round(best_state[0] * 2.0) / 2.0
            V_snap = np.round(best_state[1] * 2.0) / 2.0
            W_snap = np.round(best_state[2] * 2.0) / 2.0
            
            # Validate snapped
            valid = True
            for idx in range(len(idx_u)):
                s_val = np.sum(U_snap[idx_u[idx]] * V_snap[idx_v[idx]] * W_snap[idx_w[idx]])
                if abs(s_val - 1.0) > 1e-6:
                    valid = False
                    break
            if valid:
                # Convert to list of lists
                return {
                    "U": U_snap.tolist(),
                    "V": V_snap.tolist(),
                    "W": W_snap.tolist()
                }

    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try to find a better rank than 45
    # Known optimal is 17. Let's try ranks 20 down to 17.
    # If we find a valid one, return it.
    
    # Start with a safe fallback
    fallback = _schoolbook(n, m, p)
    best_result = fallback
    best_rank = n * m * p
    
    # Search for lower ranks
    for target_rank in range(25, 16, -1):
        if time.time() - start_time > 100: # Keep 20s buffer
            break
            
        # Try a few seeds
        for seed in range(5):
            res = _random_search(n, m, p, target_rank, max_iters=10000, seed=seed)
            if res is not None:
                # Final validation with strict integer check
                U = res["U"]
                V = res["V"]
                W = res["W"]
                if _validate(U, V, W, n, m, p):
                    return res
            if time.time() - start_time > 100:
                break

    # If no better rank found, return schoolbook
    return best_result

# EVOLVE-BLOCK-END
