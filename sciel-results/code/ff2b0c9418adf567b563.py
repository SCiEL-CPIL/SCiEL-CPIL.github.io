import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def _verify_fast(U, V, W, n, m, p, atol=1e-9):
    """
    Fast vectorized verification.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    """
    R = U.shape[1]
    # Construct the tensor T of shape (n*m, m*p, p*n)
    # T[u, v, w] = sum_k U[u, k] * V[v, k] * W[w, k]
    # We can compute this as:
    # T = np.einsum('uk,vk,wk->uvw', U, V, W)
    # This might be memory heavy if R is large, but R is small (<40).
    # n*m = 9, m*p = 12, p*n = 12.
    # 9*12*12 = 1296 entries. Very small.
    
    T = np.einsum('uk,vk,wk->uvw', U, V, W)
    
    # Check active entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                if not np.isclose(T[u, v, w], 1.0, atol=atol):
                    return False
    
    # Check inactive entries
    # Create a mask of valid entries
    valid_mask = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                valid_mask[u, v, w] = True
                
    # Inactive entries are where valid_mask is False
    if np.any(np.abs(T[~valid_mask]) > atol):
        return False
        
    return True

def _schoolbook(n, m, p):
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
    return U, V, W

def _search_rank(n, m, p, target_rank, deadline, n_attempts=500):
    """
    Search for a valid decomposition of the given rank.
    Uses random initialization with rounding to integers/half-integers.
    """
    nm, mp, pn = n * m, m * p, p * n
    values = np.array([-1, -0.5, 0, 0.5, 1])
    
    for _ in range(n_attempts):
        if time.time() > deadline - 2:
            break
            
        U = np.random.choice(values, size=(nm, target_rank))
        V = np.random.choice(values, size=(mp, target_rank))
        W = np.random.choice(values, size=(pn, target_rank))
        
        if _verify_fast(U, V, W, n, m, p):
            return U, V, W
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Schoolbook rank is 36
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_rank = 36
    best_solution = (U_school, V_school, W_school)
    
    # Try to find lower rank solutions
    # Known bounds: Rank(3x3x3)=23, Rank(3x3x4)=25?
    # We try from 25 up to 35.
    
    for r in range(25, 36):
        if time.time() > deadline - 10:
            break
        
        # Allocate more attempts for lower ranks as they are harder
        attempts = 100 if r < 30 else 50
        
        result = _search_rank(n, m, p, r, deadline, n_attempts=attempts)
        if result is not None:
            best_rank = r
            best_solution = result
            # If we find a very low rank, we can stop or continue to try even lower?
            # We are iterating from low to high, so if we find 25, we can return.
            # But maybe 24 exists? Unlikely.
            # We will return the first one found (lowest r).
            return {"U": best_solution[0].tolist(), "V": best_solution[1].tolist(), "W": best_solution[2].tolist()}
            
    # If no lower rank found, return schoolbook
    return {"U": best_solution[0].tolist(), "V": best_solution[1].tolist(), "W": best_solution[2].tolist()}
