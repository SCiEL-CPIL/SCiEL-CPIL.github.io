import numpy as np
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    return {"U": U_school, "V": V_school, "W": W_school}

def try_find_decomposition(r, max_trials=500, seed=42):
    n, m, p = DIMENSIONS
    num_eq = n * m * p
    num_var = p * n * r
    
    # Indices
    # U: i*m+j
    # V: j*p+k
    # W: k*n+i
    
    # We want to solve A W' = 1 for integer W'
    # A is num_eq x num_var
    # Row idx(i,j,k), Col idx(k,i,c) -> A[row, col] = U[i*m+j, c] * V[j*p+k, c]
    
    rng = np.random.default_rng(seed)
    
    for _ in range(max_trials):
        # Generate random U, V with small integers
        # Using -1, 0, 1 to keep numbers small
        U_rand = rng.integers(-1, 2, size=(n * m, r))
        V_rand = rng.integers(-1, 2, size=(m * p, r))
        
        # Build A
        A = np.zeros((num_eq, num_var), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = i * m * p + j * p + k
                    for c in range(r):
                        col = k * n * r + i * r + c
                        A[row, col] = U_rand[i * m + j, c] * V_rand[j * p + k, c]
        
        # Solve A W' = 1
        # Use least squares or check consistency
        # Since it's integer, we can use np.linalg.lstsq and check residual
        # Or better, check if 1 is in the column space of A over Q
        # For small systems, we can use exact rational arithmetic? No, too slow.
        # Use float and check if solution is close to integers.
        
        try:
            W_prime, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(num_eq), rcond=None)
            # Check if residual is small
            if residuals.size > 0 and residuals[0] < 1e-6:
                # Check if W_prime is close to integers
                if np.allclose(W_prime, np.round(W_prime), atol=1e-6):
                    W_int = np.round(W_prime).astype(int)
                    # Verify exactly
                    if np.allclose(A @ W_int, np.ones(num_eq)):
                        # We found a solution!
                        # W = W_int / 2
                        # But we need to return U, V, W as lists of lists
                        # U and V are integers, W is half-integers
                        # The task allows half-integers.
                        
                        # Convert to lists
                        U_list = U_rand.tolist()
                        V_list = V_rand.tolist()
                        W_list = (W_int / 2.0).tolist()
                        
                        return {"U": U_list, "V": V_list, "W": W_list}
        except Exception:
            continue
            
    return None

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    # Try ranks from 22 upwards
    for r in range(22, 37):
        if time.time() > deadline:
            break
        result = try_find_decomposition(r, max_trials=200, seed=r*100)
        if result is not None:
            return result
        # If time is running out, reduce trials
        if time.time() > 90.0:
            result = try_find_decomposition(r, max_trials=50, seed=r*100+1)
            if result is not None:
                return result
            
    # Fallback to schoolbook
    return construct_schoolbook()

def construct_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
