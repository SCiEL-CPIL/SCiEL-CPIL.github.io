"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def _validate(U, V, W, n=2, m=3, p=4):
    """Check if the decomposition exactly reproduces matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Check shapes
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != rank or V.shape[1] != rank or W.shape[1] != rank:
        return False
    # Check entries are integer or half-integer
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    # Check tensor decomposition: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # T[i,j,k] should be 1 for all valid i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _snap_to_half_integers(mat, tol=1e-6):
    """Snap values to nearest half-integer if close enough."""
    mat = np.array(mat, dtype=float)
    snapped = np.round(mat * 2) / 2.0
    if np.max(np.abs(mat - snapped)) < tol:
        return snapped
    return None

def _try_random_decomposition(rank, n=2, m=3, p=4, max_iter=500, seed=None):
    """Try to find a rank-rank decomposition using random init and local search."""
    if seed is not None:
        random.seed(seed)
    np.random.seed(seed)
    
    # Random initialization with half-integers
    values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    U = np.array([[random.choice(values) for _ in range(rank)] for _ in range(n*m)])
    V = np.array([[random.choice(values) for _ in range(rank)] for _ in range(m*p)])
    W = np.array([[random.choice(values) for _ in range(rank)] for _ in range(p*n)])
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err += (s - 1.0)**2
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if best_err < 1e-12:
                break
        
        # Random perturbation: pick a random element and change it
        mat = random.choice(['U', 'V', 'W'])
        row = random.randint(0, {'U': n*m, 'V': m*p, 'W': p*n}[mat] - 1)
        col = random.randint(0, rank - 1)
        if mat == 'U':
            U[row, col] = random.choice(values)
        elif mat == 'V':
            V[row, col] = random.choice(values)
        else:
            W[row, col] = random.choice(values)
    
    # Check if best solution is valid
    if _validate(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p):
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # First, return schoolbook as fallback
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    for rank in range(16, schoolbook_rank):  # Try ranks from 16 up to 23
        if time.time() > deadline:
            break
        
        # Multiple random attempts for each rank
        attempts = 0
        max_attempts = 200
        while time.time() < deadline and attempts < max_attempts:
            result = _try_random_decomposition(rank, n, m, p, max_iter=100, seed=attempts * 1000 + rank)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                break  # Found valid decomposition for this rank, try lower
            attempts += 1
    
    return best_result
# EVOLVE-BLOCK-END
