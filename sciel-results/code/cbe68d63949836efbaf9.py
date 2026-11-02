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

def _compute_error_vectorized(U, V, W, n=2, m=3, p=4):
    """Compute error using vectorized operations for speed."""
    rank = U.shape[1]
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j], V[j*p+k]) * W[k*n+i]
                # Actually need sum over c of U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                err += (s - 1.0)**2
    return err

def _try_random_decomposition(rank, n=2, m=3, p=4, max_iter=500, seed=None, use_schoolbook_init=False):
    """Try to find a rank-rank decomposition using random init and local search."""
    if seed is not None:
        random.seed(seed)
    np.random.seed(seed)
    
    values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    if use_schoolbook_init:
        # Start from schoolbook and randomly set some entries to 0
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        # Initialize first min(rank, n*m*p) columns like schoolbook
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= rank:
                        break
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        # Randomly perturb some entries
        for _ in range(rank * 2):
            mat = random.choice(['U', 'V', 'W'])
            row = random.randint(0, {'U': n*m, 'V': m*p, 'W': p*n}[mat] - 1)
            col = random.randint(0, rank - 1)
            if mat == 'U':
                U[row, col] = random.choice(values)
            elif mat == 'V':
                V[row, col] = random.choice(values)
            else:
                W[row, col] = random.choice(values)
    else:
        # Random initialization with half-integers
        U = np.array([[random.choice(values) for _ in range(rank)] for _ in range(n*m)])
        V = np.array([[random.choice(values) for _ in range(rank)] for _ in range(m*p)])
        W = np.array([[random.choice(values) for _ in range(rank)] for _ in range(p*n)])
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        # Compute error
        err = _compute_error_vectorized(U, V, W, n, m, p)
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

def _polish_decomposition(U, V, W, n=2, m=3, p=4, max_iter=200):
    """Try to polish a near-solution by snapping to half-integers and checking."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Try snapping to half-integers
    U_snapped = _snap_to_half_integers(U, tol=1e-4)
    V_snapped = _snap_to_half_integers(V, tol=1e-4)
    W_snapped = _snap_to_half_integers(W, tol=1e-4)
    
    if U_snapped is not None and V_snapped is not None and W_snapped is not None:
        if _validate(U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist(), n, m, p):
            return U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()
    
    # Try local search with tighter tolerance
    values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    best_err = _compute_error_vectorized(U, V, W, n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        # Try each possible change and keep the best
        best_change_err = best_err
        best_change = None
        for mat_name in ['U', 'V', 'W']:
            mat = {'U': U, 'V': V, 'W': W}[mat_name]
            rows = {'U': n*m, 'V': m*p, 'W': p*n}[mat_name]
            for row in range(rows):
                for col in range(rank):
                    old_val = mat[row, col]
                    for new_val in values:
                        if abs(new_val - old_val) < 1e-9:
                            continue
                        mat[row, col] = new_val
                        err = _compute_error_vectorized(U, V, W, n, m, p)
                        if err < best_change_err:
                            best_change_err = err
                            best_change = (mat_name, row, col, new_val)
                        mat[row, col] = old_val
        
        if best_change is None:
            break
        
        mat_name, row, col, new_val = best_change
        if mat_name == 'U':
            U[row, col] = new_val
        elif mat_name == 'V':
            V[row, col] = new_val
        else:
            W[row, col] = new_val
        
        best_err = best_change_err
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < 1e-12:
            break
    
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
        max_attempts = 150
        while time.time() < deadline and attempts < max_attempts:
            # Alternate between random init and schoolbook-based init
            use_schoolbook = (attempts % 3 == 0)
            result = _try_random_decomposition(rank, n, m, p, max_iter=80, seed=attempts * 1000 + rank, use_schoolbook_init=use_schoolbook)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                break  # Found valid decomposition for this rank, try lower
            attempts += 1
    
    # If we have a near-miss, try to polish it
    if time.time() < deadline - 10:
        for rank in range(best_rank, min(best_rank + 3, schoolbook_rank)):
            if time.time() > deadline - 5:
                break
            # Try to find a near-miss and polish
            for attempts in range(20):
                if time.time() > deadline - 5:
                    break
                result = _try_random_decomposition(rank, n, m, p, max_iter=50, seed=attempts * 2000 + rank + 10000)
                if result is not None:
                    U, V, W = result
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    break
    
    return best_result
# EVOLVE-BLOCK-END
