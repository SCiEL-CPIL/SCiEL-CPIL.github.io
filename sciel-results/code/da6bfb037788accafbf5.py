import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _validate(U, V, W):
    """
    Validate that U, V, W form an exact decomposition of the matmul(3,5,6) tensor.
    Returns True if valid, False otherwise.
    """
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    r = len(U[0])
    
    # Check dimensions
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if not all(len(row) == r for row in U):
        return False
    if not all(len(row) == r for row in V):
        return False
    if not all(len(row) == r for row in W):
        return False
    if r == 0:
        return False
    
    # Convert to numpy for efficient computation
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # For each (i,j,k), check if sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i*m + j]
                v_val = V_np[j*p + k]
                w_val = W_np[k*n + i]
                # Element-wise product and sum
                val = np.sum(u_val * v_val * w_val)
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _validate_exact(U, V, W):
    """
    Exact validation using Fraction arithmetic.
    """
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    r = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if r == 0:
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(r):
                    total += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if total != Fraction(1):
                    return False
    return True

def _solve_linear_system_exact(A, b):
    """
    Solve Ax = b exactly using Gaussian elimination with Fractions.
    A is a list of lists of Fractions, b is a list of Fractions.
    Returns x as a list of Fractions, or None if no solution.
    """
    n_rows = len(A)
    n_cols = len(A[0]) if n_rows > 0 else 0
    
    if n_rows == 0:
        return [Fraction(0)] * n_cols
    
    # Augmented matrix
    aug = [row[:] + [b[i]] for i, row in enumerate(A)]
    
    # Forward elimination with partial pivoting
    rank = 0
    pivot_cols = []
    
    for col in range(n_cols):
        # Find pivot
        pivot = None
        for row in range(rank, n_rows):
            if aug[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            continue
        
        # Swap rows
        if pivot != rank:
            aug[rank], aug[pivot] = aug[pivot], aug[rank]
        
        # Normalize pivot row
        pivot_val = aug[rank][col]
        for j in range(col, n_cols + 1):
            aug[rank][j] /= pivot_val
        
        # Eliminate other rows
        for row in range(n_rows):
            if row == rank:
                continue
            factor = aug[row][col]
            if factor != 0:
                for j in range(col, n_cols + 1):
                    aug[row][j] -= factor * aug[rank][j]
        
        rank += 1
        pivot_cols.append(col)
    
    # Check consistency
    for row in range(rank, n_rows):
        if any(aug[row][j] != 0 for j in range(n_cols)):
            return None
    
    # Extract solution (set free variables to 0)
    x = [Fraction(0)] * n_cols
    for i, col in enumerate(pivot_cols):
        x[col] = aug[i][n_cols]
    
    return x

def _compute_decomposition_error(U, V, W, n, m, p):
    """Compute the decomposition error using vectorized operations."""
    errors = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m + j], V[j*p + k] * W[k*n + i])
                errors.append((val - 1.0) ** 2)
    return np.sqrt(np.sum(errors))

def _alternating_min_exact(U, V, W, n, m, p, max_iters=20, deadline=None):
    """
    Alternating minimization using exact rational arithmetic.
    U, V, W are lists of lists of Fractions.
    """
    rank = len(U[0])
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline - 10:
            break
        
        # Update U: for each (i,j), solve for u such that sum_k u_k * v_k * w_k = 1
        for i in range(n):
            for j in range(m):
                # Build system: for each k, coefficient = V[j*p+k][c] * W[k*n+i][c]
                A = []
                b = []
                for k in range(p):
                    row = []
                    for c in range(rank):
                        row.append(V[j*p+k][c] * W[k*n+i][c])
                    A.append(row)
                    b.append(Fraction(1))
                
                u_new = _solve_linear_system_exact(A, b)
                if u_new is not None:
                    U[i*m+j] = u_new
        
        # Update V: for each (j,k), solve for v such that sum_i u_i * v_k * w_i = 1
        for j in range(m):
            for k in range(p):
                A = []
                b = []
                for i in range(n):
                    row = []
                    for c in range(rank):
                        row.append(U[i*m+j][c] * W[k*n+i][c])
                    A.append(row)
                    b.append(Fraction(1))
                
                v_new = _solve_linear_system_exact(A, b)
                if v_new is not None:
                    V[j*p+k] = v_new
        
        # Update W: for each (k,i), solve for w such that sum_j u_j * v_j * w_k = 1
        for k in range(p):
            for i in range(n):
                A = []
                b = []
                for j in range(m):
                    row = []
                    for c in range(rank):
                        row.append(U[i*m+j][c] * V[j*p+k][c])
                    A.append(row)
                    b.append(Fraction(1))
                
                w_new = _solve_linear_system_exact(A, b)
                if w_new is not None:
                    W[k*n+i] = w_new
    
    # Check if we have an exact solution
    if _validate_exact(U, V, W):
        return U, V, W, True
    return U, V, W, False

def _structured_init(rank, n, m, p, seed=0):
    """
    Create a structured initialization for the decomposition.
    """
    rng = random.Random(seed)
    
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    
    # Assign each column to a specific (i,j,k) pattern with some randomness
    for c in range(rank):
        i_idx = c % n
        j_idx = (c // n) % m
        k_idx = (c // (n * m)) % p
        
        # Add some variation
        if rng.random() < 0.3:
            i_idx = (i_idx + rng.randint(0, n-1)) % n
        if rng.random() < 0.3:
            j_idx = (j_idx + rng.randint(0, m-1)) % m
        if rng.random() < 0.3:
            k_idx = (k_idx + rng.randint(0, p-1)) % p
        
        U[i_idx * m + j_idx][c] = Fraction(1)
        V[j_idx * p + k_idx][c] = Fraction(1)
        W[k_idx * n + i_idx][c] = Fraction(1)
        
        # Add small perturbations
        if rng.random() < 0.5:
            U[i_idx * m + j_idx][c] += Fraction(rng.randint(-1, 1), 2)
        if rng.random() < 0.5:
            V[j_idx * p + k_idx][c] += Fraction(rng.randint(-1, 1), 2)
        if rng.random() < 0.5:
            W[k_idx * n + i_idx][c] += Fraction(rng.randint(-1, 1), 2)
    
    return U, V, W

def _try_rank_exact(rank, n, m, p, deadline, max_attempts=8):
    """
    Try to find a valid decomposition using exact rational arithmetic.
    """
    for attempt in range(max_attempts):
        if time.time() > deadline - 15:
            break
        
        # Structured initialization
        U, V, W = _structured_init(rank, n, m, p, seed=attempt * 1009 + rank * 7)
        
        # Run exact alternating minimization
        U2, V2, W2, success = _alternating_min_exact(U, V, W, n, m, p, max_iters=25, deadline=deadline)
        
        if success:
            return U2, V2, W2
    
    return None

def _try_rank_float(rank, n, m, p, deadline, max_attempts=6):
    """
    Try with floating-point alternating minimization, then check if near half-integers.
    """
    for attempt in range(max_attempts):
        if time.time() > deadline - 15:
            break
        
        np.random.seed(attempt * 7919 + rank * 104729)
        
        # Structured init
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for c in range(rank):
            i_idx = c % n
            j_idx = (c // n) % m
            k_idx = (c // (n * m)) % p
            U[i_idx * m + j_idx, c] = 1.0
            V[j_idx * p + k_idx, c] = 1.0
            W[k_idx * n + i_idx, c] = 1.0
        
        # Add small random noise
        U += np.random.randn(n * m, rank) * 0.1
        V += np.random.randn(m * p, rank) * 0.1
        W += np.random.randn(p * n, rank) * 0.1
        
        # Alternating minimization
        for iteration in range(40):
            if time.time() > deadline - 15:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, rank))
                    for k in range(p):
                        M[k, :] = V[j*p + k] * W[k*n + i]
                    try:
                        u_new = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                        U[i*m + j] = u_new
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, rank))
                    for i in range(n):
                        M[i, :] = U[i*m + j] * W[k*n + i]
                    try:
                        v_new = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                        V[j*p + k] = v_new
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, rank))
                    for j in range(m):
                        M[j, :] = U[i*m + j] * V[j*p + k]
                    try:
                        w_new = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                        W[k*n + i] = w_new
                    except:
                        pass
        
        # Check error
        err = _compute_decomposition_error(U, V, W, n, m, p)
        if err < 1e-8:
            # Round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            U_list = U_r.tolist()
            V_list = V_r.tolist()
            W_list = W_r.tolist()
            
            if _validate(U_list, V_list, W_list):
                return U_list, V_list, W_list
    
    return None

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing lists of lists.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Start with schoolbook as fallback
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j][col] = 1
                V_sb[j*p + k][col] = 1
                W_sb[k*n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = [row[:] for row in U_sb], [row[:] for row in V_sb], [row[:] for row in W_sb]
    best_rank = schoolbook_rank
    
    # Known lower bound: max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    lower_bound = max(n * m, m * p, p * n)
    
    # Try ranks from lower_bound upward
    ranks_to_try = list(range(lower_bound, min(lower_bound + 20, schoolbook_rank)))
    
    for rank in ranks_to_try:
        if time.time() > deadline - 20:
            break
        
        # Try exact rational approach first
        result = _try_rank_exact(rank, n, m, p, deadline, max_attempts=5)
        
        if result is None:
            # Fallback to float approach
            result = _try_rank_float(rank, n, m, p, deadline, max_attempts=4)
        
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = rank
            break
    
    return {"U": best_U, "V": best_W, "W": best_W}
# EVOLVE-BLOCK-END
