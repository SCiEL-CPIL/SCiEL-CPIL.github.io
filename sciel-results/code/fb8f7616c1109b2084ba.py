"""Tensor decomposition search for 3x4x7 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]
START_TIME = time.time()
TIME_LIMIT = 120.0

def _elapsed():
    return time.time() - START_TIME

def _schoolbook(n, m, p):
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

def _verify_decomposition(U, V, W, n, m, p):
    """Check if decomposition exactly decomposes matrix multiplication tensor."""
    if not U or not V or not W:
        return False
    rank = len(U[0]) if U else 0
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.sum(Ua[i*m+j, :] * Va[j*p+k, :] * Wa[k*n+i, :])
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _is_valid_entries(mat):
    """Check if all entries are integers or half-integers."""
    for row in mat:
        for val in row:
            if not isinstance(val, (int, float)):
                return False
            doubled = val * 2
            if abs(doubled - round(doubled)) > 1e-9:
                return False
    return True

def _solve_for_W(U, V, n, m, p, r):
    """Given U and V of rank r, solve linear system for W."""
    # Unknowns: W[k*n+i, c] for k in range(p), i in range(n), c in range(r)
    # Constraints: for each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    num_unknowns = p * n * r
    num_constraints = n * m * p
    
    if num_constraints > num_unknowns:
        return None
    
    A = np.zeros((num_constraints, num_unknowns))
    b = np.ones(num_constraints)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for c in range(r):
                    coeff = U[i*m+j][c] * V[j*p+k][c]
                    if coeff != 0:
                        A[idx, (k*n+i)*r + c] = coeff
                idx += 1
    
    try:
        W_sol, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=1e-10)
        
        # Check if solution is exact
        if not np.allclose(A @ W_sol, b, atol=1e-8):
            return None
        
        # Reshape and round to nearest half-integer
        W_mat = W_sol.reshape(p * n, r)
        W_rounded = np.round(W_mat * 2) / 2
        
        # Verify rounded solution
        if not np.allclose(A @ W_rounded.flatten(), b, atol=1e-8):
            return None
        
        W_list = W_rounded.tolist()
        # Ensure clean values
        for i in range(len(W_list)):
            for j in range(len(W_list[i])):
                W_list[i][j] = round(W_list[i][j] * 2) / 2
        
        return W_list
    except Exception:
        return None

def _construct_structured_UV(n, m, p, r, strategy, seed=42):
    """Construct structured U and V matrices of rank r."""
    np.random.seed(seed)
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    
    if strategy == 0:
        # Each column c: pick one (i,j) for U and one (j,k) for V
        # U[i*m+j, c] = 1, V[j*p+k, c] = 1 (matching j)
        for c in range(r):
            i = c % n
            j = (c // n) % m
            k = (c // (n*m)) % p
            U[i*m + j, c] = 1
            V[j*p + k, c] = 1
    elif strategy == 1:
        # Spread: U has 1s at multiple positions per column
        for c in range(r):
            # Pick 2 positions in U
            idx1 = np.random.randint(0, n*m)
            idx2 = np.random.randint(0, n*m)
            U[idx1, c] = 1
            U[idx2, c] = 1
            # Pick 2 positions in V
            vidx1 = np.random.randint(0, m*p)
            vidx2 = np.random.randint(0, m*p)
            V[vidx1, c] = 1
            V[vidx2, c] = 1
    elif strategy == 2:
        # Block structure: group columns by i
        cols_per_i = r // n
        for i in range(n):
            for ci in range(cols_per_i):
                c = i * cols_per_i + ci
                if c >= r:
                    break
                # U: all j for this i
                for j in range(m):
                    U[i*m + j, c] = 1
                # V: pick one k
                k = ci % p
                for j in range(m):
                    V[j*p + k, c] = 1
    elif strategy == 3:
        # Use half-integers for more flexibility
        for c in range(r):
            i = c % n
            j = (c // n) % m
            k = (c // (n*m)) % p
            U[i*m + j, c] = 0.5
            V[j*p + k, c] = 2.0  # 0.5 * 2.0 = 1.0
    elif strategy == 4:
        # Random sparse with density control
        density = min(1.0, 4.0 / r)
        for c in range(r):
            # Pick ~4 nonzeros in U
            num_u = min(n*m, max(1, int(r * density * 0.5)))
            u_indices = np.random.choice(n*m, size=num_u, replace=False)
            for idx in u_indices:
                U[idx, c] = 1
            num_v = min(m*p, max(1, int(r * density * 0.5)))
            v_indices = np.random.choice(m*p, size=num_v, replace=False)
            for idx in v_indices:
                V[idx, c] = 1
    elif strategy == 5:
        # Structured: U[i,j,c] nonzero only if i == c%n, V[j,k,c] nonzero only if j == c%m
        for c in range(r):
            i = c % n
            j = c % m
            U[i*m + j, c] = 1
            # V: all k for this j
            for k in range(p):
                V[j*p + k, c] = 1
    elif strategy == 6:
        # Alternative: U all-ones per block, V selective
        for c in range(r):
            j = c % m
            k = (c // m) % p
            # U: all i for this j
            for i in range(n):
                U[i*m + j, c] = 1
            V[j*p + k, c] = 1
    elif strategy == 7:
        # Use the fact that we need sum over j: C[i,k] = sum_j A[i,j] B[j,k]
        # Try: for each (i,k), allocate some columns
        # U[i*m+j, c] and V[j*p+k, c] must "agree" on j
        for c in range(r):
            i = c % n
            k = (c // n) % p
            # Distribute among j values
            j = (c // (n*p)) % m
            if (c // (n*p)) >= m:
                j = (c % m)
            U[i*m + j, c] = 1
            V[j*p + k, c] = 1
    elif strategy == 8:
        # Dense random with small values
        U = np.random.choice([0, 0.5, 1], size=(n*m, r))
        V = np.random.choice([0, 0.5, 1], size=(m*p, r))
    elif strategy == 9:
        # Try to exploit structure: for each j, create columns that handle all (i,k) pairs
        # Need n*p = 21 columns per j, total m*n*p = 84 (schoolbook)
        # But maybe we can share across j with linear combinations
        for c in range(r):
            j = c % m
            # For this j, handle some (i,k) pairs
            i = (c // m) % n
            k = (c // (m*n)) % p
            U[i*m + j, c] = 1
            V[j*p + k, c] = 1
    
    return U.tolist(), V.tolist()

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition
    result = _schoolbook(n, m, p)
    U, V, W = result["U"], result["V"], result["W"]
    best_rank = n * m * p
    
    # Lower bound: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    lower_bound = max(n*m, m*p, p*n)
    
    # Try to find lower rank decompositions
    # For each rank r from lower_bound upward, try different strategies
    for r in range(lower_bound, best_rank):
        if _elapsed() > TIME_LIMIT - 15:
            break
        
        for strategy in range(10):
            if _elapsed() > TIME_LIMIT - 10:
                break
            
            # Try multiple seeds for randomized strategies
            seeds = [42, 123, 456, 789, 1000] if strategy in [1, 4, 8] else [42]
            
            for seed in seeds:
                if _elapsed() > TIME_LIMIT - 8:
                    break
                
                U_list, V_list = _construct_structured_UV(n, m, p, r, strategy, seed)
                
                # Solve for W
                W_list = _solve_for_W(U_list, V_list, n, m, p, r)
                
                if W_list is not None:
                    # Verify the full decomposition
                    if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                        # Check entries are valid
                        if _is_valid_entries(U_list) and _is_valid_entries(V_list) and _is_valid_entries(W_list):
                            best_rank = r
                            U, V, W = U_list, V_list, W_list
                            # Found a valid decomposition, try to go lower
                            break
            
            if best_rank == r:
                break  # Found valid at this rank, try lower
    
    return {"U": U, "V": V, "W": W}
