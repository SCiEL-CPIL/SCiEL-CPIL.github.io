import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return schoolbook decomposition for n×m × m×p matrix multiplication."""
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

def _validate_exact(A, B, C, n, m, p):
    """
    Exact integer validation using NumPy.
    A, B, C are integer arrays (representing 2*U, 2*V, 2*W).
    Check that for all (i,j,k): sum_c A[i*m+j, c] * B[j*p+k, c] * C[k*n+i, c] == 8
    """
    r = A.shape[1]
    # We need to check all (i, j, k)
    # A_idx = i*m + j
    # B_idx = j*p + k
    # C_idx = k*n + i
    
    # Precompute indices
    i_vals = np.repeat(np.arange(n), m * p)
    j_vals = np.tile(np.repeat(np.arange(m), p), n)
    k_vals = np.tile(np.arange(p), n * m)
    
    a_idx = i_vals * m + j_vals
    b_idx = j_vals * p + k_vals
    c_idx = k_vals * n + i_vals
    
    # Extract rows
    A_rows = A[a_idx, :]
    B_rows = B[b_idx, :]
    C_rows = C[c_idx, :]
    
    # Compute sum over c
    # (num_eqs, r) * (num_eqs, r) -> (num_eqs, r)
    # Element-wise multiply then sum
    products = A_rows * B_rows * C_rows
    sums = products.sum(axis=1)
    
    return np.all(sums == 8)

def _solve_for_W(U_int, V_int, n, m, p, r):
    """
    Given integer U_int and V_int (each r columns), solve for W_int.
    System: for each (i,j,k), sum_c U_int[i*m+j, c] * V_int[j*p+k, c] * W_int[k*n+i, c] = 8
    
    This is a linear system in W_int. Build the matrix and solve.
    Returns W_int as a 2D list if a valid integer solution exists, else None.
    """
    dim_c = p * n
    num_eqs = n * m * p  # 36 for 3x3x4
    
    # Build system: M * w_vec = b
    # w_vec is dim_c * r elements, indexed as [col * dim_c + c_idx]
    # Each equation (i,j,k) constrains: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 8
    
    M = np.zeros((num_eqs, dim_c * r), dtype=np.int64)
    b = np.ones(num_eqs, dtype=np.int64) * 8
    
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            a_idx = i * m + j
            for k in range(p):
                b_idx = j * p + k
                c_idx = k * n + i
                for col in range(r):
                    coeff = U_int[a_idx][col] * V_int[b_idx][col]
                    if coeff != 0:
                        M[eq_idx, col * dim_c + c_idx] = coeff
                eq_idx += 1
    
    # Solve using least squares, then check if solution is integer
    try:
        # Use float for lstsq
        w_sol, residuals, rank, sv = np.linalg.lstsq(M.astype(np.float64), b.astype(np.float64), rcond=None)
        if len(residuals) > 0 and residuals[0] > 1e-6:
            return None
        
        # Check if solution is close to integer
        w_rounded = np.round(w_sol).astype(np.int64)
        
        # Verify exactness
        residual_check = M @ w_rounded - b
        if not np.all(residual_check == 0):
            return None
        
        # Reshape w_rounded to (r, dim_c) then transpose to get W_int (dim_c, r)
        # w_rounded[col * dim_c + c_idx] = W_int[c_idx, col]
        W_int = w_rounded.reshape(r, dim_c).T  # shape (dim_c, r)
        
        return W_int.tolist()
    except Exception:
        return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Schoolbook as fallback
    schoolbook = _schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p  # 36
    
    dim_a = n * m  # 9
    dim_b = m * p  # 12
    dim_c = p * n  # 12
    
    rng = random.Random(42)
    
    # Try ranks from low to high (we want minimum rank)
    # Lower bound: max(dim_a, dim_b, dim_c) = 12
    # Try a range of ranks
    ranks_to_try = list(range(12, best_rank))  # 12 to 35
    
    # For each rank, try multiple random U, V pairs
    # Use different value ranges for U, V entries
    
    value_ranges = [
        [-1, 0, 1],
        [-2, -1, 0, 1, 2],
        [-3, -2, -1, 0, 1, 2, 3],
        [0, 1],
        [-1, 1],
    ]
    
    # Interleave: for each rank, try several value ranges and random draws
    # Prioritize lower ranks
    
    for r in ranks_to_try:
        if time.time() - start_time > deadline - 5:
            break
        
        # Number of trials per rank: more for lower ranks
        # We can afford more trials now due to faster validation
        trials = 500
        
        for trial in range(trials):
            if time.time() - start_time > deadline - 5:
                break
            
            # Choose value range
            vr = value_ranges[trial % len(value_ranges)]
            
            # Generate random U_int and V_int
            U_int = np.array([[rng.choice(vr) for _ in range(r)] for _ in range(dim_a)], dtype=np.int64)
            V_int = np.array([[rng.choice(vr) for _ in range(r)] for _ in range(dim_b)], dtype=np.int64)
            
            # Solve for W_int
            W_int_list = _solve_for_W(U_int, V_int, n, m, p, r)
            
            if W_int_list is not None:
                # Validate exactness
                W_int = np.array(W_int_list, dtype=np.int64)
                if _validate_exact(U_int, V_int, W_int, n, m, p):
                    # Found valid decomposition!
                    # Convert to half-integers
                    U_half = [[x / 2 for x in row] for row in U_int]
                    V_half = [[x / 2 for x in row] for row in V_int]
                    W_half = [[x / 2 for x in row] for row in W_int]
                    
                    best_result = {"U": U_half, "V": V_half, "W": W_half}
                    best_rank = r
                    # Found a valid one at rank r, try to go lower
                    break
        
        # If we found a valid one at this rank, continue to lower ranks
        # (already handled by loop structure)
    
    return best_result
# EVOLVE-BLOCK-END
