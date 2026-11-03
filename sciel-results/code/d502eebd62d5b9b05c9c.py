"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _TIME_LIMIT - _elapsed()

def _compute_tensor_error_fast(U, V, W, n, m, p):
    """Compute the maximum error of the tensor decomposition using vectorized ops."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the full tensor check using einsum-like approach
    # For each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
    # For invalid (i,j,k) combos, should be 0
    
    max_err = 0.0
    
    # Valid entries: all (i,j,k) in range
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                max_err = max(max_err, abs(s - 1.0))
    
    # Invalid entries
    for x in range(p*n):
        i = x % n
        k = x // n
        for y in range(n*m):
            j = y % m
            i2 = y // m
            for z in range(m*p):
                j2 = z // p
                k2 = z % p
                if i == i2 and j == j2 and k == k2:
                    continue
                s = np.dot(U[y, :], V[z, :] * W[x, :])
                max_err = max(max_err, abs(s))
    
    return max_err

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    return _compute_tensor_error_fast(U, V, W, n, m, p) < tol

def _build_54_rank(n, m, p):
    """
    Build 2x5x6 decomposition using block Strassen. Rank 54.
    Split j into {0,1}, {2,3}, {4}. Split k into {0,1}, {2,3}, {4,5}.
    """
    rank = 54
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # P1 = (A_0,j0 + A_1,j1)(B_j0,k0 + B_j1,k1)
            U[0*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P2 = (A_1,j0 + A_1,j1) B_j0,k0
            U[1*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            W[k0*n + 1][col] += 1
            W[k1*n + 1][col] -= 1
            col += 1
            
            # P3 = A_0,j0 (B_j0,k1 - B_j1,k1)
            U[0*m + j0][col] += 1
            V[j0*p + k1][col] += 1
            V[j1*p + k1][col] -= 1
            W[k1*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P4 = A_1,j1 (B_j1,k0 - B_j0,k0)
            U[1*m + j1][col] += 1
            V[j1*p + k0][col] += 1
            V[j0*p + k0][col] -= 1
            W[k0*n + 0][col] += 1
            W[k0*n + 1][col] += 1
            col += 1
            
            # P5 = (A_0,j0 + A_0,j1) B_j1,k1
            U[0*m + j0][col] += 1
            U[0*m + j1][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] -= 1
            W[k1*n + 0][col] += 1
            col += 1
            
            # P6 = (A_1,j0 - A_0,j0)(B_j0,k0 + B_j0,k1)
            U[1*m + j0][col] += 1
            U[0*m + j0][col] -= 1
            V[j0*p + k0][col] += 1
            V[j0*p + k1][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P7 = (A_0,j1 - A_1,j1)(B_j1,k0 + B_j1,k1)
            U[0*m + j1][col] += 1
            U[1*m + j1][col] -= 1
            V[j1*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            col += 1
            
    # Remaining j=4
    for k in range(p):
        U[0*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 0][col] += 1
        col += 1
        
        U[1*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 1][col] += 1
        col += 1
    
    return {"U": U, "V": V, "W": W}

def _build_50_rank(n, m, p):
    """
    Try to build a rank-50 solution by using a smarter decomposition.
    Use Strassen for 2x2 blocks but share some structure across the j=4 column.
    
    Strategy: 
    - For j in {0,1} x k in {0,1,2,3,4,5}: use 3 Strassen blocks (rank 21)
    - For j in {2,3} x k in {0,1,2,3,4,5}: use 3 Strassen blocks (rank 21)  
    - For j=4: use 4 products instead of 6, by combining k pairs
    Total: 21 + 21 + 8 = 50
    """
    rank = 50
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # P1 = (A_0,j0 + A_1,j1)(B_j0,k0 + B_j1,k1)
            U[0*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P2 = (A_1,j0 + A_1,j1) B_j0,k0
            U[1*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            W[k0*n + 1][col] += 1
            W[k1*n + 1][col] -= 1
            col += 1
            
            # P3 = A_0,j0 (B_j0,k1 - B_j1,k1)
            U[0*m + j0][col] += 1
            V[j0*p + k1][col] += 1
            V[j1*p + k1][col] -= 1
            W[k1*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P4 = A_1,j1 (B_j1,k0 - B_j0,k0)
            U[1*m + j1][col] += 1
            V[j1*p + k0][col] += 1
            V[j0*p + k0][col] -= 1
            W[k0*n + 0][col] += 1
            W[k0*n + 1][col] += 1
            col += 1
            
            # P5 = (A_0,j0 + A_0,j1) B_j1,k1
            U[0*m + j0][col] += 1
            U[0*m + j1][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] -= 1
            W[k1*n + 0][col] += 1
            col += 1
            
            # P6 = (A_1,j0 - A_0,j0)(B_j0,k0 + B_j0,k1)
            U[1*m + j0][col] += 1
            U[0*m + j0][col] -= 1
            V[j0*p + k0][col] += 1
            V[j0*p + k1][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P7 = (A_0,j1 - A_1,j1)(B_j1,k0 + B_j1,k1)
            U[0*m + j1][col] += 1
            U[1*m + j1][col] -= 1
            V[j1*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            col += 1
            
    # For j=4, use 4 products by pairing k indices
    # C_0,k and C_1,k for k=0,1: use (A_0,4 + A_1,4)(B_4,0 + B_4,1) etc.
    # Actually, let's just use 4 direct products with shared structure
    # C_0,0 = A_0,4 * B_4,0
    # C_0,1 = A_0,4 * B_4,1
    # C_1,0 = A_1,4 * B_4,0
    # C_1,1 = A_1,4 * B_4,1
    # We can use: P = (A_0,4 + A_1,4)(B_4,0 + B_4,1), Q = A_0,4*B_4,0, etc.
    # But that's still 4 terms minimum for 2x2.
    
    # Use 4 products for j=4, k in {0,1,2,3,4,5} by grouping:
    # For k pairs (0,1), (2,3), (4,5): use 2 products each = 6 total
    # That's the same as before. Let me try a different approach.
    
    # Alternative: use 4 products total for j=4
    # S = A_0,4 + A_1,4
    # D = A_0,4 - A_1,4
    # For each k pair (k0, k1):
    #   P1 = S * (B_4,k0 + B_4,k1)  -> gives C_0,k0 + C_0,k1 + C_1,k0 + C_1,k1
    #   P2 = D * (B_4,k0 - B_4,k1)  -> gives C_0,k0 - C_0,k1 - C_1,k0 + C_1,k1
    #   P3 = A_0,4 * B_4,k0
    #   P4 = A_1,4 * B_4,k1
    # From these 4 we can extract all 4 values. But that's 4 per pair, 12 total.
    
    # Let's just use 6 direct products for j=4 (same as before)
    for k in range(p):
        U[0*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 0][col] += 1
        col += 1
        
        U[1*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 1][col] += 1
        col += 1
    
    # This gives rank 54 again. The rank-50 idea doesn't work directly.
    # Return None to indicate this approach failed
    return None

def _snap_to_half_integer(x, tol=1e-6):
    """Snap a value to the nearest half-integer if close."""
    half = round(x * 2) / 2.0
    if abs(x - half) < tol:
        return half
    return x

def _snap_to_integer(x, tol=1e-6):
    """Snap a value to the nearest integer if close."""
    whole = round(x)
    if abs(x - whole) < tol:
        return float(whole)
    return x

def _try_solve_for_W(U, V, n, m, p, target_rank):
    """
    Given U and V with target_rank columns, try to solve for W.
    The constraint is: for all (i,j,k) in valid range,
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    For invalid (i,j,k), sum should be 0.
    
    This is a system of linear equations in W.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    
    # W has shape (p*n, target_rank)
    # Unknowns: W[x, r] for x in range(p*n), r in range(target_rank)
    # Total unknowns: p*n*target_rank
    
    # Equations: for each (i,j,k) combination
    # Valid: (i,j,k) where i in [0,n), j in [0,m), k in [0,p) -> p*n*... 
    # Actually, the tensor T[i,j,k] is defined for all i,j,k in their ranges.
    # T[i,j,k] = 1 if this is a "valid" multiplication entry, else 0.
    # For matrix multiplication MM_{n,m,p}, T[i,j,k] = 1 for all i,j,k in range.
    # Wait, that's the full tensor. The "invalid" entries are those where
    # the indices don't correspond to a valid (i,j,k) triple.
    # Actually for MM_{n,m,p}, the tensor is T[i,j,k] = 1 for all i in [0,n),
    # j in [0,m), k in [0,p). There are no "invalid" entries in the standard
    # tensor definition. The "invalid" check in the original code is checking
    # that for a given (x,y,z) = (k*n+i, i2*m+j, j2*p+k2), if (i,j,k) != (i2,j2,k2),
    # then the sum should be 0. This is the condition that the decomposition
    # exactly represents the tensor.
    
    # Let me reformulate. The tensor has entries T[i,j,k] for i in [0,n), j in [0,m), k in [0,p).
    # T[i,j,k] = 1 for all such (i,j,k).
    # The decomposition requires: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # So we have n*m*p equations (all should equal 1) and p*n*target_rank unknowns.
    
    num_eqs = n * m * p
    num_unk = p * n * target_rank
    
    # Build the linear system: A * W_vec = b
    # where W_vec is the flattened W matrix
    
    # For each equation (i,j,k):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is: sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1
    
    # The unknown W[k*n+i, r] appears in equation (i,j,k) with coefficient U[i*m+j, r] * V[j*p+k, r]
    
    # Let's index the unknowns: W_vec[x * target_rank + r] = W[x, r] where x = k*n+i
    # For equation (i,j,k), the coefficient of W_vec[(k*n+i) * target_rank + r] is U[i*m+j, r] * V[j*p+k, r]
    
    # Build A matrix
    A = np.zeros((num_eqs, num_unk))
    b = np.ones(num_eqs)
    
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                x = k * n + i  # row index in W
                for r in range(target_rank):
                    A[eq_idx, x * target_rank + r] = U[i*m+j, r] * V[j*p+k, r]
                eq_idx += 1
    
    # Solve the linear system
    try:
        W_vec, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    
    W = W_vec.reshape(p * n, target_rank)
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p, tol=1e-7):
        return None
    
    return W

def _try_sparse_patterns(n, m, p, target_rank, num_trials=200):
    """
    Try random sparse patterns for U and V, then solve for W.
    """
    best_W = None
    rng = np.random.RandomState(42)
    
    for trial in range(num_trials):
        if _elapsed() > _TIME_LIMIT - 10:
            break
        
        # Create sparse U and V
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        
        for r in range(target_rank):
            # Each column of U and V should have a few non-zero entries
            # Try different sparsity patterns
            num_nonzero_U = rng.randint(1, 4)
            num_nonzero_V = rng.randint(1, 4)
            
            U_indices = rng.choice(n*m, size=num_nonzero_U, replace=False)
            V_indices = rng.choice(m*p, size=num_nonzero_V, replace=False)
            
            for idx in U_indices:
                U[idx, r] = rng.choice([-1, 1])
            for idx in V_indices:
                V[idx, r] = rng.choice([-1, 1])
        
        W = _try_solve_for_W(U, V, n, m, p, target_rank)
        if W is not None:
            best_W = W
            break
    
    return best_W

def _try_structured_reduction(n, m, p, base_U, base_V, base_W, target_rank):
    """
    Try to reduce rank by modifying the base solution.
    Use gradient-based optimization to drive some columns toward zero
    while maintaining the tensor decomposition.
    """
    current_rank = base_U.shape[1]
    if target_rank >= current_rank:
        return None
    
    U_work = base_U.copy().astype(float)
    V_work = base_V.copy().astype(float)
    W_work = base_W.copy().astype(float)
    
    cols_to_zero = current_rank - target_rank
    lr = 0.01
    max_iters = 500
    
    for it in range(max_iters):
        if _elapsed() > _TIME_LIMIT - 10:
            break
        
        # Compute the tensor error
        err = _compute_tensor_error_fast(U_work, V_work, W_work, n, m, p)
        
        # Gradient of penalty: sum of norms of columns to zero
        # We want to minimize sum_{r in zero_cols} (||U[:,r]||^2 + ||V[:,r]||^2 + ||W[:,r]||^2)
        # subject to tensor error = 0
        
        # Simple approach: for each column to zero, compute gradient and take step
        zero_cols = list(range(current_rank - cols_to_zero, current_rank))
        
        # Compute gradient of penalty w.r.t. U, V, W
        # d/dU[idx,r] = 2 * U[idx,r] for r in zero_cols
        # But we also need to maintain the constraint, so we use a Lagrangian approach
        
        # For now, just do a simple projection: scale down the zero columns
        # and see if the error stays small
        for r in zero_cols:
            U_work[:, r] *= 0.98
            V_work[:, r] *= 0.98
            W_work[:, r] *= 0.98
        
        # Check if we can now drop these columns
        col_norms = np.sum(U_work**2, axis=0) + np.sum(V_work**2, axis=0) + np.sum(W_work**2, axis=0)
        small_cols = [r for r in zero_cols if col_norms[r] < 1e-8]
        
        if len(small_cols) >= cols_to_zero:
            keep_cols = [r for r in range(current_rank) if r not in small_cols[:cols_to_zero]]
            U_new = U_work[:, keep_cols]
            V_new = V_work[:, keep_cols]
            W_new = W_work[:, keep_cols]
            
            if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                return {"U": U_new, "V": V_new, "W": W_new}
        
        # Revert if error is too large
        if err > 1e-4:
            U_work = base_U.copy().astype(float)
            V_work = base_V.copy().astype(float)
            W_work = base_W.copy().astype(float)
            break
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Start with the known rank 54 solution
    base_sol = _build_54_rank(n, m, p)
    best_result = base_sol
    best_rank = 54
    
    # Verify the base solution
    if not _verify_decomposition(base_sol["U"], base_sol["V"], base_sol["W"], n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank using structured search
    if _time_left() > 30:
        # Try rank 53
        result_53 = _try_structured_reduction(n, m, p, base_sol["U"], base_sol["V"], base_sol["W"], 53)
        if result_53 is not None:
            best_result = result_53
            best_rank = 53
    
    if best_rank == 54 and _time_left() > 20:
        # Try random sparse patterns for rank 53
        W_53 = _try_sparse_patterns(n, m, p, 53, num_trials=100)
        if W_53 is not None:
            # We need U and V too - they were the ones used in the successful trial
            # This approach is incomplete; let's skip for now
            pass
    
    if best_rank == 54 and _time_left() > 15:
        # Try rank 52
        result_52 = _try_structured_reduction(n, m, p, base_sol["U"], base_sol["V"], base_sol["W"], 52)
        if result_52 is not None:
            best_result = result_52
            best_rank = 52
    
    # Snap to half-integers
    U = best_result["U"]
    V = best_result["V"]
    W = best_result["W"]
    
    U_snap = np.array([[_snap_to_half_integer(x) for x in row] for row in U])
    V_snap = np.array([[_snap_to_half_integer(x) for x in row] for row in V])
    W_snap = np.array([[_snap_to_half_integer(x) for x in row] for row in W])
    
    if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
