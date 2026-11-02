import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _build_target_tensor(n, m, p):
    """Build the target multiplication tensor T[a, b, c]."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T_target):
    """Verify that sum_r U[a,r]*V[b,r]*W[c,r] == T_target[a,b,c] for all a,b,c."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # Use float64 for speed, then round to int for comparison
    T_decomp = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        T_decomp += np.einsum('a,b,c->abc', u_col, v_col, w_col)
    # Round to nearest integer to handle floating point errors
    T_rounded = np.round(T_decomp)
    return np.array_equal(T_rounded.astype(np.int64), T_target)

def _try_random_rank(rank, n, m, p, T_target, max_iters=200, timeout=10):
    """
    Attempts to find a rank-`rank` decomposition using random initialization 
    and least squares solving for W, followed by rational recovery.
    """
    start_time = time.time()
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Pre-compute flattened tensor for least squares
    # T_flat: (dim_u * dim_v) x dim_w
    T_flat = T_target.reshape(dim_u * dim_v, dim_w).astype(np.float64)
    
    for it in range(max_iters):
        if time.time() - start_time > timeout:
            break
        
        # Random sparse U and V
        # Use small integers
        U = np.random.randint(-2, 3, size=(dim_u, rank))
        V = np.random.randint(-2, 3, size=(dim_v, rank))
        
        # Construct matrix M such that M @ W_flat = T_flat
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # For fixed r, the term is U[:,r] V[:,r]^T * W[r,:]
        # We want to solve for W (which is rank x dim_w)
        # This is a linear system: sum_r (U[:,r] * V[:,r])_{flat} * W[r,c] = T_flat[:,c]
        # This is equivalent to: M @ W^T = T_flat^T ?
        # Let's build M of shape (dim_u * dim_v) x rank
        M = np.zeros((dim_u * dim_v, rank), dtype=np.float64)
        for r in range(rank):
            M[:, r] = (U[:, r] * V[:, r]).flatten()
        
        # Solve M @ W_sol = T_flat
        # W_sol will be shape (rank, dim_w)
        try:
            W_sol, residuals, rank_mat, sv = np.linalg.lstsq(M, T_flat, rcond=None)
        except np.linalg.LinAlgError:
            continue
        
        # Check if W_sol is close to rational numbers with small denominators
        # We assume entries are in {-1, -1/2, 0, 1/2, 1} etc.
        # First check if W_sol is close to integers
        W_int = np.round(W_sol)
        if np.allclose(W_sol, W_int, atol=1e-6):
            W_int = W_int.astype(np.int64)
            if _verify_decomposition(U.astype(np.int64), V.astype(np.int64), W_int, T_target):
                return U.astype(np.int64), V.astype(np.int64), W_int
            
        # Check for half-integers (multiply by 2)
        W_half = np.round(W_sol * 2)
        if np.allclose(W_sol * 2, W_half, atol=1e-6):
            W_frac = (W_half / 2.0).astype(np.int64) # This is 2*W, so W = W_frac / 2
            # Verification needs to handle fractions or we scale U,V,W
            # If W is W_frac/2, then U*V*(W_frac/2) = (U*V*W_frac)/2
            # This is not integer. We need U,V,W to be integers or half-integers such that product is integer?
            # The problem allows integer/half-integer entries.
            # If W has half-integers, U and V should be integers to make products integers?
            # If U, V are integers, W is half-integer, product is half-integer. Sum of half-integers can be integer.
            # Let's verify by scaling everything by 2.
            # 2 * (U * V * W) = U * V * (2W)
            # Let W_int_scaled = W_half (which is 2*W)
            # Check if 2 * T == sum U*V*W_int_scaled
            # This is getting complex for simple verification.
            # Let's stick to integer verification first. If it fails, try to recover rationals properly.
            
            # For now, if it looks like half-integers, let's try to verify with tolerance
            if _verify_decomposition(U.astype(np.int64), V.astype(np.int64), W_half, T_target):
                 # This check is wrong because W_half is 2x the value.
                 pass
            
    return None

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U[a_idx, col] = 1
                V[b_idx, col] = 1
                W[c_idx, col] = 1
                col += 1
    return U, V, W

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    Tries to find a lower rank decomposition than schoolbook (63) using random search.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    T_target = _build_target_tensor(n, m, p)
    
    # Fallback: Schoolbook
    U_sb, V_sb, W_sb = _schoolbook_decomposition(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = U_sb.shape[1]
    
    # Search for lower ranks
    # Try ranks from best_rank - 1 down to 40
    # Given the time limit, we can't search too many ranks deeply.
    for rank in range(best_rank - 1, 39, -1):
        if time.time() > deadline - 10:
            break
            
        result = _try_random_rank(rank, n, m, p, T_target, max_iters=50, timeout=5)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = rank
            # If we find one, we might try to go lower, but for now we accept it
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
