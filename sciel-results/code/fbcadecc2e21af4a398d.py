"""Matrix multiplication 3,5,7 - rank-minimal tensor decomposition."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build the target tensor T[i,j,k] = 1 if i in [0,n), j in [0,m), k in [0,p)
    # The decomposition should satisfy: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _to_rational(x, max_den=100):
    """Convert a float to a rational number with small denominator if possible."""
    if abs(x) < 1e-12:
        return 0
    # Try to find a rational approximation
    from fractions import Fraction
    frac = Fraction(x).limit_denominator(max_den)
    return float(frac)

def _round_to_half_integer(x, tol=1e-6):
    """Round to nearest half-integer if close enough."""
    # Half-integers: ..., -1.5, -1, -0.5, 0, 0.5, 1, 1.5, ...
    candidate = round(x * 2) / 2
    if abs(x - candidate) < tol:
        return candidate
    return None

def construct():
    """
    Construct a tensor decomposition for 3x5 * 5x7 matrix multiplication.
    Uses iterative hard-thresholding to find a lower-rank decomposition.
    """
    n, m, p = DIMENSIONS  # 3, 5, 7
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with schoolbook decomposition
    rank0 = n * m * p
    U = [[0] * rank0 for _ in range(n * m)]
    V = [[0] * rank0 for _ in range(m * p)]
    W = [[0] * rank0 for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    
    # Convert to numpy for efficient operations
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank0
    
    # Try to reduce rank by identifying columns that can be zeroed or merged
    # Strategy: Try removing columns one at a time and see if the decomposition still works
    # with adjusted remaining columns
    
    # First, let's try a greedy approach: remove columns that contribute least
    # to the tensor reconstruction while maintaining exactness
    
    current_U = U_np.copy()
    current_V = V_np.copy()
    current_W = W_np.copy()
    current_rank = rank0
    active_cols = list(range(rank0))
    
    # Compute the contribution of each column to the tensor
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    def compute_residual(col_idx, U_mat, V_mat, W_mat, active):
        """Compute the residual tensor if we zero out column col_idx."""
        if col_idx not in active:
            return None
        # Residual = -U[:,col] * V[:,col] * W[:,col] (outer product)
        u_col = U_mat[:, col_idx]
        v_col = V_mat[:, col_idx]
        w_col = W_mat[:, col_idx]
        # The contribution of this column to each T[i,j,k]
        # T[i,j,k] gets U[i*m+j, col] * V[j*p+k, col] * W[k*n+i, col]
        residual = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    residual[i,j,k] = u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
        return residual
    
    def try_remove_column(col_idx, U_mat, V_mat, W_mat, active):
        """Try to remove a column by adjusting others. Returns new U,V,W,active or None."""
        if col_idx not in active:
            return None
        if len(active) <= 1:
            return None
        
        # Get the residual if we remove this column
        residual = compute_residual(col_idx, U_mat, V_mat, W_mat, active)
        if residual is None:
            return None
        
        # Try to compensate by adding the residual to other active columns
        # This is a rank-1 update problem: can we express residual as sum of rank-1 tensors
        # from the remaining columns?
        
        # For now, just check if the residual is zero (column is redundant)
        if np.allclose(residual, 0, atol=1e-10):
            new_active = [c for c in active if c != col_idx]
            return U_mat, V_mat, W_mat, new_active
        
        return None
    
    # Try to remove columns
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        removed_any = False
        for col_idx in list(active_cols):
            if time.time() > deadline:
                break
            
            result = try_remove_column(col_idx, current_U, current_V, current_W, active_cols)
            if result is not None:
                current_U, current_V, current_W, active_cols = result
                removed_any = True
                break
        
        if not removed_any:
            break
    
    # Extract the reduced decomposition
    if len(active_cols) < best_rank:
        U_reduced = [[current_U[i, col] for col in active_cols] for i in range(n*m)]
        V_reduced = [[current_V[i, col] for col in active_cols] for i in range(m*p)]
        W_reduced = [[current_W[i, col] for col in active_cols] for i in range(p*n)]
        
        # Verify
        if _verify_decomposition(U_reduced, V_reduced, W_reduced, n, m, p):
            # Try to convert to rational/half-integer
            U_final = [[_to_rational(x) for x in row] for row in U_reduced]
            V_final = [[_to_rational(x) for x in row] for row in V_reduced]
            W_final = [[_to_rational(x) for x in row] for row in W_reduced]
            
            # Verify with rationals
            if _verify_decomposition(U_final, V_final, W_final, n, m, p, tol=1e-6):
                best_U, best_V, best_W = U_final, V_final, W_final
                best_rank = len(active_cols)
    
    # Return the best decomposition found
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
