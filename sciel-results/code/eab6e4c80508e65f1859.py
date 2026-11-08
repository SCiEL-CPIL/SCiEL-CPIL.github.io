"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import sys
import numpy as np

def _verify(U, V, W, n, m, p):
    """Check that U,V,W form a valid decomposition."""
    rank = len(U[0])
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for c in range(rank):
                    actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _try_merge(U, V, W, c1, c2, n, m, p):
    """Try to merge columns c1 and c2 by adding a rank-1 perturbation.
    Returns (new_U, new_V, new_W) with rank-1 if successful, else None."""
    rank = len(U[0])
    
    # The idea: if we can find vectors a, b, c such that
    # for all (i,j,k): U[i*m+j,c1]*V[j*p+k,c1]*W[k*n+i,c1] + 
    #                   U[i*m+j,c2]*V[j*p+k,c2]*W[k*n+i,c2]
    # can be replaced by a single term plus a correction...
    # 
    # Simpler approach: try to zero out one column by expressing it as
    # a combination of others. This is hard in general.
    # 
    # Alternative: use the fact that if two terms share structure, we can
    # combine them. For matrix multiplication, the known lower bound is
    # around n*m*p - min(n,m,p) + 1 or similar.
    # 
    # Let's try a different approach: randomized local search.
    # Start from schoolbook, try small perturbations to see if we can
    # reduce rank while maintaining correctness.
    return None

def construct():
    deadline = time.time() + 110  # 110s budget, 10s safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Initialize with schoolbook
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U, V, W
    
    # Convert to numpy for faster operations
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Try to reduce rank by removing columns that are linearly dependent
    # or can be expressed through others.
    # 
    # Strategy: For each pair of columns, check if their "tensor contribution"
    # can be absorbed. A column c contributes T_c[i,j,k] = U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c].
    # We want to see if T_c can be written as a linear combination of other T_c's.
    #
    # This is equivalent to checking if the vector (U[:,c], V[:,c], W[:,c]) in some
    # tensor space is in the span of others, but it's nonlinear.
    #
    # Simpler: try to find if any column is "redundant" by checking if setting it
    # to zero and adjusting others still works. This is an NP-hard problem in general,
    # but for small dimensions we can try.
    
    # Alternative approach: use known results. For 3x3 x 3x5, the rank is known
    # to be at most 45 (schoolbook) and likely lower. Let's try a systematic
    # approach: try to remove columns one at a time.
    
    def _check_decomposition(U_mat, V_mat, W_mat, n, m, p):
        """Fast check using numpy."""
        # Build the full tensor and check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_mat[i*m+j], V_mat[j*p+k])
                    # Actually need: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
                    # This is not a simple dot product
                    actual = 0.0
                    for c in range(len(U_mat[i*m+j])):
                        actual += U_mat[i*m+j][c] * V_mat[j*p+k][c] * W_mat[k*n+i][c]
                    if abs(actual - 1.0) > 1e-8:
                        return False
        return True
    
    # Try removing each column and see if we can fix the decomposition
    # by slightly adjusting remaining columns.
    
    # For each column to remove, solve for adjustments to other columns.
    # This is a system of nonlinear equations, but for small rank we can try.
    
    # Let's try a simpler heuristic: for each pair of columns (c1, c2),
    # check if their contributions are "similar" and can be merged.
    
    # Actually, let's try a more principled approach:
    # The matrix multiplication tensor has a known structure. We can try
    # to find a lower-rank decomposition by exploiting symmetries.
    
    # For 3x3 * 3x5, let's try rank 44, 43, etc.
    # We'll use a local search: start from schoolbook, try to eliminate
    # one term at a time by expressing it as a combination of others.
    
    # Term (i,j,k) has contribution U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] for its column c.
    # In schoolbook, each term has its own column with value 1.
    
    # To eliminate term (i0,j0,k0), we need to find coefficients alpha_c for
    # other columns c such that:
    # For all (i,j,k): sum_c alpha_c * U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] 
    #   = U[i*m+j,c0]*V[j*p+k,c0]*W[k*n+i,c0]
    # where c0 is the column for (i0,j0,k0).
    
    # Since in schoolbook U[i*m+j,c] is 1 only when c corresponds to (i,j,*), etc.,
    # this becomes a linear system.
    
    # Let's set up the linear system for eliminating a specific term.
    
    def _can_eliminate_term(U_mat, V_mat, W_mat, n, m, p, target_col, exclude_cols):
        """Check if we can eliminate target_col by adjusting other columns.
        Returns the adjustment vector if possible, else None."""
        rank = len(U_mat[0])
        # Number of equations: n*m*p
        num_eq = n * m * p
        
        # Unknowns: adjustments to columns in exclude_cols
        # We want: sum_{c in exclude_cols} alpha_c * (U[:,c] ⊙ V[:,c] ⊙ W[:,c]) = U[:,target] ⊙ V[:,target] ⊙ W[:,target]
        # where ⊙ denotes the "tensor product" mapping (i,j,k) -> U[i*m+j]*V[j*p+k]*W[k*n+i]
        
        # Build the system
        # For each (i,j,k), the equation is:
        # sum_c alpha_c * U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = U[i*m+j,target]*V[j*p+k,target]*W[k*n+i,target]
        
        # This is linear in alpha_c.
        A = np.zeros((num_eq, rank - 1))  # exclude target column
        b = np.zeros(num_eq)
        
        col_idx = 0
        for c in range(rank):
            if c == target_col:
                continue
            for eq in range(num_eq):
                i = eq // (m * p)
                rem = eq % (m * p)
                j = rem // p
                k = rem % p
                A[eq, col_idx] = U_mat[i*m+j][c] * V_mat[j*p+k][c] * W_mat[k*n+i][c]
            col_idx += 1
        
        # RHS: contribution of target column
        for eq in range(num_eq):
            i = eq // (m * p)
            rem = eq % (m * p)
            j = rem // p
            k = rem % p
            b[eq] = U_mat[i*m+j][target_col] * V_mat[j*p+k][target_col] * W_mat[k*n+i][target_col]
        
        # Solve the linear system
        try:
            alpha, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=None)
            # Check if the solution is exact
            residual = np.linalg.norm(A @ alpha - b)
            if residual < 1e-8:
                return alpha
        except:
            pass
        return None
    
    # Try to eliminate each column
    current_rank = rank_schoolbook
    U_cur = U_np.copy()
    V_cur = V_np.copy()
    W_cur = W_np.copy()
    
    # Try eliminating columns in a greedy manner
    eliminated = []
    for attempt in range(rank_schoolbook):
        if time.time() > deadline:
            break
        
        found_elimination = False
        for target_col in range(current_rank):
            if target_col in eliminated:
                continue
            
            # Check if we can eliminate this column
            alpha = _can_eliminate_term(U_cur, V_cur, W_cur, n, m, p, target_col, set(range(current_rank)) - {target_col})
            if alpha is not None:
                # Apply the elimination
                # The new decomposition has rank-1
                # We need to update U, V, W to reflect the elimination
                # Actually, this approach doesn't directly give us new U,V,W.
                # We need a different formulation.
                pass
        
        if not found_elimination:
            break
    
    # The above approach is flawed. Let's try a different strategy.
    # 
    # Known result: For matrix multiplication, the rank is at most n*m*p - 1
    # for certain cases. Let's try to find a rank 44 decomposition for 3x3*3x5.
    #
    # Actually, let's just try to verify if the schoolbook is already minimal
    # or if we can do better. For now, return the schoolbook if we can't find
    # anything better within the time budget.
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
