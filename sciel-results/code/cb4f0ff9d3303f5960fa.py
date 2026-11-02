"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _validate_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    # Build the multiplication tensor T[i,j,k] = 1 if the (i,j) element of A times (j,k) element of B
    # contributes to (i,k) element of C.
    # The tensor is T[i,j,k] where A is n×m, B is m×p, C is n×p
    # T[i,j,k] = 1 for all valid i,j,k (it's the structure tensor)
    # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # But we need to check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Check a sample of entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1) > 1e-10:
                    return False
    return True

def _rank_of_matrix(M):
    """Compute the numerical rank of a matrix."""
    try:
        s = np.linalg.svd(np.array(M, dtype=float), compute_uv=False)
        return int(np.sum(s > 1e-10))
    except:
        return len(M[0]) if M else 0

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    deadline = time.time() + 130  # 130s to leave safety margin
    
    # Start with schoolbook decomposition
    best_U = [[0] * schoolbook_rank for _ in range(n * m)]
    best_V = [[0] * schoolbook_rank for _ in range(m * p)]
    best_W = [[0] * schoolbook_rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
    
    best_rank = schoolbook_rank
    
    # Try to reduce rank by finding linear dependencies
    # The idea: if we can find coefficients alpha_c such that
    # sum_c alpha_c * U[:,c] = 0, sum_c alpha_c * V[:,c] = 0, sum_c alpha_c * W[:,c] = 0
    # then we can remove that direction and reduce rank by 1.
    
    # More generally, try to find a set of columns that are linearly dependent
    # across all three tensors simultaneously.
    
    # Strategy: randomly perturb the decomposition and try to find lower rank
    # using the Bini-Capovani method
    
    # First, try a simpler approach: check if any subset of columns is linearly dependent
    # in all three matrices simultaneously.
    
    # For each pair of columns, check if they can be combined
    # Actually, let's try a more systematic approach:
    # Try to find rank-1 updates that allow reduction
    
    # Method: Try to express one column as a linear combination of others
    # in all three tensors simultaneously.
    
    # For a more effective search, use random projections
    np.random.seed(42)
    
    attempts = 0
    while time.time() < deadline and attempts < 5000:
        attempts += 1
        
        # Try to reduce rank by 1 by finding a linear dependency
        # among the current rank columns
        
        # Build the constraint matrix: we want alpha such that
        # sum_c alpha_c * U[:,c] = 0
        # sum_c alpha_c * V[:,c] = 0  
        # sum_c alpha_c * W[:,c] = 0
        # This is a homogeneous system. Non-trivial solutions exist if
        # the combined matrix has rank < current_rank.
        
        if best_rank <= 1:
            break
        
        # Stack the columns from all three tensors
        # The constraint is: for each row i in U, sum_c alpha_c * U[i,c] = 0
        # for each row j in V, sum_c alpha_c * V[j,c] = 0
        # for each row k in W, sum_c alpha_c * W[k,c] = 0
        
        U_mat = np.array(best_U[:best_rank], dtype=float)  # n*m x best_rank
        V_mat = np.array(best_V[:best_rank], dtype=float)  # m*p x best_rank
        W_mat = np.array(best_W[:best_rank], dtype=float)  # p*n x best_rank
        
        # The null space of the stacked matrix gives us linear dependencies
        # But we need the dependency to hold simultaneously in all three.
        # Actually, if sum_c alpha_c * U[:,c] = 0, this doesn't guarantee
        # sum_c alpha_c * V[:,c] = 0 or sum_c alpha_c * W[:,c] = 0.
        
        # We need to find alpha in the intersection of null spaces.
        # This is equivalent to finding alpha such that:
        # U_mat^T * alpha = 0, V_mat^T * alpha = 0, W_mat^T * alpha = 0
        # i.e., alpha is in the null space of [U_mat^T; V_mat^T; W_mat^T]
        
        # But this is a very high-dimensional constraint. Let's think differently.
        
        # Alternative: Try to find a rank-1 update that allows merging two columns.
        # If columns a and b can be merged, there exist scalars s,t such that
        # U[:,a] = s*U[:,b] + u', V[:,a] = s*V[:,b] + v', W[:,a] = s*W[:,b] + w'
        # where u'*v'*w' = 0 (the rank-1 component vanishes).
        
        # This is complex. Let's try a randomized approach:
        # Randomly perturb the decomposition and check if we can find a lower rank
        # representation.
        
        # Simpler: Try to find if any two columns are proportional in all three tensors
        found_reduction = False
        for a in range(best_rank - 1):
            if time.time() > deadline:
                break
            for b in range(a + 1, best_rank):
                # Check if column a is proportional to column b in all three
                # U[:,a] = s * U[:,b], V[:,a] = s * V[:,b], W[:,a] = s * W[:,b]
                
                # Find s from U
                nonzero_a = np.nonzero(U_mat[:, a])[0]
                nonzero_b = np.nonzero(U_mat[:, b])[0]
                
                if len(nonzero_a) == 0 or len(nonzero_b) == 0:
                    continue
                    
                # Check proportionality
                idx = nonzero_a[0]
                if U_mat[idx, b] == 0:
                    continue
                s = U_mat[idx, a] / U_mat[idx, b]
                
                # Verify proportionality in U
                if not np.allclose(U_mat[nonzero_a, a], s * U_mat[nonzero_a, b], atol=1e-10):
                    continue
                if not np.allclose(U_mat[nonzero_b, a], s * U_mat[nonzero_b, b], atol=1e-10):
                    continue
                
                # Verify in V
                if not np.allclose(V_mat[:, a], s * V_mat[:, b], atol=1e-10):
                    continue
                
                # Verify in W
                if not np.allclose(W_mat[:, a], s * W_mat[:, b], atol=1e-10):
                    continue
                
                # Found a proportional pair! Remove column a
                # Create new matrices without column a
                mask = [c for c in range(best_rank) if c != a]
                new_U = [best_U[i][:] for i in range(n * m)]
                new_V = [best_V[i][:] for i in range(m * p)]
                new_W = [best_W[i][:] for i in range(p * n)]
                
                for i in range(n * m):
                    new_U[i] = [best_U[i][c] for c in mask]
                for i in range(m * p):
                    new_V[i] = [best_V[i][c] for c in mask]
                for i in range(p * n):
                    new_W[i] = [best_W[i][c] for c in mask]
                
                # Validate
                if _validate_decomposition(new_U, new_V, new_W, n, m, p):
                    best_U = new_U
                    best_V = new_V
                    best_W = new_W
                    best_rank -= 1
                    found_reduction = True
                    break
            
            if found_reduction:
                break
        
        if not found_reduction and best_rank > 1:
            # Try random perturbation approach
            # Add small random noise and see if we can find a lower rank structure
            # This is heuristic and may not work well
            pass
    
    # Final validation
    if not _validate_decomposition(best_U, best_V, best_W, n, m, p):
        # Fallback to schoolbook
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * m * p + j * p + k
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
