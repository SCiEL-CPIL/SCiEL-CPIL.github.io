"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the multiplication tensor T[i,j,k] = delta_{i,k}
    # For each rank-1 component c, we have:
    # U[:,c] is a vector in R^{n*m}, V[:,c] in R^{m*p}, W[:,c] in R^{p*n}
    # The decomposition says: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # Actually, the tensor for matmul is T[i,j,k] where the product is sum_j A[i,j] B[j,k]
    # The rank-1 decomposition: T = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c[i,j] = U[i*m+j, c], v_c[j,k] = V[j*p+k, c], w_c[k,i] = W[k*n+i, c]
    # And T[i,j,k] = delta_{i,k} (the identity for the middle index)
    # Wait, let me think again.
    
    # For matrix multiplication C = A * B where A is n×m and B is m×p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has indices (i, j, k) and T[i,j,k] = 1 (it's the "multiplication tensor")
    # We want: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # But this doesn't look right dimensionally.
    
    # Let me re-read the problem. The output factor follows W[k*n+i, r].
    # So the decomposition is: for each rank-1 term c,
    # U contributes to A indices, V to B indices, W to C indices
    # The condition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should give
    # the coefficient of A[i,j]*B[j,k] in C[i,k]
    # This coefficient is 1 for all valid i,j,k.
    
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Create schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W, rank

def _local_search(n, m, p, max_time=100):
    """Try to find a lower rank decomposition via local search."""
    start_time = time.time()
    
    # Start with schoolbook
    U, V, W, rank = _schoolbook(n, m, p)
    best_rank = rank
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    # Verify initial
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    # Try to reduce rank by finding linear dependencies
    # Strategy: try to express one column as a combination of others
    # and remove it if possible
    
    current_U = [row[:] for row in U]
    current_V = [row[:] for row in V]
    current_W = [row[:] for row in W]
    current_rank = rank
    
    # Random restarts with different initial perturbations
    for restart in range(50):
        if time.time() - start_time > max_time:
            break
        
        # Work with current best
        U_cur = [row[:] for row in best_U]
        V_cur = [row[:] for row in best_V]
        W_cur = [row[:] for row in best_W]
        r_cur = len(U_cur[0])
        
        # Try removing columns one at a time
        for col_to_remove in range(r_cur):
            if time.time() - start_time > max_time:
                break
            
            # Check if column col_to_remove is in the span of others
            # Build matrix from other columns
            if r_cur <= 1:
                continue
                
            other_cols = [c for c in range(r_cur) if c != col_to_remove]
            
            # For each row in U, V, W, check if the value at col_to_remove 
            # can be expressed as a linear combination of other columns
            # This is a system of equations. Let's use least squares.
            
            # Collect all constraints:
            # For U: U[i, col_to_remove] = sum_{c in other} alpha_c * U[i, c]
            # For V: V[j, col_to_remove] = sum_{c in other} alpha_c * V[j, c]  
            # For W: W[k, col_to_remove] = sum_{c in other} alpha_c * W[k, c]
            
            # All these must use the SAME alpha vector
            # Build the system:
            n_eqs = len(U_cur) + len(V_cur) + len(W_cur)
            n_unknowns = len(other_cols)
            
            A_mat = np.zeros((n_eqs, n_unknowns))
            b_vec = np.zeros(n_eqs)
            
            idx = 0
            for i in range(len(U_cur)):
                for ci, c in enumerate(other_cols):
                    A_mat[idx, ci] = U_cur[i][c]
                b_vec[idx] = U_cur[i][col_to_remove]
                idx += 1
            for j in range(len(V_cur)):
                for ci, c in enumerate(other_cols):
                    A_mat[idx, ci] = V_cur[j][c]
                b_vec[idx] = V_cur[j][col_to_remove]
                idx += 1
            for k in range(len(W_cur)):
                for ci, c in enumerate(other_cols):
                    A_mat[idx, ci] = W_cur[k][c]
                b_vec[idx] = W_cur[k][col_to_remove]
                idx += 1
            
            # Solve least squares
            try:
                alpha, residuals, rank_ls, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except:
                continue
            
            # Check if the solution is good
            residual = np.linalg.norm(A_mat @ alpha - b_vec)
            if residual < 1e-8:
                # Great! We can remove this column
                # Create new matrices without col_to_remove
                new_U = []
                for i in range(len(U_cur)):
                    new_row = []
                    for c in other_cols:
                        new_row.append(U_cur[i][c])
                    new_U.append(new_row)
                
                new_V = []
                for j in range(len(V_cur)):
                    new_row = []
                    for c in other_cols:
                        new_row.append(V_cur[j][c])
                    new_V.append(new_row)
                
                new_W = []
                for k in range(len(W_cur)):
                    new_row = []
                    for c in other_cols:
                        new_row.append(W_cur[k][c])
                    new_W.append(new_row)
                
                # Verify
                if _verify(new_U, new_V, new_W, n, m, p):
                    current_U = new_U
                    current_V = new_V
                    current_W = new_W
                    current_rank = r_cur - 1
                    if current_rank < best_rank:
                        best_rank = current_rank
                        best_U = [row[:] for row in current_U]
                        best_V = [row[:] for row in current_V]
                        best_W = [row[:] for row in current_W]
                    break  # Restart the column removal loop
        
        # After trying all columns, if we improved, continue; else try random perturbation
        if current_rank < r_cur:
            continue
        
        # Try a random perturbation approach: randomly modify entries and see if we can still verify
        # This is less systematic but might find better solutions
        if restart < 10:  # Only do this for first few restarts
            for _ in range(100):
                if time.time() - start_time > max_time:
                    break
                
                # Pick a random column and try to adjust it
                col = random.randint(0, r_cur - 1)
                
                # Try small perturbations
                for _ in range(10):
                    # Randomly perturb one entry
                    mat = random.choice(['U', 'V', 'W'])
                    row = random.randint(0, len({'U': U_cur, 'V': V_cur, 'W': W_cur}[mat]) - 1)
                    
                    if mat == 'U':
                        old_val = U_cur[row][col]
                        U_cur[row][col] = old_val + random.uniform(-0.5, 0.5)
                        if _verify(U_cur, V_cur, W_cur, n, m, p):
                            pass  # Still valid, keep it
                        else:
                            U_cur[row][col] = old_val
                    elif mat == 'V':
                        old_val = V_cur[row][col]
                        V_cur[row][col] = old_val + random.uniform(-0.5, 0.5)
                        if _verify(U_cur, V_cur, W_cur, n, m, p):
                            pass
                        else:
                            V_cur[row][col] = old_val
                    else:
                        old_val = W_cur[row][col]
                        W_cur[row][col] = old_val + random.uniform(-0.5, 0.5)
                        if _verify(U_cur, V_cur, W_cur, n, m, p):
                            pass
                        else:
                            W_cur[row][col] = old_val
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    deadline = 110  # Safety margin before 120s limit
    
    # Get schoolbook as baseline
    U_school, V_school, W_school, rank_school = _schoolbook(n, m, p)
    
    # Try local search
    try:
        U_best, V_best, W_best, rank_best = _local_search(n, m, p, max_time=100)
    except Exception as e:
        # Fall back to schoolbook
        U_best, V_best, W_best, rank_best = U_school, V_school, W_school, rank_school
    
    # Final verification
    if not _verify(U_best, V_best, W_best, n, m, p):
        # Use schoolbook as safe fallback
        U_best, V_best, W_best = U_school, V_school, W_school
    
    return {"U": U_best, "V": V_best, "W": W_best}
# EVOLVE-BLOCK-END
