"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the multiplication tensor T[i,j,k] = delta_{j,k} (for C[i,k] = sum_j A[i,j]*B[j,k])
    # The decomposition requires: for all i,j,k: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # T[i,j,k] = 1 if j==k else 0... wait, no.
    
    # Actually for matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 (it's the identity in the middle index)
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # Wait, that's not right either. Let me think more carefully.
    
    # The matrix multiplication tensor T: T[i, j, k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # Actually the standard form: the multiplication map is a 3-tensor where
    # T[i, j, k] = 1 for the "multiplication" tensor, meaning the bilinear map
    # (A, B) -> A*B has tensor T where T[i,j,k] = delta_{j,k}... 
    
    # Let me use the convention from the problem: W[k*n+i, r]
    # The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] is the matrix multiplication tensor.
    
    # For matrix multiplication C = AB where A is n×m, B is m×p, C is n×p:
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # The tensor T[i,j,k] = 1 (all ones? No...)
    
    # Actually, the multiplication tensor T satisfies:
    # T[i, j, k] = 1 if the entry (i,j) of A and entry (j,k) of B contribute to entry (i,k) of C
    # So T[i,j,k] = 1 for all valid i,j,k? No, that's not a tensor decomposition.
    
    # Let me reconsider. The bilinear map is: f(A,B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # This is a 3-tensor T where T[i, j, k] = 1 (the coefficient of A_{i,j} B_{j,k} in C_{i,k})
    # But we need T[i,j,k] such that the tensor product decomposition gives the right result.
    
    # The correct tensor: T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # No wait, that would mean sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1 for ALL i,j,k
    # But that's not what matrix multiplication does.
    
    # I think I'm confusing myself. Let me look at the schoolbook example:
    # U[i*m+j, column] = 1, V[j*p+k, column] = 1, W[k*n+i, column] = 1
    # for each (i,j,k) triple, with a unique column.
    # So sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1 if there's a column for (i,j,k), else 0
    # Since each (i,j,k) gets its own column, the sum equals 1 for every (i,j,k).
    
    # So the target tensor T[i,j,k] = 1 for all i,j,k? That can't be right for matrix multiplication.
    
    # Ah, I see the issue. The tensor for matrix multiplication is NOT all ones.
    # Let me re-examine: the bilinear form is f(A,B) = A*B
    # The tensor T is defined by T[i,j,k] = 1 if the product A[i,j]*B[j,k] appears in C[i,k]
    # Since every A[i,j]*B[j,k] contributes to C[i,k], T[i,j,k] = 1 for all i,j,k.
    
    # But wait, that means the schoolbook decomposition with rank n*m*p gives T[i,j,k]=1 for all i,j,k.
    # And indeed, sum over all (i',j',k') of delta_{i,i'}delta_{j,j'}delta_{k,k'} = 1.
    
    # So yes, T[i,j,k] = 1 for all i,j,k in the standard indexing.
    
    # Let me verify: for the schoolbook, U[i*m+j, c_{i,j,k}] = 1, V[j*p+k, c_{i,j,k}] = 1, W[k*n+i, c_{i,j,k}] = 1
    # Then sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1 (only the term r=c_{i,j,k} contributes)
    # So T[i,j,k] = 1 for all i,j,k. ✓
    
    # Now for verification, I need to check that for all i,j,k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    errors = 0
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
                if err > 1e-10:
                    errors += 1
    return errors == 0, max_err

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Start with schoolbook as fallback
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
    # Known lower bounds: For matrix multiplication m_{n,p}, rank >= max(n*p, n*m, m*p) = max(24, 20, 30) = 30
    # Also rank >= (n*p + n*m + m*p - something)... 
    # For 4x5x6, let's try ranks from a reasonable lower bound up
    
    # Use numpy for efficient verification
    U_np = np.array(U_school, dtype=float)
    V_np = np.array(V_school, dtype=float)
    W_np = np.array(W_school, dtype=float)
    
    def verify_numpy(U_mat, V_mat, W_mat, n, m, p):
        """Efficient verification using numpy."""
        rank = U_mat.shape[1]
        # Reshape: U_mat is (n*m, rank), V_mat is (m*p, rank), W_mat is (p*n, rank)
        # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This is equivalent to: for each i,k: sum_j,k' ... let me think
        # Actually, let's compute the tensor and check
        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # We can compute this as a batch operation
        
        errors = 0
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U_mat[i*m+j, :]  # (rank,)
                for k in range(p):
                    v_row = V_mat[j*p+k, :]  # (rank,)
                    w_row = W_mat[k*n+i, :]  # (rank,)
                    val = np.dot(u_row, v_row * w_row)
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
                    if err > 1e-8:
                        errors += 1
        return errors == 0, max_err
    
    # Try randomized search for lower ranks
    # Strategy: for each rank r from lower_bound to best_rank-1, try random initializations
    lower_bound = max(n*p, n*m, m*p)  # 30
    
    # Try a few ranks
    for target_rank in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        n_tries = 5
        for trial in range(n_tries):
            if time.time() > deadline:
                break
            
            # Random initialization with small values
            np.random.seed(42 + trial + target_rank)
            U_try = np.random.randn(n*m, target_rank) * 0.5
            V_try = np.random.randn(m*p, target_rank) * 0.5
            W_try = np.random.randn(p*n, target_rank) * 0.5
            
            # Simple gradient-free local search: random perturbations
            best_err = float('inf')
            U_best = U_try.copy()
            V_best = V_try.copy()
            W_best = W_try.copy()
            
            # Do a few iterations of random perturbation
            for iteration in range(50):
                if time.time() > deadline:
                    break
                
                U_cur, V_cur, W_cur = U_best, V_best, W_best
                err, max_e = verify_numpy(U_cur, V_cur, W_cur, n, m, p)
                if err:
                    # Found valid decomposition!
                    # Convert to list of lists with rounded values
                    U_list = [[round(float(x), 10) for x in row] for row in U_cur]
                    V_list = [[round(float(x), 10) for x in row] for row in V_cur]
                    W_list = [[round(float(x), 10) for x in row] for row in W_cur]
                    
                    # Clean up near-zero values
                    for mat in [U_list, V_list, W_list]:
                        for row in mat:
                            for idx in range(len(row)):
                                if abs(row[idx]) < 1e-9:
                                    row[idx] = 0.0
                                elif abs(row[idx] - round(row[idx])) < 1e-9:
                                    row[idx] = round(row[idx])
                                elif abs(row[idx] - round(row[idx]*2)/2) < 1e-9:
                                    row[idx] = round(row[idx]*2)/2
                    
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    break
                
                # Random perturbation
                step = max_e * 0.1 + 0.01
                U_new = U_cur + np.random.randn(n*m, target_rank) * step
                V_new = V_cur + np.random.randn(m*p, target_rank) * step
                W_new = W_cur + np.random.randn(p*n, target_rank) * step
                
                new_err, new_max_e = verify_numpy(U_new, V_new, W_new, n, m, p)
                if new_max_e < best_err:
                    best_err = new_max_e
                    U_best, V_best, W_best = U_new, V_new, W_new
            
            if best_rank < schoolbook_rank:
                break
    
    # Check time
    if time.time() > deadline:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
