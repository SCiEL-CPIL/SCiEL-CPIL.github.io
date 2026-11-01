"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Efficient verification using tensor contraction
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We want T[i, j, k] == 1 for all i, j, k
    
    # Reshape to use einsum or matmul for speed
    # U: (n*m, R)
    # V: (m*p, R)
    # W: (p*n, R)
    
    # Create full tensor T of shape (n, m, p)
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Iterate over c
    # T = sum_c U_c (n*m) outer V_c (m*p) outer W_c (p*n) ? No.
    # T[i,j,k] is the coefficient.
    
    # Use broadcasting
    # U_reshaped: (n, m, R)
    # V_reshaped: (m, p, R)
    # W_reshaped: (p, n, R)
    
    U_r = U_arr.reshape(n, m, rank)
    V_r = V_arr.reshape(m, p, rank)
    W_r = W_arr.reshape(p, n, rank)
    
    # Sum over R
    # T[i, j, k] = sum_r U_r[i, j, r] * V_r[j, k, r] * W_r[k, i, r]
    
    # This is a contraction of 3 tensors.
    # We can do it in a loop over r if R is small, or use einsum.
    # np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    
    # Check if T is all 1s
    if not np.allclose(T, 1.0, atol=tol):
        return False
        
    return True

def _strassen_decomposition():
    """
    Construct a Strassen-based decomposition for 2x4x8 matrix multiplication.
    Base rank 56, then attempt rank reduction via linear dependency detection.
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    strassen_terms = [
        (
            {(0,0): 1, (1,1): 1},   
            {(0,0): 1, (1,1): 1},   
            {(0,0): 1, (1,1): 1}    
        ),
        (
            {(1,0): 1, (1,1): 1},   
            {(0,0): 1},             
            {(1,0): 1, (1,1): -1}   
        ),
        (
            {(0,0): 1},             
            {(0,1): 1, (1,1): -1},  
            {(0,1): 1, (1,1): 1}    
        ),
        (
            {(1,1): 1},             
            {(1,0): 1, (0,0): -1},  
            {(0,0): 1, (1,0): 1}    
        ),
        (
            {(0,0): 1, (0,1): 1},   
            {(1,1): 1},             
            {(0,0): -1, (0,1): 1}   
        ),
        (
            {(1,0): 1, (0,0): -1},  
            {(0,0): 1, (0,1): 1},   
            {(1,1): 1}              
        ),
        (
            {(0,1): 1, (1,1): -1},  
            {(1,0): 1, (1,1): 1},   
            {(0,0): 1}              
        ),
    ]
    
    col = 0
    for a_block in range(2):
        for b_block in range(4):
            for term in strassen_terms:
                A_coeffs, B_coeffs, C_coeffs = term
                
                for (li, lj), coeff in A_coeffs.items():
                    global_i = li
                    global_j = 2 * a_block + lj
                    U[global_i * m + global_j, col] += coeff
                
                for (lj, lk), coeff in B_coeffs.items():
                    global_j = 2 * a_block + lj
                    global_k = 2 * b_block + lk
                    V[global_j * p + global_k, col] += coeff
                
                for (li, lk), coeff in C_coeffs.items():
                    global_i = li
                    global_k = 2 * b_block + lk
                    W[global_k * n + global_i, col] += coeff
                
                col += 1
    
    # Attempt rank reduction: find linearly dependent columns
    # A column c is redundant if it can be expressed as a linear combination of other columns
    # such that the tensor identity still holds.
    # 
    # For matrix multiplication tensor, column c contributes:
    #   sum_i,j,k U[i,j,c] * V[j,k,c] * W[k,i,c] * e_{i,j,k} = 1 * e_{i,j,k} for all i,j,k
    # 
    # If column c = sum_d alpha_d * column_d, then the contribution becomes:
    #   sum_d alpha_d * (contribution of column d)
    # This must equal the contribution of column c, i.e., the identity tensor.
    # But sum of all contributions = identity, so if we remove c and add alpha_d to others,
    # we need: sum_d alpha_d * T_d = T_c, where T_d is the tensor from column d.
    # 
    # A simpler approach: check if any column is a linear combination of others
    # in the joint space (U_col, V_col, W_col). This is complex.
    # 
    # Alternative: try removing each column and see if the remaining can be adjusted.
    # This is NP-hard in general, but for small rank we can try greedy removal.
    
    # Try to find columns that are linearly dependent in a structured way.
    # For the Strassen construction, columns with the same (a_block, b_block) 
    # but different term indices might have dependencies.
    
    # Let's try a simpler approach: check if the rank of the "coefficient tensor"
    # is less than 56 by examining the structure.
    
    # Actually, let's try a direct approach: for each column, check if it can be
    # written as a linear combination of the others such that the tensor identity holds.
    
    # This is equivalent to checking if the column is in the span of the others
    # in the "trilinear" sense. A practical approach:
    # 1. For each candidate column c, try to find coefficients alpha_d (d != c)
    #    such that for all (i,j,k): sum_d alpha_d * U[i,j,d]*V[j,k,d]*W[k,i,d] 
    #    = U[i,j,c]*V[j,k,c]*W[k,i,c]
    # 2. This is a system of n*m*p = 64 linear equations in 55 unknowns.
    # 3. If a solution exists, we can remove column c.
    
    # Let's implement this greedy removal.
    
    def _try_remove_column(U_arr, V_arr, W_arr, c, n, m, p, tol=1e-9):
        """Try to express column c as a linear combination of other columns."""
        R = U_arr.shape[1]
        if R <= 1:
            return False, None
        
        # Build the system: for each (i,j,k), sum_{d != c} alpha_d * prod_d(i,j,k) = prod_c(i,j,k)
        # where prod_d(i,j,k) = U[i*m+j, d] * V[j*p+k, d] * W[k*n+i, d]
        
        # Number of equations: n*m*p
        num_eq = n * m * p
        num_unknowns = R - 1
        
        # Build matrix A (num_eq x num_unknowns) and vector b (num_eq,)
        A = np.zeros((num_eq, num_unknowns), dtype=float)
        b = np.zeros(num_eq, dtype=float)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Target: contribution from column c
                    b[idx] = U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    
                    # Coefficients for other columns
                    for d_idx, d in enumerate(range(R)):
                        if d == c:
                            continue
                        A[idx, d_idx] = U_arr[i*m+j, d] * V_arr[j*p+k, d] * W_arr[k*n+i, d]
                    idx += 1
        
        # Solve least squares
        try:
            alpha, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=tol)
        except np.linalg.LinAlgError:
            return False, None
        
        # Check if the solution is good
        if len(residuals) > 0 and residuals[0] > tol * (np.linalg.norm(b)**2 + 1):
            return False, None
        
        # Verify the solution
        A_alpha = A @ alpha
        if not np.allclose(A_alpha, b, atol=tol * 10):
            return False, None
        
        return True, alpha
    
    # Greedy removal: try to remove each column
    U_work = U.copy()
    V_work = V.copy()
    W_work = W.copy()
    R_work = rank
    removed = []
    
    # Try each column in order (start with last for simplicity)
    for c in range(R_work - 1, -1, -1):
        success, alpha = _try_remove_column(U_work, V_work, W_work, c, n, m, p)
        if success:
            # Remove column c: shift columns after c left
            # U_work[:, c:] = U_work[:, c+1:]
            # This is O(R^2) but R is small (56)
            U_work = np.delete(U_work, c, axis=1)
            V_work = np.delete(V_work, c, axis=1)
            W_work = np.delete(W_work, c, axis=1)
            R_work -= 1
            removed.append(c)
    
    if R_work < rank:
        return {"U": U_work.tolist(), "V": V_work.tolist(), "W": W_work.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook_decomposition():
    """Fallback schoolbook decomposition with rank n*m*p = 64."""
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline_global = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    try:
        result = _strassen_decomposition()
        U, V, W = result["U"], result["V"], result["W"]
        
        # Quick verify
        if _verify(U, V, W, n, m, p, tol=1e-9):
            return result
    except Exception:
        pass
    
    return _schoolbook_decomposition()

# EVOLVE-BLOCK-END
