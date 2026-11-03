"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import time
import sys

def construct():
    """Find a low-rank exact tensor decomposition for multiplying a 3x5 matrix by a 5x6 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns a dict with keys 'U', 'V', 'W' containing the factor matrices.
    """
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    # U shape: (n*m, rank)
    # V shape: (m*p, rank) 
    # W shape: (p*n, rank)
    
    # The matrix multiplication tensor T has entries T[i,j,k] = 1 if the 
    # (i,j) entry of A times (j,k) entry of B contributes to (i,k) entry of C.
    # Specifically, C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is T[i,j,k] = delta_{j,j} which is always 1 for the 
    # standard multiplication tensor. Wait, let me think more carefully.
    #
    # The multiplication tensor M_{n,m,p} has rank r if there exist matrices
    # U (n*m x r), V (m*p x r), W (p*n x r) such that for all i,j,k:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # The standard schoolbook decomposition has rank n*m*p with
    # U[i*m+j, r_{i,j,k}] = 1, V[j*p+k, r_{i,j,k}] = 1, W[k*n+i, r_{i,j,k}] = 1
    # where r_{i,j,k} is the index corresponding to the triple (i,j,k).
    
    # Let's first verify the schoolbook works, then try to find lower rank.
    
    def make_schoolbook():
        """Standard schoolbook decomposition with rank n*m*p."""
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
        return U, V, W, rank
    
    # First get the schoolbook as a baseline
    U_school, V_school, W_school, rank_school = make_schoolbook()
    
    def verify(U, V, W, rank):
        """Verify that the decomposition exactly represents matrix multiplication."""
        # Build the tensor from the decomposition
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For matrix multiplication, T[i,j,k] should be 1 for all i,j,k
        # because C[i,k] = sum_j A[i,j]*B[j,k] and each term A[i,j]*B[j,k] 
        # contributes to C[i,k] with coefficient 1.
        
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_reshaped = U_arr.reshape(n, m, rank)
        V_reshaped = V_arr.reshape(m, p, rank)
        W_reshaped = W_arr.reshape(p, n, rank)
        
        # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # This is a tensor contraction
        # T[i,j,k] = sum_r U_reshaped[i,j,r] * V_reshaped[j,k,r] * W_reshaped[k,i,r]
        
        # For exact verification, we need T[i,j,k] = 1 for all i,j,k
        max_error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Sum over r
                    t_val = 0.0
                    for r in range(rank):
                        t_val += U_reshaped[i, j, r] * V_reshaped[j, k, r] * W_reshaped[k, i, r]
                    error = abs(t_val - 1.0)
                    if error > max_error:
                        max_error = error
        
        return max_error < 1e-9, max_error
    
    # Verify schoolbook
    valid_school, err_school = verify(U_school, V_school, W_school, rank_school)
    
    # Now try to find a lower rank decomposition using ALS (Alternating Least Squares)
    # We'll try various ranks from low to high and see if we can find an exact solution.
    
    best_U, best_V, best_W, best_rank = U_school, V_school, W_school, rank_school
    best_valid = valid_school
    
    # Try ranks from 1 up to rank_school-1
    # For each rank, initialize randomly and run ALS
    
    def try_rank(r, max_iters=500, tol=1e-10):
        """Try to find an exact decomposition of rank r using ALS."""
        # Initialize randomly with small values
        np.random.seed(42)  # Fixed seed for reproducibility
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        # The target tensor is T[i,j,k] = 1 for all i,j,k
        # We want to minimize ||sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1||^2
        # over all i,j,k
        
        for it in range(max_iters):
            # Check time
            if time.time() > deadline:
                return None, None, None, None, False
            
            # Update U: for each (i,j), minimize sum_{k,r'} (sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1)^2
            # Taking derivative w.r.t. U[i,j,u]:
            # sum_k sum_r' V[j,k,r']*W[k,i,r'] * (sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1) * delta_{r,r'} * ... 
            # Actually, let me derive this more carefully.
            #
            # Let f[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
            # We want to minimize sum_{i,j,k} (f[i,j,k] - 1)^2
            #
            # d/dU[i,j,u] sum_{i',j',k'} (f[i',j',k'] - 1)^2
            # = sum_k' 2*(f[i,j,k'] - 1) * V[j,k',u] * W[k',i,u]
            #
            # Setting to zero: sum_k' (f[i,j,k'] - 1) * V[j,k',u] * W[k',i,u] = 0
            # sum_k' f[i,j,k'] * V[j,k',u] * W[k',i,u] = sum_k' V[j,k',u] * W[k',i,u]
            #
            # The left side is: sum_k' (sum_r' U[i,j,r']*V[j,k',r']*W[k',i,r']) * V[j,k',u] * W[k',i,u]
            # = sum_r' U[i,j,r'] * sum_k' V[j,k',r']*W[k',i,r'] * V[j,k',u] * W[k',i,u]
            #
            # Let M[r',u] = sum_k' V[j,k',r']*W[k',i,r'] * V[j,k',u] * W[k',i,u]
            # Then: sum_r' U[i,j,r'] * M[r',u] = sum_k' V[j,k',u] * W[k',i,u]
            #
            # This is a linear system: M * U[i,j,:] = b[i,j,:]
            # where M is r x r and b is r-vector
            
            # Precompute for each (j,i) the matrix M and vector b
            # M depends on j and i (through V[j,:,:] and W[:,i,:])
            # b depends on j and i
            
            # Actually, M[r',u] = sum_k V[j,k,r']*W[k,i,r'] * V[j,k,u]*W[k,i,u]
            # Let A = V[j,:,:] (p x r) and B = W[:,i,:] (p x r)
            # Then M = (A * B)^T * (A * B) where A*B is element-wise, so (A*B) is p x r
            # M = (A*B).T @ (A*B)
            
            # And b[u] = sum_k V[j,k,u] * W[k,i,u]
            
            # For each i, j:
            # M_ij = (V[j,:,:] * W[:,i,:]).T @ (V[j,:,:] * W[:,i,:])
            # b_ij = (V[j,:,:] * W[:,i,:]).sum(axis=0)
            
            # Solve M_ij @ U[i*m+j, :] = b_ij
            
            for i in range(n):
                for j in range(m):
                    # A = V[j,:,:] shape (p, r)
                    # B = W[:,i,:] shape (p, r)
                    elem_prod = V[j, :, :] * W[:, i, :]  # (p, r)
                    M = elem_prod.T @ elem_prod  # (r, r)
                    b = elem_prod.sum(axis=0)  # (r,)
                    
                    # Solve M @ u = b
                    try:
                        u_new = np.linalg.solve(M + 1e-12 * np.eye(r), b)
                    except np.linalg.LinAlgError:
                        u_new = np.linalg.lstsq(M + 1e-12 * np.eye(r), b, rcond=None)[0]
                    
                    U[i * m + j, :] = u_new
            
            # Similarly update V: for each (j,k)
            # d/dV[j,k,u] sum (f-1)^2 = sum_i 2*(f[i,j,k] - 1) * U[i,j,u] * W[k,i,u]
            # sum_i f[i,j,k] * U[i,j,u] * W[k,i,u] = sum_i U[i,j,u] * W[k,i,u]
            # sum_i (sum_r' U[i,j,r']*V[j,k,r']*W[k,i,r']) * U[i,j,u] * W[k,i,u] = sum_i U[i,j,u] * W[k,i,u]
            # sum_r' V[j,k,r'] * sum_i U[i,j,r']*W[k,i,r'] * U[i,j,u] * W[k,i,u] = sum_i U[i,j,u] * W[k,i,u]
            #
            # Let A = U[:,j,:] (n x r) and B = W[k,:,:] (n x r)  [W is indexed as W[k*n+i, r]]
            # elem_prod[i,r'] = U[i*m+j, r'] * W[k*n+i, r']
            # M[r',u] = sum_i U[i*m+j, r']*W[k*n+i, r'] * U[i*m+j, u]*W[k*n+i, u]
            # b[u] = sum_i U[i*m+j, u] * W[k*n+i, u]
            
            for j in range(m):
                for k in range(p):
                    # elem_prod[i, r'] = U[i*m+j, r'] * W[k*n+i, r']
                    elem_prod = np.array([U[i * m + j, :] * W[k * n + i, :] for i in range(n)])  # (n, r)
                    M = elem_prod.T @ elem_prod  # (r, r)
                    b = elem_prod.sum(axis=0)  # (r,)
                    
                    try:
                        v_new = np.linalg.solve(M + 1e-12 * np.eye(r), b)
                    except np.linalg.LinAlgError:
                        v_new = np.linalg.lstsq(M + 1e-12 * np.eye(r), b, rcond=None)[0]
                    
                    V[j * p + k, :] = v_new
            
            # Update W: for each (k,i)
            # d/dW[k,i,u] sum (f-1)^2 = sum_j 2*(f[i,j,k] - 1) * U[i,j,u] * V[j,k,u]
            # sum_j f[i,j,k] * U[i,j,u] * V[j,k,u] = sum_j U[i,j,u] * V[j,k,u]
            # sum_j (sum_r' U[i,j,r']*V[j,k,r']*W[k,i,r']) * U[i,j,u] * V[j,k,u] = sum_j U[i,j,u] * V[j,k,u]
            # sum_r' W[k,i,r'] * sum_j U[i,j,r']*V[j,k,r'] * U[i,j,u] * V[j,k,u] = sum_j U[i,j,u] * V[j,k,u]
            #
            # Let A = U[i*m:(i+1)*m, :] (m x r)  [rows j for fixed i]
            # Let B = V[k*p:(k+1)*p, :] (m x r)  [rows j for fixed k]
            # elem_prod[j, r'] = U[i*m+j, r'] * V[k*p+j, r']
            # M[r',u] = sum_j U[i*m+j, r']*V[k*p+j, r'] * U[i*m+j, u]*V[k*p+j, u]
            # b[u] = sum_j U[i*m+j, u] * V[k*p+j, u]
            
            for k in range(p):
                for i in range(n):
                    # elem_prod[j, r'] = U[i*m+j, r'] * V[k*p+j, r']
                    elem_prod = np.array([U[i * m + j, :] * V[k * p + j, :] for j in range(m)])  # (m, r)
                    M = elem_prod.T @ elem_prod  # (r, r)
                    b = elem_prod.sum(axis=0)  # (r,)
                    
                    try:
                        w_new = np.linalg.solve(M + 1e-12 * np.eye(r), b)
                    except np.linalg.LinAlgError:
                        w_new = np.linalg.lstsq(M + 1e-12 * np.eye(r), b, rcond=None)[0]
                    
                    W[k * n + i, :] = w_new
            
            # Check convergence
            # Compute max error
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        f_val = 0.0
                        for r_idx in range(r):
                            f_val += U[i * m + j, r_idx] * V[j * p + k, r_idx] * W[k * n + i, r_idx]
                        err = abs(f_val - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err < tol:
                return U, V, W, r, True
        
        return U, V, W, r, False
    
    # Try ranks from low to high
    # Start with some promising ranks
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # So rank must be at least 30.
    # Let's try from 30 up to rank_school
    
    min_possible_rank = max(n * m, m * p, p * n)
    
    for rank in range(min_possible_rank, rank_school):
        if time.time() > deadline - 5:  # Leave 5 seconds for finalization
            break
        
        U_try, V_try, W_try, r_try, success = try_rank(rank)
        
        if success:
            # Verify with exact arithmetic or tight tolerance
            valid, err = verify(U_try.tolist(), V_try.tolist(), W_try.tolist(), rank)
            if valid:
                best_U = U_try.tolist()
                best_V = V_try.tolist()
                best_W = W_try.tolist()
                best_rank = rank
                best_valid = True
                break  # Found the lowest rank, we're done
        
        # Try a few more ranks if we're running out of time
        if time.time() > deadline - 10:
            break
    
    # If we didn't find a better rank, use schoolbook
    if not best_valid or best_rank >= rank_school:
        best_U, best_V, best_W, best_rank = U_school, V_school, W_school, rank_school
    
    # Convert to lists if needed
    if not isinstance(best_U, list):
        best_U = best_U.tolist()
    if not isinstance(best_V, list):
        best_V = best_V.tolist()
    if not isinstance(best_W, list):
        best_W = best_W.tolist()
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
