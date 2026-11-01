"""Matrix multiplication 2x4 x 4x7 rank-minimal tensor decomposition."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the expected tensor Texp for matrix multiplication."""
    Texp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                Texp[a, b, c] = 1.0
    return Texp

def _schoolbook(n, m, p):
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

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that U,V,W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build expected tensor Texp
    Texp = _build_tensor(n, m, p)
    
    # Compute actual tensor T = sum_r U[:,r] V[:,r] W[:,r]
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    
    err = np.max(np.abs(T - Texp))
    return err < tol

def _try_reduce_rank(U, V, W, n, m, p, tol=1e-9):
    """
    Attempt to reduce the rank by 1 by removing a term.
    U, V, W are numpy arrays of shape (dim, rank).
    Returns (new_U, new_V, new_W, success).
    """
    rank = U.shape[1]
    if rank <= 1:
        return U, V, W, False
    
    Texp = _build_tensor(n, m, p)
    
    # Try removing each term
    for r_remove in range(rank):
        # Create copies without the r_remove-th column
        U_new = np.delete(U, r_remove, axis=1)
        V_new = np.delete(V, r_remove, axis=1)
        W_new = np.delete(W, r_remove, axis=1)
        
        # The residual tensor that the remaining terms must cover
        # Texp = sum_{i != r} u_i v_i w_i + u_r v_r w_r
        # sum_{i != r} u_i v_i w_i = Texp - u_r v_r w_r
        u_r = U[:, r_remove]
        v_r = V[:, r_remove]
        w_r = W[:, r_remove]
        T_target = Texp - np.einsum('a,b,c->abc', u_r, v_r, w_r)
        
        # We need to check if T_target can be written as sum_{i != r} u'_i v'_i w'_i
        # where u'_i, v'_i, w'_i are linear combinations of the remaining basis vectors?
        # No, the vectors U_new, V_new, W_new are fixed.
        # We are checking if there exist coefficients c_i such that
        # T_target = sum_i c_i (U_new[:,i] \otimes V_new[:,i] \otimes W_new[:,i])
        
        # This is a linear system.
        # Flatten T_target to a vector of length N*M*P.
        # Flatten each rank-1 tensor to a vector.
        # A = [vec(u_1 v_1 w_1), ..., vec(u_{R-1} v_{R-1} w_{R-1})]
        # b = vec(T_target)
        # Solve A c = b
        
        N, M, P = n*m, m*p, p*n
        A = np.zeros((N*M*P, rank - 1))
        for i in range(rank - 1):
            u_i = U_new[:, i]
            v_i = V_new[:, i]
            w_i = W_new[:, i]
            tensor_i = np.einsum('a,b,c->abc', u_i, v_i, w_i)
            A[:, i] = tensor_i.flatten()
        
        b = T_target.flatten()
        
        try:
            # Solve the least squares problem
            # If the system is consistent, the residual will be 0.
            # We use np.linalg.lstsq
            sol, residuals, rank_lstsq, s = np.linalg.lstsq(A, b, rcond=None)
            
            # Check if the solution is exact
            # Reconstruct T_approx
            T_approx = np.zeros((N, M, P))
            for i in range(rank - 1):
                T_approx += sol[i] * np.einsum('a,b,c->abc', U_new[:, i], V_new[:, i], W_new[:, i])
            
            err = np.max(np.abs(T_approx - T_target))
            
            if err < tol:
                # We found a valid decomposition of rank-1
                # Note: The vectors U_new, V_new, W_new are the same, but the coefficients are sol.
                # We can absorb sol into U, V, or W.
                # Let's absorb into U.
                U_final = U_new * sol[None, :]
                V_final = V_new
                W_final = W_new
                
                return U_final, V_final, W_final, True
        except np.linalg.LinAlgError:
            continue
            
    return U, V, W, False

def _greedy_rank_reduction(n, m, p, max_iterations=100):
    """
    Start with schoolbook and iteratively try to reduce rank.
    """
    result = _schoolbook(n, m, p)
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    
    deadline = time.time() + 100 # Leave some time for construct
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to reduce rank by 1
        U_new, V_new, W_new, success = _try_reduce_rank(U, V, W, n, m, p)
        if success:
            U, V, W = U_new, V_new, W_new
            improved = True
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    
    # Try greedy rank reduction
    if time.time() < deadline - 20:
        reduced_result = _greedy_rank_reduction(n, m, p)
        # Verify the result
        if _verify(reduced_result['U'], reduced_result['V'], reduced_result['W'], n, m, p):
            result = reduced_result
    
    return result

# EVOLVE-BLOCK-END
