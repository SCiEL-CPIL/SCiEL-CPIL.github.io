<<<<<<< SEARCH
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    # For each (i, j, k), the tensor entry is 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # Actually, the tensor T[i,j,k] = 1 for all i,j,k in the standard matrix mult.
    # The decomposition requires: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # Wait, let me re-read the convention.
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # Standard tensor decomposition for matmul (n x m) * (m x p):
    # T_{i,j,k} = sum_r U_{i,j,r} * V_{j,k,r} * W_{k,i,r}
    # where i in [0,n), j in [0,m), k in [0,p)
    # U has shape (n*m, rank), indexed by (i*m+j, r)
    # V has shape (m*p, rank), indexed by (j*p+k, r)
    # W has shape (p*n, rank), indexed by (k*n+i, r)
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    if V_arr.shape[1] != rank or W_arr.shape[1] != rank:
        return False
    
    # Check all tensor entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i*m + j, :]
                v_val = V_arr[j*p + k, :]
                w_val = W_arr[k*n + i, :]
                s = np.dot(np.dot(u_val, v_val), w_val)
                if abs(s - 1.0) > tol:
                    return False
    return True
=======
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    if V_arr.shape[1] != rank or W_arr.shape[1] != rank:
        return False
    
    # Check all tensor entries
    # Condition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i*m + j, :]
                v_val = V_arr[j*p + k, :]
                w_val = W_arr[k*n + i, :]
                # Element-wise product then sum
                s = np.sum(u_val * v_val * w_val)
                if abs(s - 1.0) > tol:
                    return False
    return True
>>>>>>> REPLACE
