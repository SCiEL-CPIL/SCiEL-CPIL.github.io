"""Find rank-minimal exact tensor decomposition for 2x3x4 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [2, 3, 4]
_START_TIME = time.time()
_DEADLINE = 145.0  # Safety margin before 150s limit


def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[(i,j),(j,k),(k,i')] = 1 if j==j' (the shared index matches).
    Actually, for C = A*B where A is n×m and B is m×p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    So the tensor T has T[(i,j),(j',k),(k',i')] = 1 if j==j' and k==k' and i==i'.
    Wait, let me reconsider. The tensor is in R^{(n*m) × (m*p) × (p*n)}.
    Entry T[alpha, beta, gamma] where:
      alpha = i*m + j  (i in 0..n-1, j in 0..m-1)
      beta  = j'*p + k (j' in 0..m-1, k in 0..p-1)
      gamma = k'*n + i'(k' in 0..p-1, i' in 0..n-1)
    T[alpha, beta, gamma] = 1 if j == j' and k == k' and i == i'
    """
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1.0
    return T


def _decompose_tensor(T, rank, n_iter=200, tol=1e-10):
    """Try to decompose tensor T into rank-r factors using iterative
    alternating least squares (ALS) on the tensor.
    
    We want T ≈ sum_c u_c ⊗ v_c ⊗ w_c
    where u_c is in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}
    
    Using the unfolding approach:
    Fix V, W: solve for U via least squares
    Fix U, W: solve for V via least squares
    Fix U, V: solve for W via least squares
    """
    shape = T.shape
    n1, n2, n3 = shape
    
    # Initialize factors randomly
    U = np.random.randn(n1, rank)
    V = np.random.randn(n2, rank)
    W = np.random.randn(n3, rank)
    
    # Normalize columns
    for c in range(rank):
        norms = [np.linalg.norm(U[:, c]), np.linalg.norm(V[:, c]), np.linalg.norm(W[:, c])]
        if all(n > 1e-12 for n in norms):
            scale = (norms[0] * norms[1] * norms[2]) ** (1.0 / 3.0)
            if scale > 1e-12:
                U[:, c] /= norms[0]
                V[:, c] /= norms[1]
                W[:, c] /= norms[2]
                # Absorb scale into one factor
                U[:, c] *= scale
    
    best_err = np.inf
    best_factors = (U.copy(), V.copy(), W.copy())
    
    for it in range(n_iter):
        # Check deadline
        if time.time() > _DEADLINE - 5:
            break
        
        # Update U: For each entry (i, j, k), T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
        # Fix V, W. Then for each c, we have a linear system.
        # Actually, let's use the matrix unfolding approach.
        # T_1 (unfold along first mode): shape (n1, n2*n3)
        # T_1[i, (j,k)] = sum_c U[i,c] * (V[j,c] * W[k,c])
        # So T_1 = U * diag(1) * ... hmm, this is tricky.
        
        # Let's use the approach: fix V and W, solve for U.
        # T[i, j, k] = sum_c U[i, c] * V[j, c] * W[k, c]
        # Define for each c: W_c = V ⊙ W (element-wise product over c? No...)
        
        # Alternative: think of it as a bilinear optimization.
        # Fix V and W. Then for each i, j, k:
        # T[i,j,k] = sum_c U[i,c] * (V[j,c] * W[k,c])
        # Define M_jk[c] = V[j,c] * W[k,c]
        # Then T[i, :, :] (flattened) = U[i, :] @ M_jk.T? No...
        
        # Let me think differently. Reshape T as a matrix.
        # T_12 (unfold along first mode): shape (n1, n2*n3)
        # T_12[i, j*n3+k] = T[i, j, k] = sum_c U[i,c] * V[j,c] * W[k,c]
        
        # This can be written as:
        # T_12 = U * (V ⊗ W)^T where V⊗W is the Khatri-Rao product... 
        # Actually, if we define B as the matrix where B[j*n3+k, c] = V[j,c]*W[k,c],
        # then T_12 = U @ B.T
        
        # Build B: shape (n2*n3, rank)
        B = np.zeros((n2 * n3, rank))
        for j in range(n2):
            for k in range(n3):
                B[j * n3 + k, :] = V[j, :] * W[k, :]
        
        # T_12 = U @ B.T => U = T_12 @ pinv(B.T) = T_12 @ pinv(B).T
        T_12 = T.reshape(n1, n2 * n3)
        # Solve U @ B.T = T_12 => U = T_12 @ pinv(B.T)
        # Or equivalently, solve least squares: U @ B.T = T_12
        try:
            U_new, _, _, _ = np.linalg.lstsq(B.T, T_12.T, rcond=None)
            U = U_new.T
        except:
            break
        
        # Update V: fix U and W
        # T[i, j, k] = sum_c V[j,c] * (U[i,c] * W[k,c])
        # T_21 (unfold along second mode): shape (n2, n1*n3)
        # T_21[j, i*n3+k] = sum_c V[j,c] * U[i,c] * W[k,c]
        C = np.zeros((n1 * n3, rank))
        for i in range(n1):
            for k in range(n3):
                C[i * n3 + k, :] = U[i, :] * W[k, :]
        T_21 = T.transpose(1, 0, 2).reshape(n2, n1 * n3)
        try:
            V_new, _, _, _ = np.linalg.lstsq(C.T, T_21.T, rcond=None)
            V = V_new.T
        except:
            break
        
        # Update W: fix U and V
        # T[i, j, k] = sum_c W[k,c] * (U[i,c] * V[j,c])
        # T_31 (unfold along third mode): shape (n3, n1*n2)
        # T_31[k, i*n2+j] = sum_c W[k,c] * U[i,c] * V[j,c]
        D = np.zeros((n1 * n2, rank))
        for i in range(n1):
            for j in range(n2):
                D[i * n2 + j, :] = U[i, :] * V[j, :]
        T_31 = T.transpose(2, 0, 1).reshape(n3, n1 * n2)
        try:
            W_new, _, _, _ = np.linalg.lstsq(D.T, T_31.T, rcond=None)
            W = W_new.T
        except:
            break
        
        # Check error
        # Reconstruct T
        T_rec = np.zeros(shape)
        for c in range(rank):
            T_rec += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        err = np.linalg.norm(T - T_rec) / np.linalg.norm(T)
        
        if err < best_err:
            best_err = err
            best_factors = (U.copy(), V.copy(), W.copy())
        
        if err < tol:
            break
    
    U, V, W = best_factors
    # Final check
    T_rec = np.zeros(shape)
    for c in range(rank):
        T_rec += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
    err = np.linalg.norm(T - T_rec) / np.linalg.norm(T)
    
    return U, V, W, err


def construct():
    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)
    T_norm = np.linalg.norm(T)
    
    # Try ranks from low to high
    for rank in range(18, 25):
        if time.time() > _DEADLINE - 10:
            break
        
        # Try multiple random initializations
        found = False
        for trial in range(5):
            if time.time() > _DEADLINE - 5:
                break
            
            np.random.seed(rank * 100 + trial)
            U, V, W, err = _decompose_tensor(T, rank, n_iter=150, tol=1e-12)
            
            if err < 1e-10:
                # Verify exactness
                T_rec = np.zeros(T.shape)
                for c in range(rank):
                    T_rec += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
                if np.allclose(T_rec, T, atol=1e-9):
                    # Convert to list format
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        # If no exact solution found for this rank, continue to next
        pass
    
    # Fallback: schoolbook decomposition
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
