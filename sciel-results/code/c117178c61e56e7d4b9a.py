import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def _generate_target_tensor(n, m, p):
    """
    Generates the target tensor T[i, j, k] = 1 if it corresponds to a valid multiplication term.
    Indices:
    i: 0..n-1 (row of A, row of C)
    j: 0..m-1 (col of A, row of B)
    k: 0..p-1 (col of B, col of C)
    
    The decomposition is:
    C[i, k] = sum_j A[i, j] B[j, k]
    
    In the tensor form:
    T[i, j, k] = 1 for all i, j, k.
    Wait, the tensor decomposition is:
    T = sum_r U_r \otimes V_r \otimes W_r
    where U_r is a vector in R^{n*m}, V_r in R^{m*p}, W_r in R^{p*n}.
    
    The condition is:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k.
    """
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _als_decompose(T, n, m, p, rank, max_iter=2000, tol=1e-8, seed=42):
    """
    Alternating Least Squares for tensor decomposition.
    T: (n, m, p) target tensor.
    Returns U, V, W or None if not converged.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize U, V, W
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    # Reshape T for easier indexing
    # We need to solve for U given V, W.
    # For each r, U[:, r] is such that sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = T[i,j,k]
    
    # This is a linear system for each column r? No, it's coupled.
    # Actually, ALS solves for one matrix at a time, holding others fixed.
    # To solve for U (size n*m x rank):
    # For each i, j, the term is sum_r U[i*m+j, r] * (V[:, r] * W[:, r] projected?)
    # No, the equation is:
    # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    
    # Let's fix V and W. We want to find U.
    # For a fixed i, j, we have:
    # sum_k T[i, j, k] W[k*n+i, r] ... no, this is not linear in U in a simple way because V and W are coupled in r.
    
    # Correct ALS formulation:
    # We want to minimize || T - sum_r U_r \otimes V_r \otimes W_r ||^2
    # 
    # To update U:
    # For each i, j, the residual is R[i, j, k] = T[i, j, k] - sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # This is not linear in U if we consider all i,j,k together?
    # Actually, for a fixed i, j, the variable is the vector U[i*m+j, :].
    # The equation for all k is:
    # T[i, j, :] = sum_r U[i*m+j, r] * (V[j*p+ :, r] * W[:, r] ? No)
    
    # Let's look at the indices again.
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # For a fixed i, j, we vary k.
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # This is a system of p equations for the rank variables U[i*m+j, 0..rank-1].
    # Let u_ij be the vector U[i*m+j, :].
    # Let M_ij be a p x rank matrix where M_ij[k, r] = V[j*p+k, r] * W[k*n+i, r].
    # Then T[i, j, :] = u_ij @ M_ij.
    # So u_ij = T[i, j, :] @ pinv(M_ij).
    
    # Similarly for V and W.
    
    for it in range(max_iter):
        # Update U
        for i in range(n):
            for j in range(m):
                # Construct M_ij: p x rank
                M_ij = np.zeros((p, rank))
                for k in range(p):
                    for r in range(rank):
                        M_ij[k, r] = V[j*p + k, r] * W[k*n + i, r]
                
                # Solve u_ij @ M_ij = T[i, j, :]
                # u_ij is 1 x rank.
                # Use least squares.
                try:
                    u_ij, residuals, rank_ls, sv = np.linalg.lstsq(M_ij.T, T[i, j, :], rcond=None)
                    U[i*m + j, :] = u_ij
                except np.linalg.LinAlgError:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                # Construct M_jk: n x rank
                # Equation: T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # For fixed j, k, variable is v_jk = V[j*p+k, :].
                # T[:, j, k] = v_jk @ M_jk
                # M_jk[i, r] = U[i*m+j, r] * W[k*n+i, r]
                M_jk = np.zeros((n, rank))
                for i in range(n):
                    for r in range(rank):
                        M_jk[i, r] = U[i*m + j, r] * W[k*n + i, r]
                
                try:
                    v_jk, residuals, rank_ls, sv = np.linalg.lstsq(M_jk.T, T[:, j, k], rcond=None)
                    V[j*p + k, :] = v_jk
                except np.linalg.LinAlgError:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                # Construct M_ki: m x rank
                # Equation: T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # For fixed k, i, variable is w_ki = W[k*n+i, :].
                # T[i, :, k] = w_ki @ M_ki
                # M_ki[j, r] = U[i*m+j, r] * V[j*p+k, r]
                M_ki = np.zeros((m, rank))
                for j in range(m):
                    for r in range(rank):
                        M_ki[j, r] = U[i*m + j, r] * V[j*p + k, r]
                
                try:
                    w_ki, residuals, rank_ls, sv = np.linalg.lstsq(M_ki.T, T[i, :, k], rcond=None)
                    W[k*n + i, :] = w_ki
                except np.linalg.LinAlgError:
                    pass
        
        # Check convergence
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    error += (val - T[i, j, k])**2
        if error < tol:
            return U, V, W
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Target tensor for the whole problem
    T_full = _generate_target_tensor(n, m, p)
    
    # Split into 3x3x3 and 3x3x4
    # T_333: i in 0..2, j in 0..2, k in 0..2
    # T_334: i in 0..2, j in 0..2, k in 3..6
    
    # We will try to find decompositions for these sub-tensors.
    # Note: The indices in U, V, W for the sub-tensors must be mapped correctly.
    
    # For 3x3x3:
    # U_333: (9, R1)
    # V_333: (9, R1)  # j*p+k with p=3? No, V index is j*p+k. In the full problem p=7.
    # This splitting is tricky because V and W indices depend on the global p and n.
    
    # Let's stick to decomposing the FULL tensor directly using ALS.
    # It's safer.
    
    # Try ranks from 45 to 62
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p # 63
    
    # Trivial solution as fallback
    U_triv, V_triv, W_triv = _trivial_decomposition(n, m, p)
    
    for rank in range(45, 63):
        if time.time() > deadline - 10:
            break
            
        # Try a few random seeds
        found = False
        for seed in [42, 123, 456]:
            if time.time() > deadline - 5:
                break
                
            U, V, W = _als_decompose(T_full, n, m, p, rank, max_iter=500, tol=1e-8, seed=seed)
            if U is not None:
                # Verify
                if _verify(U, V, W, n, m, p):
                    best_U, best_V, best_W = U, V, W
                    best_rank = rank
                    found = True
                    break
        if found:
            break
            
    if best_U is None:
        return {"U": U_triv.tolist(), "V": V_triv.tolist(), "W": W_triv.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def _verify(U, V, W, n, m, p, tol=1e-10):
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _trivial_decomposition(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return U, V, W
