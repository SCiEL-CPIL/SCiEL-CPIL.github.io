import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    rank_schoolbook = n * m * p  # 72
    
    # Schoolbook decomposition (guaranteed valid)
    U_schoolbook = np.zeros((n * m, rank_schoolbook))
    V_schoolbook = np.zeros((m * p, rank_schoolbook))
    W_schoolbook = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j, col] = 1
                V_schoolbook[j * p + k, col] = 1
                W_schoolbook[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_schoolbook.copy()
    best_V = V_schoolbook.copy()
    best_W = W_schoolbook.copy()
    
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U, V, W decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1) > tol:
                        return False
        return True
    
    def build_linear_system_for(target_var, U, V, W):
        """
        Build the linear system A x = b where x is the flattened target variable.
        For each (i,j,k), we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        """
        # Number of equations: n*m*p
        num_eqs = n * m * p
        
        if target_var == 'U':
            num_cols = n * m * U.shape[1]
            A = np.zeros((num_eqs, num_cols))
            b = np.ones(num_eqs)
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Coefficient for U[i*m+j, c] is V[j*p+k, c] * W[k*n+i, c]
                        for c in range(U.shape[1]):
                            A[eq_idx, (i * m + j) * U.shape[1] + c] = V[j * p + k, c] * W[k * n + i, c]
                        eq_idx += 1
        elif target_var == 'V':
            num_cols = m * p * V.shape[1]
            A = np.zeros((num_eqs, num_cols))
            b = np.ones(num_eqs)
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(V.shape[1]):
                            A[eq_idx, (j * p + k) * V.shape[1] + c] = U[i * m + j, c] * W[k * n + i, c]
                        eq_idx += 1
        elif target_var == 'W':
            num_cols = p * n * W.shape[1]
            A = np.zeros((num_eqs, num_cols))
            b = np.ones(num_eqs)
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(W.shape[1]):
                            A[eq_idx, (k * n + i) * W.shape[1] + c] = U[i * m + j, c] * V[j * p + k, c]
                        eq_idx += 1
        else:
            raise ValueError("Unknown target variable")
        
        return A, b
    
    def alternating_minimization(U, V, W, max_iter=30, tol=1e-10):
        """
        Alternating minimization to refine U, V, W to exactly satisfy the decomposition.
        """
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Solve for W (linear in W)
            A_w, b_w = build_linear_system_for('W', U, V, W)
            w_sol, _, _, _ = np.linalg.lstsq(A_w, b_w, rcond=None)
            W = w_sol.reshape(p * n, W.shape[1])
            
            # Solve for U (linear in U)
            A_u, b_u = build_linear_system_for('U', U, V, W)
            u_sol, _, _, _ = np.linalg.lstsq(A_u, b_u, rcond=None)
            U = u_sol.reshape(n * m, U.shape[1])
            
            # Solve for V (linear in V)
            A_v, b_v = build_linear_system_for('V', U, V, W)
            v_sol, _, _, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
            V = v_sol.reshape(m * p, V.shape[1])
            
            # Check error
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                        max_err = max(max_err, abs(val - 1))
            
            if max_err < tol:
                break
        
        return U, V, W, max_err
    
    def try_round_and_verify(U, V, W, scale=2):
        """Try rounding to half-integers and verify."""
        U_r = np.round(U * scale) / scale
        V_r = np.round(V * scale) / scale
        W_r = np.round(W * scale) / scale
        
        if check_decomposition(U_r, V_r, W_r, tol=1e-8):
            return U_r, V_r, W_r
        return None
    
    # Try ranks from lower to higher
    # For 3x3x8, known lower bounds suggest rank might be achievable below 72
    target_ranks = [60, 62, 64, 66, 68, 70]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 15:
            break
        
        found = False
        
        # Multiple random initializations
        for seed in range(15):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(seed * 1000 + target_rank)
            
            # Initialize with random values
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Run alternating minimization
            U_ref, V_ref, W_ref, max_err = alternating_minimization(U, V, W, max_iter=25)
            
            if max_err < 1e-8:
                # Try to round to half-integers
                rounded = try_round_and_verify(U_ref, V_ref, W_ref, scale=2)
                if rounded is not None:
                    U_r, V_r, W_r = rounded
                    best_rank = target_rank
                    best_U, best_V, best_W = U_r, V_r, W_r
                    found = True
                    break
                else:
                    # Use the unrounded solution if it's exact
                    best_rank = target_rank
                    best_U, best_V, best_W = U_ref, V_ref, W_ref
                    found = True
                    break
        
        if found:
            break
    
    # If we still have schoolbook rank, try one more time with structured init
    if best_rank == rank_schoolbook and time.time() < deadline - 10:
        for target_rank in [64, 66, 68]:
            if time.time() > deadline - 10:
                break
            
            for seed in range(10):
                if time.time() > deadline - 10:
                    break
                
                np.random.seed(seed * 5000 + target_rank)
                
                # Structured initialization: start from schoolbook and perturb
                U = np.zeros((n * m, target_rank))
                V = np.zeros((m * p, target_rank))
                W = np.zeros((p * n, target_rank))
                
                # Copy first min(target_rank, 72) columns from schoolbook
                copy_cols = min(target_rank, rank_schoolbook)
                for c in range(copy_cols):
                    i = c // (m * p)
                    j = (c // p) % m
                    k = c % p
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                
                # Add random noise to remaining columns
                for c in range(copy_cols, target_rank):
                    U[:, c] = np.random.randn(n * m) * 0.1
                    V[:, c] = np.random.randn(m * p) * 0.1
                    W[:, c] = np.random.randn(p * n) * 0.1
                
                U_ref, V_ref, W_ref, max_err = alternating_minimization(U, V, W, max_iter=25)
                
                if max_err < 1e-8:
                    rounded = try_round_and_verify(U_ref, V_ref, W_ref, scale=2)
                    if rounded is not None:
                        U_r, V_r, W_r = rounded
                        best_rank = target_rank
                        best_U, best_V, best_W = U_r, V_r, W_r
                        break
                    else:
                        best_rank = target_rank
                        best_U, best_V, best_W = U_ref, V_ref, W_ref
                        break
            
            if best_rank < rank_schoolbook:
                break
    
    # Final verification
    if not check_decomposition(best_U, best_V, best_W, tol=1e-8):
        # Fall back to schoolbook
        best_U = U_schoolbook
        best_V = V_schoolbook
        best_W = W_schoolbook
        best_rank = rank_schoolbook
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
