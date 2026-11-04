import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # Build the known rank-39 decomposition using Strassen blocks
    def build_strassen_decomposition():
        # Rank calculation:
        # 3 Strassen blocks (2x2x2) for k=0,1; 2,3; 4,5 -> 3 * 7 = 21
        # 1 Schoolbook block (2x2x1) for k=6 -> 4
        # 1 Schoolbook block (2x1x7) for j=2 -> 14
        rank = 21 + 4 + 14
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        # Helper to add a term
        def add_term(u_indices, v_indices, w_indices, coeffs=None):
            nonlocal col
            if coeffs is None:
                coeffs = [1] * len(u_indices)
            for idx, c in zip(u_indices, coeffs):
                U[idx][col] += c
            for idx in v_indices:
                V[idx][col] += 1
            for idx in w_indices:
                W[idx][col] += 1
            col += 1

        # Helper for Strassen block
        def add_strassen_block(k_start):
            b00 = 0 * 7 + k_start
            b01 = 0 * 7 + k_start + 1
            b10 = 1 * 7 + k_start
            b11 = 1 * 7 + k_start + 1
            
            c00 = k_start * 2 + 0
            c01 = (k_start + 1) * 2 + 0
            c10 = k_start * 2 + 1
            c11 = (k_start + 1) * 2 + 1
            
            # M1: (A00+A11)(B00+B11) -> C00, C11
            add_term([0, 4], [b00, b11], [c00, c11])
            
            # M2: (A10+A11)B00 -> C10, -C11
            add_term([3, 4], [b00], [c10, c11])
            W[c11][col-1] = -1
            
            # M3: A00(B01-B11) -> C01, C11
            add_term([0], [b01, b11], [c01, c11])
            V[b11][col-1] = -1
            
            # M4: A11(B10-B00) -> C00, C10
            add_term([4], [b10, b00], [c00, c10])
            V[b00][col-1] = -1
            
            # M5: (A00+A01)B11 -> -C00, C01
            add_term([0, 1], [b11], [c00, c01])
            W[c00][col-1] = -1
            
            # M6: (A10-A00)(B00+B01) -> C11
            add_term([3, 0], [b00, b01], [c11])
            U[0][col-1] = -1
            
            # M7: (A01-A11)(B10+B11) -> C00
            add_term([1, 4], [b10, b11], [c00])
            U[4][col-1] = -1

        # Add 3 Strassen blocks
        add_strassen_block(0)
        add_strassen_block(2)
        add_strassen_block(4)
        
        # Block 4: k=6, 2x2x1 Schoolbook
        b06 = 0 * 7 + 6
        b16 = 1 * 7 + 6
        c06 = 6 * 2 + 0
        c16 = 6 * 2 + 1
        
        add_term([0], [b06], [c06])
        add_term([1], [b16], [c06])
        add_term([3], [b06], [c16])
        add_term([4], [b16], [c16])
        
        # Part 2: j=2
        for k in range(7):
            b2k = 2 * 7 + k
            c0k = k * 2 + 0
            c1k = k * 2 + 1
            
            add_term([2], [b2k], [c0k])
            add_term([5], [b2k], [c1k])
        
        return {"U": U, "V": V, "W": W}

    def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        """Verify that U, V, W form a valid exact decomposition."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # Vectorized check: tensor is T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
        # We want T[i,j,k] == 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
        """Faster vectorized verification."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # For each (i,j,k), compute sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
        # We can do this more efficiently
        for i in range(n):
            for j in range(m):
                u_row = U_arr[i*m+j, :]
                for k in range(p):
                    v_row = V_arr[j*p+k, :]
                    w_row = W_arr[k*n+i, :]
                    s = np.dot(u_row, v_row * w_row)
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def try_reduce_rank(U, V, W, n, m, p, target_rank, max_iter=2000, tol=1e-10):
        """
        Try to reduce the rank by solving a least-squares problem.
        We fix a subset of columns and solve for the rest, or use 
        iterative refinement to find a lower rank decomposition.
        """
        if time.time() > deadline - 10:
            return None
            
        nm = n * m
        mp = m * p
        pn = p * n
        current_rank = len(U[0])
        
        if target_rank >= current_rank:
            return None
        
        # Build the linear system: for each (i,j,k), we need
        # sum_{r=0}^{target_rank-1} U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
        # This is a trilinear system. We use a simple alternating minimization.
        
        # Initialize with the first target_rank columns of the current decomposition
        U_full = np.array(U, dtype=np.float64)
        V_full = np.array(V, dtype=np.float64)
        W_full = np.array(W, dtype=np.float64)
        U_new = U_full[:, :target_rank].copy()
        V_new = V_full[:, :target_rank].copy()
        W_new = W_full[:, :target_rank].copy()
        
        # Alternating minimization: fix two, solve for the third
        for iteration in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Solve for U: for each (i,j), we need
            # sum_r U[i,j,r] * (V[j,k,r] * W[k,i,r]) = 1 for all k
            # This is: U[i,j,:] * diag(V[j,k,:]*W[k,i,:]) ... no, it's
            # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
            # For fixed i,j: sum_r U[i,j,r] * (V[j,k,r] * W[k,i,r]) = 1
            # Let M_k = V[j,k,:] * W[k,i,:] (element-wise product)
            # Then we need: sum_r U[i,j,r] * M_k[r] = 1 for all k
            # This is a linear system: U[i,j,:]^T * M = 1
            # where M is a p x target_rank matrix with M[k,r] = V[j,k,r]*W[k,i,r]
            
            U_old = U_new.copy()
            for i in range(n):
                for j in range(m):
                    # Build M: p x target_rank
                    M = np.zeros((p, target_rank))
                    for k in range(p):
                        M[k, :] = V_new[j*p+k, :] * W_new[k*n+i, :]
                    # Solve: M^T * U[i,j,:]^T = ones(p)
                    # i.e., M.T @ U[i,j,:] = ones
                    try:
                        U_new[i*m+j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Solve for V: for each (j,k), we need
            # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i
            # For fixed j,k: sum_r V[j,k,r] * (U[i,j,r] * W[k,i,r]) = 1
            # Let N_i = U[i,j,:] * W[k,i,:]
            # Then: N_i[r] * V[j,k,r] summed over r = 1
            # V[j,k,:]^T * N = 1 where N is n x target_rank
            V_old = V_new.copy()
            for j in range(m):
                for k in range(p):
                    N = np.zeros((n, target_rank))
                    for i in range(n):
                        N[i, :] = U_new[i*m+j, :] * W_new[k*n+i, :]
                    try:
                        V_new[j*p+k, :] = np.linalg.lstsq(N.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Solve for W: for each (k,i), we need
            # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all j
            # For fixed k,i: sum_r W[k,i,r] * (U[i,j,r] * V[j,k,r]) = 1
            # Let P_j = U[i,j,:] * V[j,k,:]
            # W[k,i,:]^T * P = 1 where P is m x target_rank
            W_old = W_new.copy()
            for k in range(p):
                for i in range(n):
                    P = np.zeros((m, target_rank))
                    for j in range(m):
                        P[j, :] = U_new[i*m+j, :] * V_new[j*p+k, :]
                    try:
                        W_new[k*n+i, :] = np.linalg.lstsq(P.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check convergence
            if np.allclose(U_new, U_old, atol=1e-12) and np.allclose(V_new, V_old, atol=1e-12) and np.allclose(W_new, W_old, atol=1e-12):
                break
        
        # Verify the solution
        U_list = U_new.tolist()
        V_list = V_new.tolist()
        W_list = W_new.tolist()
        
        if verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-8):
            return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Start with the known rank-39 decomposition
    base_solution = build_strassen_decomposition()
    
    # Verify the base solution
    if not verify_decomposition(base_solution["U"], base_solution["V"], base_solution["W"], n, m, p):
        # Fallback to schoolbook if Strassen is broken
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
        return {"U": U, "V": V, "W": W}
    
    best_solution = base_solution
    best_rank = len(base_solution["U"][0])
    
    # Try to reduce the rank
    # The theoretical lower bound is around 2*3*7 - something
    # Let's try ranks from best_rank-1 down to a reasonable lower bound
    # Lower bound: for matrix multiplication, rank >= max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # But there are stronger bounds. Let's try a few values.
    
    lower_bound = max(n*m, m*p, p*n)  # 21
    
    # Try reducing by 1, 2, 3, 4, 5
    for delta in range(1, 6):
        target_rank = best_rank - delta
        if target_rank < lower_bound:
            break
        if time.time() > deadline - 20:
            break
            
        result = try_reduce_rank(base_solution["U"], base_solution["V"], base_solution["W"], 
                                 n, m, p, target_rank, max_iter=1000)
        if result is not None:
            best_solution = result
            best_rank = target_rank
            # Try to reduce further from the new solution
            # But we need to convert back to the base format for the next iteration
            # Actually, try_reduce_rank already returns the right format
            # Let's update base_solution for the next iteration
            # But we should be careful - the next iteration should start from the reduced solution
            # Actually, let's just try from the original base solution each time to avoid compounding errors
            pass
    
    return best_solution

# EVOLVE-BLOCK-END
