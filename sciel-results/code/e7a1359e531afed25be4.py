import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # Safety margin

    def check_solution_full(U, V, W):
        """Check if the full 2x3x5 decomposition is exact."""
        # U: (6, R), V: (15, R), W: (10, R)
        # Indices: U[i*m+j], V[j*p+k], W[k*n+i]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def build_strassen_rank28():
        """Build explicit rank-28 solution."""
        rank = 28
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        c = 0

        def add_strassen_block(j_off, k_off):
            nonlocal c
            # m1
            U[0*m + j_off, c] = 1; U[1*m + (j_off+1), c] = 1
            V[j_off*p + k_off, c] = 1; V[(j_off+1)*p + (k_off+1), c] = 1
            W[k_off*n + 0, c] = 1; W[(k_off+1)*n + 1, c] = 1
            c += 1
            # m2
            U[1*m + j_off, c] = 1; U[1*m + (j_off+1), c] = 1
            V[j_off*p + k_off, c] = 1
            W[k_off*n + 1, c] = 1; W[(k_off+1)*n + 1, c] = -1
            c += 1
            # m3
            U[0*m + j_off, c] = 1
            V[j_off*p + (k_off+1), c] = 1; V[(j_off+1)*p + (k_off+1), c] = -1
            W[(k_off+1)*n + 0, c] = 1; W[(k_off+1)*n + 1, c] = 1
            c += 1
            # m4
            U[1*m + (j_off+1), c] = 1
            V[(j_off+1)*p + k_off, c] = 1; V[j_off*p + k_off, c] = -1
            W[k_off*n + 0, c] = 1; W[k_off*n + 1, c] = 1
            c += 1
            # m5
            U[0*m + j_off, c] = 1; U[0*m + (j_off+1), c] = 1
            V[(j_off+1)*p + (k_off+1), c] = 1
            W[k_off*n + 0, c] = -1; W[(k_off+1)*n + 0, c] = 1
            c += 1
            # m6
            U[1*m + j_off, c] = 1; U[0*m + j_off, c] = -1
            V[j_off*p + k_off, c] = 1; V[j_off*p + (k_off+1), c] = 1
            W[(k_off+1)*n + 1, c] = 1
            c += 1
            # m7
            U[0*m + (j_off+1), c] = 1; U[1*m + (j_off+1), c] = -1
            V[(j_off+1)*p + k_off, c] = 1; V[(j_off+1)*p + (k_off+1), c] = 1
            W[k_off*n + 0, c] = 1
            c += 1

        # Block 1: j in {0,1}, k in {0,1}
        add_strassen_block(0, 0)
        # Block 2: j in {0,1}, k in {2,3}
        add_strassen_block(0, 2)
        
        # Remaining 14 schoolbook terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if j in (0, 1) and k in (0, 1, 2, 3):
                        continue
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        return U, V, W, rank

    # Step 1: Get guaranteed rank-28 solution
    U28, V28, W28, rank28 = build_strassen_rank28()
    err28 = check_solution_full(U28, V28, W28)
    
    if err28 > 1e-10:
        # Fallback to schoolbook rank 30
        R = n*m*p
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = rank28
    best_U, best_V, best_W = U28, V28, W28

    # Step 2: Attempt rank-27 by solving M_{2,2,5} rank 17
    if time.time() < deadline - 5:
        R_target = 27
        n_sub, m_sub, p_sub = 2, 2, 5
        R_sub = 17
        
        def solve_m225_rank17():
            """Try to find rank-17 decomposition for M_{2,2,5} using vectorized ALS."""
            max_trials = 200
            max_iters = 100
            tol = 1e-10
            
            # Precompute indices for vectorization
            # U shape: (2, 2, 17)
            # V shape: (2, 5, 17)
            # W shape: (5, 2, 17)
            
            for trial in range(max_trials):
                if time.time() > deadline - 2:
                    break
                np.random.seed(trial * 13 + 42)
                
                # Initialization: Random
                U_s = np.random.randn(n_sub, m_sub, R_sub) * 0.5
                V_s = np.random.randn(m_sub, p_sub, R_sub) * 0.5
                W_s = np.random.randn(p_sub, n_sub, R_sub) * 0.5
                
                for it in range(max_iters):
                    if time.time() > deadline - 1:
                        break
                    
                    # Update U
                    # For each i, j: solve min_u sum_k (sum_r u_r V[j,k,r] W[k,i,r] - 1)^2
                    # A[i,j,k,r] = V[j,k,r] * W[k,i,r]
                    # We need to solve (A^T A) u = A^T 1 for all i,j
                    # A^T A has shape (2, 2, 17, 17)
                    # A^T 1 has shape (2, 2, 17)
                    
                    # Compute A^T A and A^T 1
                    # V[j,k,r] W[k,i,r]
                    # Sum over k
                    # AT_A[i,j,r1,r2] = sum_k V[j,k,r1] W[k,i,r1] * V[j,k,r2] W[k,i,r2]
                    # This can be done with einsum
                    # V[j,k,r1] W[k,i,r1] -> sum_k -> Temp[j,i,r1,r2]? No, r1, r2 are separate.
                    # Let's use broadcasting.
                    # V: (2, 5, 17)
                    # W: (5, 2, 17)
                    # We want to compute for each i, j:
                    # M[i, j, r1, r2] = sum_k V[j, k, r1] * W[k, i, r1] * V[j, k, r2] * W[k, i, r2]
                    # b[i, j, r1] = sum_k V[j, k, r1] * W[k, i, r1]
                    
                    # This is expensive to compute from scratch every iteration.
                    # But R=17 is small.
                    
                    # Vectorized update for U
                    # V_exp: (1, 2, 5, 17) -> (2, 2, 5, 17)
                    # W_exp: (5, 2, 17) -> (2, 2, 5, 17)
                    # Actually, let's just loop over i, j (4 iterations) and solve.
                    # 4 lstsq calls of size (5, 17) is very fast.
                    
                    for i in range(n_sub):
                        for j in range(m_sub):
                            A = np.zeros((p_sub, R_sub))
                            for k in range(p_sub):
                                A[k, :] = V_s[j, k, :] * W_s[k, i, :]
                            b = np.ones(p_sub)
                            try:
                                U_s[i, j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                            except:
                                pass

                    # Update V
                    for j in range(m_sub):
                        for k in range(p_sub):
                            A = np.zeros((n_sub, R_sub))
                            for i in range(n_sub):
                                A[i, :] = U_s[i, j, :] * W_s[k, i, :]
                            b = np.ones(n_sub)
                            try:
                                V_s[j, k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                            except:
                                pass

                    # Update W
                    for k in range(p_sub):
                        for i in range(n_sub):
                            A = np.zeros((m_sub, R_sub))
                            for j in range(m_sub):
                                A[j, :] = U_s[i, j, :] * V_s[j, k, :]
                            b = np.ones(m_sub)
                            try:
                                W_s[k, i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                            except:
                                pass

                    # Check error
                    err = 0.0
                    for i in range(n_sub):
                        for j in range(m_sub):
                            for k in range(p_sub):
                                val = np.dot(U_s[i, j, :], V_s[j, k, :] * W_s[k, i, :])
                                err = max(err, abs(val - 1.0))
                    if err < tol:
                        return U_s, V_s, W_s
            return None

        sol_m225 = solve_m225_rank17()
        if sol_m225 is not None:
            U_sub, V_sub, W_sub = sol_m225
            # Construct full 2x3x5 solution
            U_full = np.zeros((n * m, R_target))
            V_full = np.zeros((m * p, R_target))
            W_full = np.zeros((p * n, R_target))
            
            # Copy M_{2,2,5} part (columns 0..16)
            for i in range(2):
                for j in range(2):
                    U_full[i*m + j, :17] = U_sub[i, j, :]
            for j in range(2):
                for k in range(5):
                    V_full[j*p + k, :17] = V_sub[j, k, :]
            for k in range(5):
                for i in range(2):
                    W_full[k*n + i, :17] = W_sub[k, i, :]
            
            # Add M_{2,1,5} part (columns 17..26)
            col = 17
            for i in range(2):
                for k in range(5):
                    U_full[i*m + 2, col] = 1
                    V_full[2*p + k, col] = 1
                    W_full[k*n + i, col] = 1
                    col += 1
            
            # Verify full solution
            err_full = check_solution_full(U_full, V_full, W_full)
            if err_full < 1e-8:
                best_rank = R_target
                best_U, best_V, best_W = U_full, V_full, W_full

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
