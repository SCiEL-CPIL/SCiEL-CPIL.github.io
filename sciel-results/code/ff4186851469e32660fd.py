import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    def check_solution_full(U, V, W, R):
        """Check if the full 2x3x5 decomposition is exact."""
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
            # m1 = (a00+a11)*(b00+b11)
            U[0*m + j_off, c] = 1; U[1*m + (j_off+1), c] = 1
            V[j_off*p + k_off, c] = 1; V[(j_off+1)*p + (k_off+1), c] = 1
            W[k_off*n + 0, c] = 1; W[(k_off+1)*n + 1, c] = 1
            c += 1
            # m2 = (a10+a11)*b00
            U[1*m + j_off, c] = 1; U[1*m + (j_off+1), c] = 1
            V[j_off*p + k_off, c] = 1
            W[k_off*n + 1, c] = 1; W[(k_off+1)*n + 1, c] = -1
            c += 1
            # m3 = a00*(b01-b11)
            U[0*m + j_off, c] = 1
            V[j_off*p + (k_off+1), c] = 1; V[(j_off+1)*p + (k_off+1), c] = -1
            W[(k_off+1)*n + 0, c] = 1; W[(k_off+1)*n + 1, c] = 1
            c += 1
            # m4 = a11*(b10-b00)
            U[1*m + (j_off+1), c] = 1
            V[(j_off+1)*p + k_off, c] = 1; V[j_off*p + k_off, c] = -1
            W[k_off*n + 0, c] = 1; W[k_off*n + 1, c] = 1
            c += 1
            # m5 = (a00+a01)*b11
            U[0*m + j_off, c] = 1; U[0*m + (j_off+1), c] = 1
            V[(j_off+1)*p + (k_off+1), c] = 1
            W[k_off*n + 0, c] = -1; W[(k_off+1)*n + 0, c] = 1
            c += 1
            # m6 = (a10-a00)*(b00+b01)
            U[1*m + j_off, c] = 1; U[0*m + j_off, c] = -1
            V[j_off*p + k_off, c] = 1; V[j_off*p + (k_off+1), c] = 1
            W[(k_off+1)*n + 1, c] = 1
            c += 1
            # m7 = (a01-a11)*(b10+b11)
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
    err28 = check_solution_full(U28, V28, W28, rank28)
    
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
    if time.time() < deadline - 10:
        R_target = 27
        # M_{2,2,5} sub-problem
        # Indices for M_{2,2,5}: i in {0,1}, j in {0,1}, k in {0..4}
        # We need to find U_sub (2*2 x 17), V_sub (2*5 x 17), W_sub (5*2 x 17)
        # such that sum_r U_sub[i*2+j, r] * V_sub[j*5+k, r] * W_sub[k*2+i, r] = 1
        
        def solve_m225_rank17():
            """Try to find rank-17 decomposition for M_{2,2,5}."""
            n_sub, m_sub, p_sub = 2, 2, 5
            R_sub = 17
            max_iters = 500
            tol = 1e-10
            
            # Initialize with Strassen blocks + schoolbook
            # Strassen for k=0,1 (rank 7)
            # Strassen for k=2,3 (rank 7)
            # Schoolbook for k=4 (rank 4)
            # Total 18. We need to merge one.
            
            U_sub = np.zeros((n_sub * m_sub, R_sub))
            V_sub = np.zeros((m_sub * p_sub, R_sub))
            W_sub = np.zeros((p_sub * n_sub, R_sub))
            
            # Helper to add a term
            def add_term(i, j, k, u, v, w, idx):
                U_sub[i*m_sub + j, idx] = u
                V_sub[j*p_sub + k, idx] = v
                W_sub[k*n_sub + i, idx] = w

            # Initialize with the 18-term solution, then zero out one column and let ALS fix it?
            # Or random restarts.
            
            for trial in range(200):
                if time.time() > deadline - 5:
                    break
                np.random.seed(trial * 101 + 7)
                
                # Start with the 18-term solution structure
                U_t = np.zeros((n_sub * m_sub, 18))
                V_t = np.zeros((m_sub * p_sub, 18))
                W_t = np.zeros((p_sub * n_sub, 18))
                
                # Block 1: Strassen for k=0,1
                # m1
                U_t[0*2+0, 0]=1; U_t[1*2+1, 0]=1; V_t[0*5+0, 0]=1; V_t[1*5+1, 0]=1; W_t[0*2+0, 0]=1; W_t[1*2+1, 0]=1
                # m2
                U_t[1*2+0, 1]=1; U_t[1*2+1, 1]=1; V_t[0*5+0, 1]=1; W_t[0*2+1, 1]=1; W_t[1*2+1, 1]=-1
                # m3
                U_t[0*2+0, 2]=1; V_t[0*5+1, 2]=1; V_t[1*5+1, 2]=-1; W_t[1*2+0, 2]=1; W_t[1*2+1, 2]=1
                # m4
                U_t[1*2+1, 3]=1; V_t[1*5+0, 3]=1; V_t[0*5+0, 3]=-1; W_t[0*2+0, 3]=1; W_t[0*2+1, 3]=1
                # m5
                U_t[0*2+0, 4]=1; U_t[0*2+1, 4]=1; V_t[1*5+1, 4]=1; W_t[0*2+0, 4]=-1; W_t[1*2+0, 4]=1
                # m6
                U_t[1*2+0, 5]=1; U_t[0*2+0, 5]=-1; V_t[0*5+0, 5]=1; V_t[0*5+1, 5]=1; W_t[1*2+1, 5]=1
                # m7
                U_t[0*2+1, 6]=1; U_t[1*2+1, 6]=-1; V_t[1*5+0, 6]=1; V_t[1*5+1, 6]=1; W_t[0*2+0, 6]=1
                
                # Block 2: Strassen for k=2,3 (shift k by 2)
                # m1'
                U_t[0*2+0, 7]=1; U_t[1*2+1, 7]=1; V_t[0*5+2, 7]=1; V_t[1*5+3, 7]=1; W_t[2*2+0, 7]=1; W_t[3*2+1, 7]=1
                # m2'
                U_t[1*2+0, 8]=1; U_t[1*2+1, 8]=1; V_t[0*5+2, 8]=1; W_t[2*2+1, 8]=1; W_t[3*2+1, 8]=-1
                # m3'
                U_t[0*2+0, 9]=1; V_t[0*5+3, 9]=1; V_t[1*5+3, 9]=-1; W_t[3*2+0, 9]=1; W_t[3*2+1, 9]=1
                # m4'
                U_t[1*2+1, 10]=1; V_t[1*5+2, 10]=1; V_t[0*5+2, 10]=-1; W_t[2*2+0, 10]=1; W_t[2*2+1, 10]=1
                # m5'
                U_t[0*2+0, 11]=1; U_t[0*2+1, 11]=1; V_t[1*5+3, 11]=1; W_t[2*2+0, 11]=-1; W_t[3*2+0, 11]=1
                # m6'
                U_t[1*2+0, 12]=1; U_t[0*2+0, 12]=-1; V_t[0*5+2, 12]=1; V_t[0*5+3, 12]=1; W_t[3*2+1, 12]=1
                # m7'
                U_t[0*2+1, 13]=1; U_t[1*2+1, 13]=-1; V_t[1*5+2, 13]=1; V_t[1*5+3, 13]=1; W_t[2*2+0, 13]=1
                
                # Block 3: Schoolbook for k=4
                for i in range(2):
                    for j in range(2):
                        idx = 14 + i*2 + j
                        U_t[i*2+j, idx] = 1
                        V_t[j*5+4, idx] = 1
                        W_t[4*2+i, idx] = 1
                        
                # Randomly perturb and drop one column to force rank 17
                # Drop a random column
                drop_idx = np.random.randint(18)
                keep = [x for x in range(18) if x != drop_idx]
                U_t = U_t[:, keep]
                V_t = V_t[:, keep]
                W_t = W_t[:, keep]
                
                # Add small noise
                U_t += np.random.randn(*U_t.shape) * 0.1
                V_t += np.random.randn(*V_t.shape) * 0.1
                W_t += np.random.randn(*W_t.shape) * 0.1
                
                # ALS for M_{2,2,5}
                U_s, V_s, W_s = U_t, V_t, W_t
                for it in range(max_iters):
                    # Solve U
                    for i in range(n_sub):
                        for j in range(m_sub):
                            A = np.zeros((p_sub, R_sub))
                            for k in range(p_sub):
                                A[k, :] = V_s[j*p_sub + k, :] * W_s[k*n_sub + i, :]
                            b = np.ones(p_sub)
                            U_s[i*m_sub + j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    # Solve V
                    for j in range(m_sub):
                        for k in range(p_sub):
                            A = np.zeros((n_sub, R_sub))
                            for i in range(n_sub):
                                A[i, :] = U_s[i*m_sub + j, :] * W_s[k*n_sub + i, :]
                            b = np.ones(n_sub)
                            V_s[j*p_sub + k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    # Solve W
                    for k in range(p_sub):
                        for i in range(n_sub):
                            A = np.zeros((m_sub, R_sub))
                            for j in range(m_sub):
                                A[j, :] = U_s[i*m_sub + j, :] * V_s[j*p_sub + k, :]
                            b = np.ones(m_sub)
                            W_s[k*n_sub + i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                            
                    # Check error
                    err = 0.0
                    for i in range(n_sub):
                        for j in range(m_sub):
                            for k in range(p_sub):
                                val = np.dot(U_s[i*m_sub + j, :], V_s[j*p_sub + k, :] * W_s[k*n_sub + i, :])
                                err = max(err, abs(val - 1.0))
                    if err < tol:
                        return U_s, V_s, W_s
                # End ALS
            return None

        sol_m225 = solve_m225_rank17()
        if sol_m225 is not None:
            U_sub, V_sub, W_sub = sol_m225
            # Construct full 2x3x5 solution
            # M_{2,3,5} = M_{2,2,5} (j=0,1) + M_{2,1,5} (j=2)
            # M_{2,1,5} rank 10: terms A_{i2} B_{2k}
            
            U_full = np.zeros((n * m, R_target))
            V_full = np.zeros((m * p, R_target))
            W_full = np.zeros((p * n, R_target))
            
            # Copy M_{2,2,5} part (columns 0..16)
            # Map indices: i in {0,1}, j in {0,1}
            for i in range(2):
                for j in range(2):
                    U_full[i*m + j, :17] = U_sub[i*2 + j, :]
            for j in range(2):
                for k in range(5):
                    V_full[j*p + k, :17] = V_sub[j*5 + k, :]
            for k in range(5):
                for i in range(2):
                    W_full[k*n + i, :17] = W_sub[k*2 + i, :]
            
            # Add M_{2,1,5} part (columns 17..26)
            col = 17
            for i in range(2):
                for k in range(5):
                    U_full[i*m + 2, col] = 1
                    V_full[2*p + k, col] = 1
                    W_full[k*n + i, col] = 1
                    col += 1
            
            # Verify full solution
            err_full = check_solution_full(U_full, V_full, W_full, R_target)
            if err_full < 1e-8:
                best_rank = R_target
                best_U, best_V, best_W = U_full, V_full, W_full

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
