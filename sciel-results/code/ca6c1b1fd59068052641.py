import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor: T[i,j,k] = 1 for all valid (i,j,k)
    T = np.ones((n, m, p), dtype=float)
    
    def verify(U, V, W, r, tol=1e-10):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def rank55_decomposition():
        rank = 55
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        
        # Part A: j=0 (first column of A, first row of B)
        # This part is an outer product of A[:,0] and B[0,:]. Rank 20.
        for i in range(n):
            for k in range(p):
                U[i*m + 0, col] = 1
                V[0*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
        
        # Part B: j=1,2 with Strassen on 5 blocks of 2x2
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            u_11 = 0 * m + 1
            u_12 = 0 * m + 2
            u_21 = 1 * m + 1
            u_22 = 1 * m + 2
            
            v_11 = 1 * p + k0
            v_12 = 1 * p + k1
            v_21 = 2 * p + k0
            v_22 = 2 * p + k1
            
            # M1 = (a11 + a22)(b11 + b22)
            U[u_11, col] += 1
            U[u_22, col] += 1
            V[v_11, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += 1
            W[w_11, col] += 1
            col += 1
            
            # M2 = (a21 + a22)b11
            U[u_21, col] += 1
            U[u_22, col] += 1
            V[v_11, col] += 1
            W[w_10, col] += 1
            W[w_11, col] += -1
            col += 1
            
            # M3 = a11(b12 - b22)
            U[u_11, col] += 1
            V[v_12, col] += 1
            V[v_22, col] += -1
            W[w_01, col] += 1
            W[w_11, col] += 1
            col += 1
            
            # M4 = a22(b21 - b11)
            U[u_22, col] += 1
            V[v_21, col] += 1
            V[v_11, col] += -1
            W[w_00, col] += 1
            W[w_10, col] += 1
            col += 1
            
            # M5 = (a11 + a12)b22
            U[u_11, col] += 1
            U[u_12, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += -1
            W[w_01, col] += 1
            col += 1
            
            # M6 = (a21 - a11)(b11 + b12)
            U[u_21, col] += 1
            U[u_11, col] += -1
            V[v_11, col] += 1
            V[v_12, col] += 1
            W[w_11, col] += 1
            col += 1
            
            # M7 = (a12 - a22)(b21 + b22)
            U[u_12, col] += 1
            U[u_22, col] += -1
            V[v_21, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += 1
            col += 1
        
        return U, V, W, rank
    
    U_best, V_best, W_best, rank_best = rank55_decomposition()
    err = verify(U_best, V_best, W_best, rank_best)
    
    # Attempt to find Rank 54 by solving the 2x2x10 sub-problem with rank 34
    # The j=0 part is fixed at rank 20.
    # We need a rank 34 decomposition for the j=1,2 part.
    
    def solve_subproblem_rank34():
        """
        Tries to find a rank 34 decomposition for 2x2x10 matrix multiplication.
        Returns U_sub, V_sub, W_sub if found, else None.
        """
        n_sub = 2
        m_sub = 2
        p_sub = 10
        r_sub = 34
        
        # Target tensor for sub-problem: T[i,j,k] = 1
        # U shape: (n_sub * m_sub, r_sub) = (4, 34)
        # V shape: (m_sub * p_sub, r_sub) = (20, 34)
        # W shape: (p_sub * n_sub, r_sub) = (20, 34)
        
        def verify_sub(U_s, V_s, W_s):
            max_err = 0.0
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        val = np.dot(U_s[i*m_sub + j, :], V_s[j*p_sub + k, :] * W_s[k*n_sub + i, :])
                        err = abs(val - 1.0)
                        if err > max_err:
                            max_err = err
            return max_err

        # Initialize with random values
        # Use multiple restarts
        for trial in range(50):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(trial)
            U_s = np.random.randn(n_sub * m_sub, r_sub) * 0.5
            V_s = np.random.randn(m_sub * p_sub, r_sub) * 0.5
            W_s = np.random.randn(p_sub * n_sub, r_sub) * 0.5
            
            # ALS for sub-problem
            for it in range(100):
                if time.time() > deadline - 10:
                    break
                
                # Update U
                U_new = np.zeros_like(U_s)
                for i in range(n_sub):
                    for k in range(p_sub):
                        # M[j, c] = V_s[j*p_sub+k, c] * W_s[k*n_sub+i, c]
                        M = np.zeros((m_sub, r_sub))
                        for j in range(m_sub):
                            for c in range(r_sub):
                                M[j, c] = V_s[j*p_sub + k, c] * W_s[k*n_sub + i, c]
                        try:
                            sol = np.linalg.lstsq(M, np.ones(m_sub), rcond=None)[0]
                            for j in range(m_sub):
                                U_new[i*m_sub + j, :] = sol
                        except:
                            pass
                U_s = U_new
                
                # Update V
                V_new = np.zeros_like(V_s)
                for i in range(n_sub):
                    for k in range(p_sub):
                        N = np.zeros((m_sub, r_sub))
                        for j in range(m_sub):
                            for c in range(r_sub):
                                N[j, c] = U_s[i*m_sub + j, c] * W_s[k*n_sub + i, c]
                        try:
                            sol = np.linalg.lstsq(N, np.ones(m_sub), rcond=None)[0]
                            for j in range(m_sub):
                                V_new[j*p_sub + k, :] = sol
                        except:
                            pass
                V_s = V_new
                
                # Update W
                W_new = np.zeros_like(W_s)
                for i in range(n_sub):
                    for k in range(p_sub):
                        P = np.zeros((m_sub, r_sub))
                        for j in range(m_sub):
                            for c in range(r_sub):
                                P[j, c] = U_s[i*m_sub + j, c] * V_s[j*p_sub + k, c]
                        try:
                            sol = np.linalg.lstsq(P, np.ones(m_sub), rcond=None)[0]
                            for j in range(m_sub):
                                W_new[k*n_sub + i, :] = sol
                        except:
                            pass
                W_s = W_new
                
                err_sub = verify_sub(U_s, V_s, W_s)
                if err_sub < 1e-10:
                    return U_s, V_s, W_s
        
        return None

    # Try to find rank 54
    if time.time() < deadline - 20:
        sub_res = solve_subproblem_rank34()
        if sub_res is not None:
            U_s, V_s, W_s = sub_res
            # Construct full rank 54 solution
            rank_54 = 54
            U_54 = np.zeros((nm, rank_54))
            V_54 = np.zeros((mp, rank_54))
            W_54 = np.zeros((pn, rank_54))
            
            # Copy j=0 part (first 20 columns)
            # We need to regenerate the j=0 part or extract it from U_best
            # U_best has rank 55. First 20 cols are j=0.
            U_54[:, :20] = U_best[:, :20]
            V_54[:, :20] = V_best[:, :20]
            W_54[:, :20] = W_best[:, :20]
            
            # Copy j=1,2 part (next 34 columns)
            # Map sub-problem indices to full indices
            # Sub U: i in 0..1, j in 0..1 (local). Full j is 1,2.
            # Sub V: j in 0..1 (local), k in 0..9. Full j is 1,2.
            # Sub W: k in 0..9, i in 0..1.
            
            for i in range(2):
                for j_local in range(2):
                    j_full = j_local + 1
                    U_54[i*m + j_full, 20:20+34] = U_s[i*2 + j_local, :]
                    
            for j_local in range(2):
                j_full = j_local + 1
                for k in range(10):
                    V_54[j_full*p + k, 20:20+34] = V_s[j_local*10 + k, :]
                    
            for k in range(10):
                for i in range(2):
                    W_54[k*n + i, 20:20+34] = W_s[k*2 + i, :]
            
            err_54 = verify(U_54, V_54, W_54, rank_54)
            if err_54 < 1e-10:
                U_best, V_best, W_best, rank_best = U_54, V_54, W_54, rank_54
                print(f"Found Rank 54 solution!")

    # Final verification
    final_err = verify(U_best, V_best, W_best, rank_best)
    if final_err > 1e-8:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        U_best, V_best, W_best = U, V, W
        rank_best = rank

    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
