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
    
    # Verify a decomposition
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

    # Rank-55 decomposition: j=0 separately + Strassen for j=1,2
    def rank55_decomposition():
        rank = 55
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        
        # Part A: j=0 (first column of A, first row of B)
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
    
    # Corrected ALS
    def als(U, V, W, r, max_iters=100, tol=1e-12):
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Solve for U
            U_new = np.zeros_like(U)
            for i in range(n):
                # M[j*p+k, c] = V[j*p+k, c] * W[k*n+i, c]
                # We need to form a (m*p) x r matrix
                # Row index in M corresponds to (j, k)
                # M_val[j, k, c] = V[j*p+k, c] * W[k*n+i, c]
                # Flatten (j, k) to row index j*p+k
                M_mat = np.zeros((m*p, r))
                for j in range(m):
                    for k in range(p):
                        M_mat[j*p+k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve M_mat @ x = ones(m*p)
                try:
                    sol = np.linalg.lstsq(M_mat, np.ones(m*p), rcond=None)[0]
                    U_new[i*m:(i+1)*m, :] = sol
                except:
                    break
            U = U_new
            
            # Solve for V
            V_new = np.zeros_like(V)
            for j in range(m):
                # N[i*n+k, c] = U[i*m+j, c] * W[k*n+i, c]
                # Row index in N corresponds to (i, k)
                # N_val[i, k, c] = U[i*m+j, c] * W[k*n+i, c]
                # Flatten (i, k) to row index i*p+k ? No, V index is j*p+k.
                # Wait, V has shape (m*p, r). Index is j*p+k.
                # For a fixed j, we vary k.
                # The equation is sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
                # For fixed j, unknowns are V[j*p+k, :] for all k.
                # This is a coupled system?
                # Let x_k = V[j*p+k, :].
                # sum_r U_i V_k W_k = 1.
                # This is not a simple linear system for V rows independently because
                # the term is a product of three vectors.
                # Wait, if U and W are fixed, the equation for V[j*p+k, :] is:
                # sum_r V[j*p+k, r] (U[i*m+j, r] W[k*n+i, r]) = 1
                # This is linear in V[j*p+k, :].
                # For a fixed j, we have p equations (one for each k) for the p rows of V?
                # No, for each k, we have a vector equation?
                # No, for each i, k, we have a scalar equation.
                # For fixed j, we have n*p scalar equations.
                # Unknowns: V[j*p+0, :], ..., V[j*p+p-1, :] (p vectors of length r).
                # Total unknowns: p*r.
                # Equations: n*p.
                # We can form a matrix N of size (n*p) x (p*r).
                # Row (i, k) corresponds to equation for T[i, j, k].
                # Column (k', c) corresponds to V[j*p+k', c].
                # N[i, k, k', c] = U[i*m+j, c] * W[k*n+i, c] if k' == k, else 0?
                # No, the term is sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
                # The variable is V[j*p+k, r].
                # So for a fixed k, the coefficient of V[j*p+k, r] is U[i*m+j, r] * W[k*n+i, r].
                # The coefficient of V[j*p+k', r] for k' != k is 0.
                # So the matrix is block diagonal with respect to k?
                # Yes! For a fixed j, the equations for different k are independent?
                # Let's check.
                # Equation for (i, k): sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1.
                # This equation only involves V[j*p+k, :].
                # It does NOT involve V[j*p+k', :] for k' != k.
                # So we can solve for each V[j*p+k, :] independently!
                # For fixed j, k:
                # sum_r V[j*p+k, r] (U[i*m+j, r] W[k*n+i, r]) = 1 for all i.
                # This is a system of n equations for the r unknowns V[j*p+k, 0..r-1].
                # Let N_vec[i, c] = U[i*m+j, c] * W[k*n+i, c].
                # Solve N_vec @ x = ones(n).
                
                for k in range(p):
                    N_sub = np.array([[U[i*m+j, c] * W[k*n+i, c] for c in range(r)] for i in range(n)])
                    try:
                        sol = np.linalg.lstsq(N_sub, np.ones(n), rcond=None)[0]
                        V_new[j*p+k, :] = sol
                    except:
                        break
            V = V_new
            
            # Solve for W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    # Equation: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all j.
                    # Unknown: W[k*n+i, :].
                    # Coeff: U[i*m+j, r] * V[j*p+k, r].
                    # System of m equations for r unknowns.
                    P_sub = np.array([[U[i*m+j, c] * V[j*p+k, c] for c in range(r)] for j in range(m)])
                    try:
                        sol = np.linalg.lstsq(P_sub, np.ones(m), rcond=None)[0]
                        W_new[k*n+i, :] = sol
                    except:
                        break
            W = W_new
            
            # Check convergence
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        err = abs(val - 1.0)
                        if err > max_err:
                            max_err = err
            if max_err < tol:
                break
        
        return U, V, W

    # Try to reduce rank using corrected ALS
    if time.time() < deadline - 10:
        # Try ranks from 54 down to some minimum
        for r in range(rank_best - 1, max(nm, mp, pn) - 1, -1):
            if time.time() > deadline - 5:
                break
            
            found = False
            max_restarts = 20 if r >= 50 else 10
            
            for trial in range(max_restarts):
                if time.time() > deadline - 5:
                    break
                
                np.random.seed(trial * 1000 + r)
                
                # Initialize near the rank-55 solution
                if r <= rank_best:
                    U_c = U_best[:, :r].copy() + 0.1 * np.random.randn(nm, r)
                    V_c = V_best[:, :r].copy() + 0.1 * np.random.randn(mp, r)
                    W_c = W_best[:, :r].copy() + 0.1 * np.random.randn(pn, r)
                else:
                    U_c = np.random.randn(nm, r) * 0.5
                    V_c = np.random.randn(mp, r) * 0.5
                    W_c = np.random.randn(pn, r) * 0.5
                
                # Run corrected ALS
                U_c, V_c, W_c = als(U_c, V_c, W_c, r, max_iters=50, tol=1e-12)
                
                # Check convergence
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_c[i*m+j, :], V_c[j*p+k, :] * W_c[k*n+i, :])
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                
                if max_err < 1e-8:
                    # Try to round to half-integers
                    U_r = np.round(U_c * 2) / 2
                    V_r = np.round(V_c * 2) / 2
                    W_r = np.round(W_c * 2) / 2
                    
                    max_err2 = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U_r[i*m+j, :], V_r[j*p+k, :] * W_r[k*n+i, :])
                                err = abs(val - 1.0)
                                if err > max_err2:
                                    max_err2 = err
                    
                    if max_err2 < 1e-10:
                        U_best, V_best, W_best, rank_best = U_r, V_r, W_r, r
                        found = True
                        break
                    else:
                        # Use unrounded if close enough (though task requires int/half-int)
                        # If unrounded is exact, we might still fail the "integer/half-integer" check
                        # unless we can snap it.
                        pass
            
            if found:
                break
    
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
    
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
