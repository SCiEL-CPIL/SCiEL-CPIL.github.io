import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    n_rows = n * m  # 6
    m_rows = m * p  # 27
    p_rows = p * n  # 18
    
    def verify_decomposition(U, V, W):
        """Verify that the tensor decomposition is exact."""
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(r):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if val != 1:
                        return False
        return True
    
    # Rank 50 decomposition (Strassen-based) - guaranteed fallback
    def current_solution():
        rank = 50
        U = [[0] * rank for _ in range(n_rows)]
        V = [[0] * rank for _ in range(m_rows)]
        W = [[0] * rank for _ in range(p_rows)]
        
        col = 0
        for b in range(4):
            k0 = 2 * b
            k1 = 2 * b + 1
            v_e = 0 * p + k0
            v_f = 0 * p + k1
            v_g = 1 * p + k0
            v_h = 1 * p + k1
            u_a = 0 * m + 0
            u_b = 0 * m + 1
            u_c = 1 * m + 0
            u_d = 1 * m + 1
            
            U[u_a][col] += 1; U[u_d][col] += 1; V[v_e][col] += 1; V[v_h][col] += 1
            U[u_c][col+1] += 1; U[u_d][col+1] += 1; V[v_e][col+1] += 1
            U[u_a][col+2] += 1; V[v_f][col+2] += 1; V[v_h][col+2] -= 1
            U[u_d][col+3] += 1; V[v_g][col+3] += 1; V[v_e][col+3] -= 1
            U[u_a][col+4] += 1; U[u_b][col+4] += 1; V[v_h][col+4] += 1
            U[u_c][col+5] += 1; U[u_a][col+5] -= 1; V[v_e][col+5] += 1; V[v_f][col+5] += 1
            U[u_b][col+6] += 1; U[u_d][col+6] -= 1; V[v_g][col+6] += 1; V[v_h][col+6] += 1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            W[w_00][col] += 1; W[w_00][col+3] += 1; W[w_00][col+4] -= 1; W[w_00][col+6] += 1
            W[w_01][col+2] += 1; W[w_01][col+4] += 1
            W[w_10][col+1] += 1; W[w_10][col+3] += 1
            W[w_11][col] += 1; W[w_11][col+1] -= 1; W[w_11][col+2] += 1; W[w_11][col+5] += 1
            col += 7
            
        k8 = 8
        v_e = 0 * p + k8
        v_g = 1 * p + k8
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
        U[u_a][col] += 1; V[v_e][col] += 1
        U[u_b][col+1] += 1; V[v_g][col+1] += 1
        U[u_c][col+2] += 1; V[v_e][col+2] += 1
        U[u_d][col+3] += 1; V[v_g][col+3] += 1
        
        w_08 = k8 * n + 0
        w_18 = k8 * n + 1
        W[w_08][col] += 1; W[w_08][col+1] += 1
        W[w_18][col+2] += 1; W[w_18][col+3] += 1
        col += 4
        
        u_ar0 = 0 * m + 2
        u_ar1 = 1 * m + 2
        for k in range(9):
            v_b = 2 * p + k
            U[u_ar0][col] += 1; V[v_b][col] += 1; W[k * n + 0][col] += 1
            U[u_ar1][col+1] += 1; V[v_b][col+1] += 1; W[k * n + 1][col+1] += 1
            col += 2
            
        return {"U": U, "V": V, "W": W}
    
    # Strategy: Use structured W patterns derived from Strassen blocks
    # Key insight: W columns should have specific patterns that allow linear solving
    
    def solve_for_U_given_VW(W, V, r):
        """Solve for U given fixed W and V using least squares."""
        U = np.zeros((n_rows, r), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                # For each k, we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    sol, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U[i*m+j, :] = sol
                except:
                    pass
        return U
    
    def solve_for_V_given_UW(W, U, r):
        """Solve for V given fixed W and U using least squares."""
        V = np.zeros((m_rows, r), dtype=np.float64)
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    sol, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    V[j*p+k, :] = sol
                except:
                    pass
        return V
    
    def compute_error(U, V, W):
        """Compute max error of the decomposition."""
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(len(U[0])):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    max_err = max(max_err, abs(val - 1))
        return max_err
    
    def try_rank_with_structured_W(r, max_iters=100, max_trials=20):
        """Try to find rank-r decomposition using structured W patterns."""
        if time.time() > deadline - 20:
            return None
        
        # Create structured W patterns based on Strassen-like structure
        # Each W column should have a pattern that covers specific (k,i) pairs
        w_patterns = []
        
        # Pattern 1: Single (k,i) pair
        for k in range(p):
            for i in range(n):
                pat = np.zeros(p_rows)
                pat[k*n+i] = 1
                w_patterns.append(pat)
        
        # Pattern 2: Two (k,i) pairs with same i
        for i in range(n):
            for k1 in range(p):
                for k2 in range(k1+1, p):
                    pat = np.zeros(p_rows)
                    pat[k1*n+i] = 1
                    pat[k2*n+i] = 1
                    w_patterns.append(pat)
        
        # Pattern 3: Strassen-like patterns for 2x2 blocks
        for b in range(4):
            k0 = 2*b
            k1 = 2*b+1
            # (k0,i0), (k1,i1)
            pat = np.zeros(p_rows)
            pat[k0*n+0] = 1
            pat[k1*n+1] = 1
            w_patterns.append(pat)
            # (k0,i1), (k1,i0)
            pat = np.zeros(p_rows)
            pat[k0*n+1] = 1
            pat[k1*n+0] = 1
            w_patterns.append(pat)
            # (k0,i0), (k0,i1)
            pat = np.zeros(p_rows)
            pat[k0*n+0] = 1
            pat[k0*n+1] = 1
            w_patterns.append(pat)
            # (k1,i0), (k1,i1)
            pat = np.zeros(p_rows)
            pat[k1*n+0] = 1
            pat[k1*n+1] = 1
            w_patterns.append(pat)
        
        # Shuffle patterns
        np.random.shuffle(w_patterns)
        
        best_result = None
        best_error = float('inf')
        
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                break
            
            # Select r W patterns
            selected_patterns = w_patterns[:r]
            W = np.zeros((p_rows, r), dtype=np.float64)
            for c in range(r):
                W[:, c] = selected_patterns[c]
            
            # Initialize U and V
            U = np.random.randn(n_rows, r) * 0.3
            V = np.random.randn(m_rows, r) * 0.3
            
            # Alternating least squares with refinement
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Solve for U
                U = solve_for_U_given_VW(W, V, r)
                # Solve for V
                V = solve_for_V_given_UW(W, U, r)
                
                # Compute error
                error = compute_error(U, V, W)
                if error < best_error:
                    best_error = error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if error < 1e-12:
                    break
            
            # Try to refine to exact solution
            if best_error < 1e-8:
                # Round to nearest 0.5
                U_r = np.round(best_U * 2) / 2
                V_r = np.round(best_V * 2) / 2
                W_r = np.round(best_W * 2) / 2
                
                if verify_decomposition(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
        
        return None
    
    # Try to find lower rank using structured approach
    for r in range(49, 35, -1):
        if time.time() > deadline - 30:
            break
        res = try_rank_with_structured_W(r, max_iters=80, max_trials=15)
        if res:
            return res
    
    # Fallback to rank 50
    return current_solution()
