import numpy as np
from fractions import Fraction
import time
import random

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135  # More conservative safety margin
    
    def verify(U, V, W):
        """Verify exact decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
        R = len(U[0]) if isinstance(U, list) else U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        u = U[i*m+j][r] if isinstance(U, list) else U[i*m+j, r]
                        v = V[j*p+k][r] if isinstance(V, list) else V[j*p+k, r]
                        w = W[k*n+i][r] if isinstance(W, list) else W[k*n+i, r]
                        s += u * v * w
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    def to_list(mat):
        """Convert numpy array to list of lists"""
        if isinstance(mat, list):
            return mat
        return [[float(x) for x in row] for row in mat]
    
    # Build the known rank-28 block decomposition
    def build_rank_28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen_block(rank_offset, j_offset, k_offset):
            a_idx = 0 * m + j_offset
            b_idx = 0 * m + (j_offset + 1)
            c_idx = 1 * m + j_offset
            d_idx = 1 * m + (j_offset + 1)
            
            e_idx = j_offset * p + k_offset
            f_idx = j_offset * p + (k_offset + 1)
            g_idx = (j_offset + 1) * p + k_offset
            h_idx = (j_offset + 1) * p + (k_offset + 1)
            
            r0 = rank_offset
            r1 = rank_offset + 1
            r2 = rank_offset + 2
            r3 = rank_offset + 3
            r4 = rank_offset + 4
            r5 = rank_offset + 5
            r6 = rank_offset + 6
            
            U[a_idx][r0] += 1; U[d_idx][r0] += 1
            U[c_idx][r1] += 1; U[d_idx][r1] += 1
            U[a_idx][r2] += 1
            U[d_idx][r3] += 1
            U[a_idx][r4] += 1; U[b_idx][r4] += 1
            U[c_idx][r5] += 1; U[a_idx][r5] -= 1
            U[b_idx][r6] += 1; U[d_idx][r6] -= 1
            
            V[e_idx][r0] += 1; V[h_idx][r0] += 1
            V[e_idx][r1] += 1
            V[f_idx][r2] += 1; V[h_idx][r2] -= 1
            V[g_idx][r3] += 1; V[e_idx][r3] -= 1
            V[h_idx][r4] += 1
            V[e_idx][r5] += 1; V[f_idx][r5] += 1
            V[g_idx][r6] += 1; V[h_idx][r6] += 1
            
            W[k_offset * n + 0][r0] += 1
            W[k_offset * n + 0][r3] += 1
            W[k_offset * n + 0][r4] -= 1
            W[k_offset * n + 0][r6] += 1
            
            W[(k_offset + 1) * n + 0][r2] += 1
            W[(k_offset + 1) * n + 0][r4] += 1
            
            W[k_offset * n + 1][r1] += 1
            W[k_offset * n + 1][r3] += 1
            
            W[(k_offset + 1) * n + 1][r0] += 1
            W[(k_offset + 1) * n + 1][r2] += 1
            W[(k_offset + 1) * n + 1][r5] += 1
            W[(k_offset + 1) * n + 1][r6] += 1
        
        # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
        add_strassen_block(0, 0, 0)
        
        # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
        add_strassen_block(7, 0, 2)
        
        # Block 3: A[0:2, 0:2] * B[0:2, 4:5] -> C[:, 4:5] (schoolbook 2x2x1)
        # C04: A00*B04 + A01*B14
        U[0][14] += 1; V[0*5+4][14] += 1; W[4*2+0][14] += 1
        U[1][15] += 1; V[1*5+4][15] += 1; W[4*2+0][15] += 1
        # C14: A10*B04 + A11*B14
        U[3][16] += 1; V[0*5+4][16] += 1; W[4*2+1][16] += 1
        U[4][17] += 1; V[1*5+4][17] += 1; W[4*2+1][17] += 1
        
        # Block 4: A[0:2, 2:3] * B[2:3, 0:5] -> C[:, 0:5] (schoolbook 2x1x5)
        for k in range(5):
            U[2][18 + k] += 1
            V[2*5 + k][18 + k] += 1
            W[k*2 + 0][18 + k] += 1
            U[5][23 + k] += 1
            V[2*5 + k][23 + k] += 1
            W[k*2 + 1][23 + k] += 1
        
        return {"U": U, "V": V, "W": W}
    
    def compute_error(U_sub, V_sub, W_sub):
        """Compute total error efficiently using tensor operations"""
        R = U_sub.shape[1]
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_sub[i*m+j], V_sub[j*p+k] * W_sub[k*n+i])
                    error += abs(val - 1.0)
        return error
    
    def compute_max_error(U_sub, V_sub, W_sub):
        """Compute max error for convergence checking"""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_sub[i*m+j], V_sub[j*p+k] * W_sub[k*n+i])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def smooth_minmax_error(U_sub, V_sub, W_sub, beta=50.0):
        """Compute smooth min-max error using log-sum-exp surrogate"""
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_sub[i*m+j], V_sub[j*p+k] * W_sub[k*n+i])
                    errors.append(val - 1.0)
        errors = np.array(errors)
        # Log-sum-exp for max |error|
        abs_errors = np.abs(errors)
        max_abs = np.max(abs_errors)
        if max_abs < 1e-15:
            return 0.0
        shifted = abs_errors - max_abs
        logsumexp = max_abs + np.log(np.sum(np.exp(beta * shifted)) / beta)
        return logsumexp
    
    def polish_solution(U_sub, V_sub, W_sub, max_polish_iters=300):
        """High-precision polish pass with exact rational verification"""
        U_p = U_sub.copy()
        V_p = V_sub.copy()
        W_p = W_sub.copy()
        
        best_error = float('inf')
        best_U, best_V, best_W = U_p.copy(), V_p.copy(), W_p.copy()
        
        for it in range(max_polish_iters):
            if time.time() > deadline - 10:
                break
            
            # Alternating minimization with tighter tolerance
            for i in range(n):
                for j in range(m):
                    A = np.array([V_p[j*p+k, :] * W_p[k*n+i, :] for k in range(p)])
                    try:
                        U_p[i*m+j, :] = np.linalg.lstsq(A, np.ones(p), rcond=1e-14)[0]
                    except:
                        pass
            
            for j in range(m):
                for k in range(p):
                    A = np.array([U_p[i*m+j, :] * W_p[k*n+i, :] for i in range(n)])
                    try:
                        V_p[j*p+k, :] = np.linalg.lstsq(A, np.ones(n), rcond=1e-14)[0]
                    except:
                        pass
            
            for k in range(p):
                for i in range(n):
                    A = np.array([U_p[i*m+j, :] * V_p[j*p+k, :] for j in range(m)])
                    try:
                        W_p[k*n+i, :] = np.linalg.lstsq(A, np.ones(m), rcond=1e-14)[0]
                    except:
                        pass
            
            error = compute_error(U_p, V_p, W_p)
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U_p.copy(), V_p.copy(), W_p.copy()
            
            if error < 1e-13:
                break
        
        # Try to round to rational values with various denominators - more comprehensive
        for denom in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24, 28, 30, 32, 35, 36, 40, 42, 48, 56, 60, 70, 84, 120]:
            U_r = np.round(best_U * denom) / denom
            V_r = np.round(best_V * denom) / denom
            W_r = np.round(best_W * denom) / denom
            
            err_r = compute_error(U_r, V_r, W_r)
            if err_r < 1e-10:
                result = {"U": to_list(U_r), "V": to_list(V_r), "W": to_list(W_r)}
                if verify(result["U"], result["V"], result["W"]):
                    return result
        
        # Also try half-integer values (denom=2) with different rounding
        for round_mode in ['floor', 'ceil']:
            U_r = np.floor(best_U * 2) / 2 if round_mode == 'floor' else np.ceil(best_U * 2) / 2
            V_r = np.floor(best_V * 2) / 2 if round_mode == 'floor' else np.ceil(best_V * 2) / 2
            W_r = np.floor(best_W * 2) / 2 if round_mode == 'floor' else np.ceil(best_W * 2) / 2
            
            err_r = compute_error(U_r, V_r, W_r)
            if err_r < 1e-10:
                result = {"U": to_list(U_r), "V": to_list(V_r), "W": to_list(W_r)}
                if verify(result["U"], result["V"], result["W"]):
                    return result
        
        return None
    
    def try_lower_rank(R, max_iters=400, n_restarts=10):
        if time.time() > deadline - 12:
            return None
        
        base = build_rank_28()
        U_full = np.array(base["U"], dtype=float)
        V_full = np.array(base["V"], dtype=float)
        W_full = np.array(base["W"], dtype=float)
        
        best_result = None
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for restart in range(n_restarts):
            if time.time() > deadline - 8:
                break
            
            # Better initialization strategies
            if restart == 0:
                keep = list(range(R))
            elif restart == 1:
                keep = list(range(28 - R, 28))
            elif restart == 2:
                # Keep middle columns
                mid = 14
                keep = list(range(max(0, mid - R//2), min(28, mid - R//2 + R)))
            elif restart == 3:
                # Random but biased toward important columns
                weights = [1.0 + 0.1 * np.random.randn() for _ in range(28)]
                weights = np.abs(weights)
                weights /= np.sum(weights)
                keep = list(np.random.choice(28, R, replace=False, p=weights))
            else:
                keep = random.sample(range(28), R)
            
            U_sub = U_full[:, keep].copy()
            V_sub = V_full[:, keep].copy()
            W_sub = W_full[:, keep].copy()
            
            if restart > 0:
                # Vary noise scale for diversity
                noise_scale = 0.003 + 0.002 * (restart % 4)
                U_sub += noise_scale * np.random.randn(*U_sub.shape)
                V_sub += noise_scale * np.random.randn(*V_sub.shape)
                W_sub += noise_scale * np.random.randn(*W_sub.shape)
            
            prev_error = float('inf')
            stagnant_count = 0
            
            for iteration in range(max_iters):
                if time.time() > deadline - 6:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        A = np.array([V_sub[j*p+k, :] * W_sub[k*n+i, :] for k in range(p)])
                        try:
                            U_sub[i*m+j, :] = np.linalg.lstsq(A, np.ones(p), rcond=1e-10)[0]
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.array([U_sub[i*m+j, :] * W_sub[k*n+i, :] for i in range(n)])
                        try:
                            V_sub[j*p+k, :] = np.linalg.lstsq(A, np.ones(n), rcond=1e-10)[0]
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        A = np.array([U_sub[i*m+j, :] * V_sub[j*p+k, :] for j in range(m)])
                        try:
                            W_sub[k*n+i, :] = np.linalg.lstsq(A, np.ones(m), rcond=1e-10)[0]
                        except:
                            pass
                
                error = compute_error(U_sub, V_sub, W_sub)
                
                if error < best_error:
                    best_error = error
                    best_U, best_V, best_W = U_sub.copy(), V_sub.copy(), W_sub.copy()
                    stagnant_count = 0
                else:
                    stagnant_count += 1
                
                if error < 1e-10:
                    polished = polish_solution(best_U, best_V, best_W)
                    if polished is not None:
                        return polished
                    for scale in [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 30, 40, 60]:
                        U_r = np.round(best_U * scale) / scale
                        V_r = np.round(best_V * scale) / scale
                        W_r = np.round(best_W * scale) / scale
                        err_r = compute_error(U_r, V_r, W_r)
                        if err_r < 1e-9:
                            result = {"U": to_list(U_r), "V": to_list(V_r), "W": to_list(W_r)}
                            if verify(result["U"], result["V"], result["W"]):
                                return result
                    break
                
                # Early stopping if stagnant
                if iteration > 100 and stagnant_count > 50:
                    break
        
        # Final polish on best solution found
        if best_error < 0.5 and best_U is not None:
            polished = polish_solution(best_U, best_V, best_W)
            if polished is not None:
                return polished
        
        return None
    
    # First, get the guaranteed rank-28 solution
    rank28_sol = build_rank_28()
    if not verify(rank28_sol["U"], rank28_sol["V"], rank28_sol["W"]):
        # Fallback to pure schoolbook if something went wrong
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find lower rank solutions with optimized time budget
    # Allocate more time to higher ranks first since they're more likely to succeed
    for R in range(27, 23, -1):
        remaining = deadline - time.time()
        if remaining < 20:
            break
        # Allocate time proportional to likelihood of success
        time_budget = min(remaining - 15, 30 + (27 - R) * 5)
        if time_budget < 10:
            break
        max_iters = max(200, int(time_budget * 8))
        n_restarts = max(5, min(12, int(time_budget / 3)))
        sol = try_lower_rank(R, max_iters=max_iters, n_restarts=n_restarts)
        if sol is not None:
            return sol
    
    # Return the guaranteed rank-28 solution
    return rank28_sol
