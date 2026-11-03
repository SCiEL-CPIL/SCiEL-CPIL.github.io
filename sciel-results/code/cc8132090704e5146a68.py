import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # Safety margin before 150s kill
    
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
    
    # Try to find lower rank solutions using structured pruning from rank-28
    def try_lower_rank(R, max_iters=200):
        if time.time() > deadline - 10:
            return None
        
        # Start from rank-28 solution and try to zero out (28-R) columns
        base = build_rank_28()
        U = np.array(base["U"], dtype=float)
        V = np.array(base["V"], dtype=float)
        W = np.array(base["W"], dtype=float)
        
        # Try all subsets of columns to remove
        from itertools import combinations
        cols_to_remove = 28 - R
        found = False
        
        for remove_set in combinations(range(28), cols_to_remove):
            if time.time() > deadline - 5:
                break
            keep = [c for c in range(28) if c not in remove_set]
            U_sub = U[:, keep]
            V_sub = V[:, keep]
            W_sub = W[:, keep]
            
            # Check if this sub-decomposition is still valid (it won't be, but we use it as init)
            # Now do alternating least squares to see if we can recover exact solution
            best_error = float('inf')
            best_U, best_V, best_W = None, None, None
            
            for iteration in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, R))
                        for k in range(p):
                            A[k, :] = V_sub[j*p+k, :] * W_sub[k*n+i, :]
                        if np.linalg.matrix_rank(A) > 0:
                            try:
                                U_sub[i*m+j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                            except:
                                pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, R))
                        for i in range(n):
                            A[i, :] = U_sub[i*m+j, :] * W_sub[k*n+i, :]
                        if np.linalg.matrix_rank(A) > 0:
                            try:
                                V_sub[j*p+k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                            except:
                                pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, R))
                        for j in range(m):
                            A[j, :] = U_sub[i*m+j, :] * V_sub[j*p+k, :]
                        if np.linalg.matrix_rank(A) > 0:
                            try:
                                W_sub[k*n+i, :] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                            except:
                                pass
                
                # Check error
                error = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_sub[i*m+j, :] * V_sub[j*p+k, :] * W_sub[k*n+i, :])
                            error += abs(val - 1)
                
                if error < best_error:
                    best_error = error
                    best_U, best_V, best_W = U_sub.copy(), V_sub.copy(), W_sub.copy()
                
                if error < 1e-10:
                    # Try to round to half-integers
                    for scale in [2, 4, 8, 16]:
                        U_r = np.round(best_U * scale) / scale
                        V_r = np.round(best_V * scale) / scale
                        W_r = np.round(best_W * scale) / scale
                        
                        err_r = 0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    val = np.sum(U_r[i*m+j, :] * V_r[j*p+k, :] * W_r[k*n+i, :])
                                    err_r += abs(val - 1)
                        
                        if err_r < 1e-9:
                            result = {"U": to_list(U_r), "V": to_list(V_r), "W": to_list(W_r)}
                            if verify(result["U"], result["V"], result["W"]):
                                return result
                    break
        
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
    
    # Try to find lower rank solutions
    for R in range(27, 22, -1):
        if time.time() > deadline - 20:
            break
        sol = try_lower_rank(R)
        if sol is not None:
            return sol
    
    # Return the guaranteed rank-28 solution
    return rank28_sol
