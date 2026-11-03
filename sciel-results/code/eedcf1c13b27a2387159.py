import numpy as np
from fractions import Fraction
import time
import random

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
    
    def compute_error(U_sub, V_sub, W_sub):
        """Compute total error efficiently using tensor operations"""
        # Build the tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        R = U_sub.shape[1]
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_sub[i*m+j], V_sub[j*p+k] * W_sub[k*n+i])
                    error += abs(val - 1.0)
        return error
    
    def try_lower_rank(R, max_iters=500, n_restarts=3):
        if time.time() > deadline - 10:
            return None
        
        base = build_rank_28()
        U_full = np.array(base["U"], dtype=float)
        V_full = np.array(base["V"], dtype=float)
        W_full = np.array(base["W"], dtype=float)
        
        best_result = None
        best_error = float('inf')
        
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            if restart == 0:
                # First restart: keep first R columns
                keep = list(range(R))
            else:
                # Random selection of R columns
                keep = random.sample(range(28), R)
            
            U_sub = U_full[:, keep].copy()
            V_sub = V_full[:, keep].copy()
            W_sub = W_full[:, keep].copy()
            
            # Add small random perturbation to escape local minima
            if restart > 0:
                noise_scale = 0.01
                U_sub += noise_scale * np.random.randn(*U_sub.shape)
                V_sub += noise_scale * np.random.randn(*V_sub.shape)
                W_sub += noise_scale * np.random.randn(*W_sub.shape)
            
            for iteration in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Update U: for each (i,j), solve A @ U[i,j] = 1 where A[k,:] = V[j,k,:] * W[k,i,:]
                for i in range(n):
                    for j in range(m):
                        A = np.array([V_sub[j*p+k, :] * W_sub[k*n+i, :] for k in range(p)])
                        if A.shape[0] > 0:
                            try:
                                U_sub[i*m+j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                            except:
                                pass
                
                # Update V: for each (j,k), solve A @ V[j,k] = 1 where A[i,:] = U[i,j,:] * W[k,i,:]
                for j in range(m):
                    for k in range(p):
                        A = np.array([U_sub[i*m+j, :] * W_sub[k*n+i, :] for i in range(n)])
                        if A.shape[0] > 0:
                            try:
                                V_sub[j*p+k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                            except:
                                pass
                
                # Update W: for each (k,i), solve A @ W[k,i] = 1 where A[j,:] = U[i,j,:] * V[j,k,:]
                for k in range(p):
                    for i in range(n):
                        A = np.array([U_sub[i*m+j, :] * V_sub[j*p+k, :] for j in range(m)])
                        if A.shape[0] > 0:
                            try:
                                W_sub[k*n+i, :] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                            except:
                                pass
                
                # Check error
                error = compute_error(U_sub, V_sub, W_sub)
                
                if error < best_error:
                    best_error = error
                    best_U, best_V, best_W = U_sub.copy(), V_sub.copy(), W_sub.copy()
                
                if error < 1e-10:
                    # Try to round to half-integers using multiple scales
                    for scale in [2, 4, 8, 16, 32]:
                        U_r = np.round(best_U * scale) / scale
                        V_r = np.round(best_V * scale) / scale
                        W_r = np.round(best_W * scale) / scale
                        
                        err_r = compute_error(U_r, V_r, W_r)
                        
                        if err_r < 1e-9:
                            result = {"U": to_list(U_r), "V": to_list(V_r), "W": to_list(W_r)}
                            if verify(result["U"], result["V"], result["W"]):
                                return result
                    break
                
                # Early stopping if error is not improving
                if iteration > 50 and error > best_error * 1.1:
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
    
    # Try to find lower rank solutions with increased time budget per rank
    for R in range(27, 22, -1):
        if time.time() > deadline - 15:
            break
        sol = try_lower_rank(R, max_iters=300, n_restarts=4)
        if sol is not None:
            return sol
    
    # Return the guaranteed rank-28 solution
    return rank28_sol
