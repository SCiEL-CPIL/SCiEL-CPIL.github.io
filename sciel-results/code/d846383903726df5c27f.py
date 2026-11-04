import numpy as np
import time
import sys
from scipy.linalg import null_space

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 100s margin
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Try to find lower rank
    # Start from 20 as it is a known lower bound/achievable rank for 3x3x4
    for R in range(20, rank_school):
        if time.time() > deadline:
            break
            
        # Try multiple random initializations
        for trial in range(50):
            if time.time() > deadline:
                break
                
            # Random init with small integers
            U = np.random.randint(-1, 2, (n * m, R))
            V = np.random.randint(-1, 2, (m * p, R))
            W = np.random.randint(-1, 2, (p * n, R))
            
            # Iterative refinement
            for it in range(100):
                if time.time() > deadline:
                    break
                
                # Solve for W
                W_new = np.zeros_like(W)
                for i in range(n):
                    for k in range(p):
                        A = np.zeros((m, R))
                        for j in range(m):
                            for r in range(R):
                                A[j, r] = U[i * m + j, r] * V[j * p + k, r]
                        
                        # Find min-norm solution and null space
                        try:
                            # Use lstsq to get x0
                            x0, residuals, rank_a, s = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            
                            # Get null space
                            if rank_a < R:
                                N = null_space(A)
                            else:
                                N = np.zeros((R, 0))
                            
                            best_w = None
                            # Try a few random perturbations in the null space
                            for _ in range(20):
                                if N.shape[1] > 0:
                                    coeffs = np.random.randn(N.shape[1]) * 0.5
                                    z = N @ coeffs
                                else:
                                    z = np.zeros(R)
                                
                                x_cand = x0 + z
                                w_cand = np.round(x_cand * 2) / 2
                                
                                # Check if w_cand satisfies A @ w_cand = 1
                                # Note: A and w_cand are half-integers/ints, so result should be exact
                                res = A @ w_cand
                                if np.allclose(res, np.ones(m), atol=1e-9):
                                    best_w = w_cand
                                    break
                            
                            if best_w is None:
                                # Fallback to min-norm rounded
                                best_w = np.round(x0 * 2) / 2
                                
                            W_new[k * n + i, :] = best_w
                        except:
                            pass
                
                W = W_new
                
                # Solve for V
                V_new = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        B = np.zeros((n, R))
                        for i in range(n):
                            for r in range(R):
                                B[i, r] = U[i * m + j, r] * W[k * n + i, r]
                        
                        try:
                            x0, residuals, rank_b, s = np.linalg.lstsq(B, np.ones(n), rcond=None)
                            
                            if rank_b < R:
                                N = null_space(B)
                            else:
                                N = np.zeros((R, 0))
                            
                            best_v = None
                            for _ in range(20):
                                if N.shape[1] > 0:
                                    coeffs = np.random.randn(N.shape[1]) * 0.5
                                    z = N @ coeffs
                                else:
                                    z = np.zeros(R)
                                
                                x_cand = x0 + z
                                v_cand = np.round(x_cand * 2) / 2
                                
                                res = B @ v_cand
                                if np.allclose(res, np.ones(n), atol=1e-9):
                                    best_v = v_cand
                                    break
                            
                            if best_v is None:
                                best_v = np.round(x0 * 2) / 2
                                
                            V_new[j * p + k, :] = best_v
                        except:
                            pass
                V = V_new
                
                # Solve for U
                U_new = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        C = np.zeros((p, R))
                        for k in range(p):
                            for r in range(R):
                                C[k, r] = V[j * p + k, r] * W[k * n + i, r]
                        
                        try:
                            x0, residuals, rank_c, s = np.linalg.lstsq(C, np.ones(p), rcond=None)
                            
                            if rank_c < R:
                                N = null_space(C)
                            else:
                                N = np.zeros((R, 0))
                            
                            best_u = None
                            for _ in range(20):
                                if N.shape[1] > 0:
                                    coeffs = np.random.randn(N.shape[1]) * 0.5
                                    z = N @ coeffs
                                else:
                                    z = np.zeros(R)
                                
                                x_cand = x0 + z
                                u_cand = np.round(x_cand * 2) / 2
                                
                                res = C @ u_cand
                                if np.allclose(res, np.ones(p), atol=1e-9):
                                    best_u = u_cand
                                    break
                            
                            if best_u is None:
                                best_u = np.round(x0 * 2) / 2
                                
                            U_new[i * m + j, :] = best_u
                        except:
                            pass
                U = U_new
                
                # Check exactness
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s_val = 0.0
                            for r in range(R):
                                s_val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                            if abs(s_val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if valid:
                    # Convert to lists
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}

    return best_result
