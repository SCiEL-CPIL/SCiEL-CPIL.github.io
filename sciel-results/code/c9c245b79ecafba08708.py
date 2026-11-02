"""
Program: Structural schoolbook fallback + systematic rank-reduction search.
This version attempts to find a rank < 45 decomposition for 3x3 * 3x5 matrix multiplication.
It uses a randomized local search approach to solve the system of polynomial equations
defining the tensor decomposition. If no better solution is found within the time limit,
it falls back to the schoolbook algorithm (rank 45).
"""
import numpy as np
import time
import itertools
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Dimensions of the factor matrices
    dim_u = n * m  # 9
    dim_v = m * p  # 15
    dim_w = p * n  # 15
    
    # The tensor T has dimensions (dim_u, dim_v, dim_w).
    # T[u_idx, v_idx, w_idx] = 1 if the term A[i,j] B[j,k] contributes to C[i,k].
    # u_idx = i*m + j
    # v_idx = j*p + k
    # w_idx = k*n + i
    # The condition for contribution is always true for valid i,j,k in standard matmul.
    # So T[u_idx, v_idx, w_idx] = 1 for all valid triples (i,j,k).
    # However, the indices are coupled.
    # Let's construct the list of equations.
    # We need to find U, V, W such that:
    # sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r] = 1
    # for all valid (i,j,k).
    
    # There are n*m*p = 45 equations.
    # Each equation involves the product of one element from U, one from V, and one from W for each term r.
    
    # Let's try to search for a rank R < 45.
    # We can fix one matrix, say W, to have a specific structure to reduce degrees of freedom,
    # or just random initialize and optimize.
    
    # The problem is non-convex and highly sensitive.
    # A common strategy is to fix some entries to break symmetry and scale invariance.
    # Scale invariance: U_r * alpha, V_r / alpha, W_r * 1 for any alpha != 0.
    # We can fix W[w_idx, r] = 1 for a specific w_idx and r? No, that's too restrictive.
    # We can fix U[0, r] = 1 for all r? This fixes the scale of each term.
    # If U[0, r] = 0, we have a singularity.
    
    # Let's try a local search from a rank-45 solution, merging terms.
    # Or random restarts.
    
    best_rank = n * m * p
    best_solution = None
    
    # Precompute the index mapping for the equations
    # eq_idx maps (i, j, k) to the equation index
    # u_idx, v_idx, w_idx are derived from i, j, k
    eqs = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                eqs.append((u_idx, v_idx, w_idx))
                
    num_eqs = len(eqs)
    
    # Try ranks from 44 down to 30 (heuristic lower bound)
    # This is very slow, so we might only try a few ranks or a few iterations.
    
    def objective(params, R):
        """
        params: flattened array of size R * (dim_u + dim_v + dim_w)
        """
        U = params[:R*dim_u].reshape(R, dim_u)
        V = params[R*dim_u:R*(dim_u+dim_v)].reshape(R, dim_v)
        W = params[R*(dim_u+dim_v):].reshape(R, dim_w)
        
        residuals = []
        for idx, (u_idx, v_idx, w_idx) in enumerate(eqs):
            # sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
            val = np.sum(U[:, u_idx] * V[:, v_idx] * W[:, w_idx])
            residuals.append(val - 1.0)
            
        return residuals
    
    # To handle scale invariance, we can fix U[0, r] = 1 for all r.
    # This reduces the number of free variables.
    # Let's modify the parameterization.
    # U has shape (R, dim_u). We fix U[:, 0] = 1.
    # So we only optimize U[:, 1:], V, W.
    # Total variables: R * (dim_u - 1 + dim_v + dim_w)
    
    def objective_constrained(params, R):
        U_rest = params[:R*(dim_u-1)].reshape(R, dim_u-1)
        V = params[R*(dim_u-1):R*(dim_u-1+dim_v)].reshape(R, dim_v)
        W = params[R*(dim_u-1+dim_v):].reshape(R, dim_w)
        
        # Reconstruct U
        U = np.zeros((R, dim_u))
        U[:, 0] = 1.0
        U[:, 1:] = U_rest
        
        residuals = []
        for u_idx, v_idx, w_idx in eqs:
            val = np.sum(U[:, u_idx] * V[:, v_idx] * W[:, w_idx])
            residuals.append(val - 1.0)
            
        return residuals
    
    # Try to find a solution for R = 44
    # If we find one, we can try to reduce further.
    
    # Given the difficulty, let's try a few random initializations for R=44.
    # If it fails, we return schoolbook.
    
    found = False
    current_best_rank = best_rank
    current_best_U, current_best_V, current_best_W = None, None, None
    
    # Heuristic: The rank of 3x3*3x5 is likely 45 or 44?
    # Actually, 45 is the schoolbook.
    # Let's try R=44.
    
    # Time limit check
    if time.time() > deadline - 20:
        # Not enough time to search, return schoolbook
        pass
    else:
        for R in range(44, 40, -1):  # Try 44, 43, 42, 41
            if time.time() > deadline - 10:
                break
                
            num_vars = R * (dim_u - 1 + dim_v + dim_w)
            
            # Random initialization
            # Scale: values around 1.
            for attempt in range(5):
                if time.time() > deadline - 5:
                    break
                    
                # Initialize with small random values
                x0 = np.random.randn(num_vars) * 0.1
                # Add a small bias to avoid zeroing out U[:, 0] issues? No, U[:, 0] is fixed.
                # But V and W can be small.
                
                try:
                    result = least_squares(objective_constrained, x0, args=(R,), method='lm', max_nfev=1000)
                    if result.success or result.cost < 1e-8:
                        # Verify solution
                        U_rest = result.x[:R*(dim_u-1)].reshape(R, dim_u-1)
                        V = result.x[R*(dim_u-1):R*(dim_u-1+dim_v)].reshape(R, dim_v)
                        W = result.x[R*(dim_u-1+dim_v):].reshape(R, dim_w)
                        
                        U = np.zeros((R, dim_u))
                        U[:, 0] = 1.0
                        U[:, 1:] = U_rest
                        
                        # Check residuals
                        max_res = 0
                        for u_idx, v_idx, w_idx in eqs:
                            val = np.sum(U[:, u_idx] * V[:, v_idx] * W[:, w_idx])
                            max_res = max(max_res, abs(val - 1.0))
                        
                        if max_res < 1e-6:
                            found = True
                            current_best_rank = R
                            current_best_U = U
                            current_best_V = V
                            current_best_W = W
                            break
                except Exception:
                    continue
            
            if found:
                break
    
    if found:
        # Convert to list of lists
        U_list = current_best_U.tolist()
        V_list = current_best_V.tolist()
        W_list = current_best_W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
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
