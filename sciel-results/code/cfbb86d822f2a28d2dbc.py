"""Matrix multiplication 2,4,6 - search-based tensor decomposition via nonlinear optimization."""
import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    start_time = time.time()
    deadline = start_time + 115

    # Tensor T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B
    # contribute to C[i,k]. In our convention:
    # U index: i*m + j  (A element)
    # V index: j*p + k  (B element)
    # W index: k*n + i  (C element, trace convention)
    # The decomposition requires: for each (i,j,k) with T[i,j,k]=1,
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # and for each (i,j,k) with T[i,j,k]=0, the sum must be 0.
    #
    # Actually, the tensor for matrix multiplication is:
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because every A[i,j] * B[j,k] contributes to C[i,k].
    # So the condition is simply:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all i,j,k
    #
    # This is a system of n*m*p = 48 equations in r*(n*m + m*p + p*n) = r*(8+24+12) = 44r unknowns.
    # For r=42, that's 1848 unknowns for 48 equations -- very underdetermined.
    #
    # Strategy: Start from a known valid decomposition (schoolbook rank 48),
    # then try to reduce rank by finding a lower-rank approximation.
    #
    # Alternative structural approach: Use the fact that the matrix multiplication
    # tensor has a Kronecker product structure. We can write:
    #   M_{n,m,p} = sum of terms involving Kronecker products of smaller tensors.
    #
    # For 2x4 x 4x6: Think of A as 2x(2*2), B as (2*2)x(2*3).
    # Actually, let's use a different partition:
    # A (2x4) = [A_0 | A_1] where A_0, A_1 are 2x2
    # B (4x6) = [B_0; B_1] where B_0, B_1 are 2x6
    # C = A_0 * B_0 + A_1 * B_1
    #
    # Now, 2x2 * 2x6: we can further decompose B_0 = [B_00 B_01 B_02] (three 2x2 blocks)
    # A_0 * B_0 = [A_0*B_00, A_0*B_01, A_0*B_02]
    # Each A_0*B_0t is 2x2*2x2, rank 7 by Strassen. Total for A_0*B_0: 21.
    # Similarly A_1*B_1: 21. Total: 42.
    #
    # To beat 42, we need to share multiplications between the two halves.
    #
    # NEW IDEA: Use Laderman's rank-7 algorithm for 3x3 matrix multiplication.
    # Embed our problem: pad A to 3x4, B to 4x6, compute 3x6 product, extract 2x6.
    # But 3x4*4x6: split 3x4 as 3x(3+1), 4x6 as (3+1)x(3+3).
    # This gives cross terms that are annoying.
    #
    # BETTER: Use the fact that rank(2x4*4x6) <= rank(3x4*4x6) and find
    # a low-rank decomposition for 3x4*4x6 that, when restricted to the
    # first 2 rows of A, gives a valid 2x4*4x6 decomposition.
    #
    # 3x4*4x6: schoolbook rank 72. Can we do better?
    # 3x4*4x6 = 3x(2*2) * (2*2)x(2*3).
    #
    # Actually, let me try a completely different structural approach:
    # Use the "border rank" or "secant variety" approach: the set of rank-r
    # decompositions is a variety, and we can use numerical algebraic geometry
    # to find points on it.
    #
    # For now, let me implement a practical search: start from the rank-42
    # Strassen decomposition, then try to reduce rank by 1, 2, 3, ... using
    # scipy.optimize to find a valid decomposition at the lower rank.
    #
    # The optimization: minimize sum of squared errors of the tensor equations
    # subject to the constraint that U, V, W have the correct shapes.
    
    def build_schoolbook():
        r = n * m * p  # 48
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    def build_strassen_42():
        rank = 42
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        def strassen_2x2(U, V, W, rank_offset, a_row, a_col, b_row, b_col, c_row, c_col):
            r = rank_offset
            U[a_row + a_col, r] += 1
            U[a_row+1 + a_col+1, r] += 1
            V[b_row + b_col, r] += 1
            V[b_row+1 + b_col+1, r] += 1
            W[c_row + c_col, r] += 1
            W[c_row+1 + c_col+1, r] += 1
            
            r = rank_offset + 1
            U[a_row+1 + a_col, r] += 1
            U[a_row+1 + a_col+1, r] += 1
            V[b_row + b_col, r] += 1
            W[c_row+1 + c_col, r] += 1
            
            r = rank_offset + 2
            U[a_row + a_col, r] += 1
            V[b_row + b_col+1, r] += 1
            V[b_row+1 + b_col+1, r] += -1
            W[c_row + c_col+1, r] += 1
            
            r = rank_offset + 3
            U[a_row+1 + a_col+1, r] += 1
            V[b_row+1 + b_col, r] += 1
            V[b_row + b_col, r] += -1
            W[c_row+1 + c_col, r] += 1
            
            r = rank_offset + 4
            U[a_row + a_col, r] += 1
            U[a_row + a_col+1, r] += 1
            V[b_row+1 + b_col+1, r] += 1
            W[c_row + c_col+1, r] += 1
            
            r = rank_offset + 5
            U[a_row+1 + a_col, r] += 1
            U[a_row + a_col, r] += -1
            V[b_row + b_col, r] += 1
            V[b_row + b_col+1, r] += 1
            W[c_row+1 + c_col+1, r] += 1
            
            r = rank_offset + 6
            U[a_row + a_col+1, r] += 1
            U[a_row+1 + a_col+1, r] += -1
            V[b_row+1 + b_col, r] += 1
            V[b_row+1 + b_col+1, r] += 1
            W[c_row + c_col, r] += 1
            
            return rank_offset + 7
        
        rank_offset = 0
        for t in range(3):
            rank_offset = strassen_2x2(U, V, W, rank_offset,
                                        a_row=0, a_col=0,
                                        b_row=0, b_col=2*t,
                                        c_row=0, c_col=2*t)
            rank_offset = strassen_2x2(U, V, W, rank_offset,
                                        a_row=0, a_col=2,
                                        b_row=2, b_col=2*t,
                                        c_row=0, c_col=2*t)
        return U, V, W

    def tensor_error(U, V, W):
        """Compute the maximum absolute error of the tensor equations."""
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        r = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def check_valid(U, V, W, tol=1e-6):
        return tensor_error(U, V, W) < tol

    # Start with the known rank-42 decomposition
    U42, V42, W42 = build_strassen_42()
    
    if not check_valid(U42, V42, W42):
        # Fallback to schoolbook
        U, V, W = build_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = 42
    best_U, best_V, best_W = U42, V42, W42
    
    # Try to reduce rank using optimization
    # For each target rank r < 42, try to find a valid decomposition
    for target_rank in range(41, 25, -1):
        if time.time() > deadline - 10:
            break
        
        # Initialize from the current best, truncated
        U_init = best_U[:, :target_rank] if best_U.shape[1] >= target_rank else np.zeros((n*m, target_rank))
        V_init = best_V[:, :target_rank] if best_V.shape[1] >= target_rank else np.zeros((m*p, target_rank))
        W_init = best_W[:, :target_rank] if best_W.shape[1] >= target_rank else np.zeros((p*n, target_rank))
        
        # Flatten for optimization
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def objective(x):
            U_f = x[:n*m*target_rank].reshape(n*m, target_rank)
            V_f = x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_f = x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
            return tensor_error(U_f, V_f, W_f)
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
            
            if result.success or result.fun < 1e-6:
                U_f = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_f = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_f = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                if check_valid(U_f, V_f, W_f, tol=1e-5):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_f, V_f, W_f
                    # Round to nearest half-integer if possible
                    U_r = np.round(U_f * 2) / 2
                    V_r = np.round(V_f * 2) / 2
                    W_r = np.round(W_f * 2) / 2
                    if check_valid(U_r, V_r, W_r, tol=1e-4):
                        best_U, best_V, best_W = U_r, V_r, W_r
                    break  # Found a valid lower rank, stop
        except Exception:
            pass
        
        if time.time() > deadline - 15:
            break
    
    # Final check
    if not check_valid(best_U, best_V, best_W, tol=1e-5):
        U, V, W = build_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
